# app.py
import os
import asyncio
import json
import uuid
from typing import Dict, List, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain.chat_models import init_chat_model
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.callbacks import BaseCallbackHandler

APP = FastAPI()
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "my_docs")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
GOOGLE_KEY = os.environ.get("GOOGLE_API_KEY")

# In-memory session history: {session_id: [ {"role":"user"/"assistant", "text": "..."} , ... ] }
SESSIONS: Dict[str, List[Dict[str, str]]] = {}

class WebsocketCallbackHandler(BaseCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self._queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        # called for every new token
        self._queue.put_nowait({"type": "token", "text": token})

    def on_llm_end(self, response: Any, **kwargs) -> None:
        # signal end of stream
        self._queue.put_nowait({"type": "end"})

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        self._queue.put_nowait({"type": "error", "error": str(error)})

@APP.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        msg = await ws.receive_text()
        payload = json.loads(msg)
        question = payload.get("question") or ""
        session_id = payload.get("session_id") or str(uuid.uuid4())
        # initialize session history if missing
        history = SESSIONS.setdefault(session_id, [])

        # Create queue + callback handler
        queue: asyncio.Queue = asyncio.Queue()
        cb = WebsocketCallbackHandler(queue)

        # init LLM with streaming enabled
        llm = init_chat_model(
            model="gemini-2.5-flash",
            model_provider="google_genai",
            callbacks=[cb],
            disable_streaming=False,
        )

        # build retriever from Qdrant
        # embeddings builder is not needed for querying via LangChain Qdrant wrapper
        qdrant = Qdrant(
            client=QdrantClient(url=QDRANT_URL),
            collection_name=COLLECTION_NAME,
        )
        retriever = qdrant.as_retriever(search_kwargs={"k": 4})

        # conversational chain (memory handled here by passing history)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        # Kick off chain invocation in background so callbacks fire into the queue
        async def run_chain():
            inputs = {"question": question, "chat_history": history}
            # Use async predict/invoke; chain will call llm and callbacks will stream tokens
            await chain.apredict(inputs)

        chain_task = asyncio.create_task(run_chain())

        # stream tokens from queue to websocket
        while True:
            item = await queue.get()
            if item["type"] == "token":
                await ws.send_text(json.dumps({"type": "token", "text": item["text"]}))
            elif item["type"] == "error":
                await ws.send_text(json.dumps({"type": "error", "error": item["error"]}))
                break
            elif item["type"] == "end":
                break

        # Ensure chain finished
        await chain_task

        # After completion, you may want to compute final answer and sources
        # Here we run the chain synchronously one last time to get sources and final answer (safe fallback)
        result = await chain.apredict({"question": question, "chat_history": history})
        answer = result.get("answer") or str(result)
        sources = result.get("source_documents", [])

        # save into history
        history.append({"role": "user", "text": question})
        history.append({"role": "assistant", "text": answer})
        SESSIONS[session_id] = history[-50:]  # keep last 50 messages

        await ws.send_text(json.dumps({
            "type": "done",
            "answer": answer,
            "sources": [
                {"source": getattr(doc, "metadata", {}).get("source"), "page": getattr(doc, "metadata", {}).get("page")}
                for doc in sources
            ],
            "session_id": session_id
        }))
        await ws.close()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await ws.send_text(json.dumps({"type": "error", "error": str(e)}))
        await ws.close()

if __name__ == "__main__":
    uvicorn.run("app:APP", host="0.0.0.0", port=8000, reload=False)
