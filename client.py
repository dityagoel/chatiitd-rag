#!/usr/bin/env python3
"""
CLI WebSocket test client for the FastAPI RAG server.
Usage example:
    python client_cli.py --question "What is in the docs?" --url ws://localhost:8000/ws
"""
import asyncio
import argparse
import json
import sys
import uuid
import websockets

async def run(url: str, question: str, session_id: str | None) -> int:
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    try:
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps(payload))
            answer_parts = []
            while True:
                msg = await ws.recv()
                try:
                    data = json.loads(msg)
                except Exception:
                    sys.stdout.write(msg)
                    sys.stdout.flush()
                    continue
                t = data.get("type")
                if t == "token":
                    token = data.get("text", "")
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    answer_parts.append(token)
                elif t == "error":
                    print()
                    print(json.dumps(data, ensure_ascii=False))
                    return 1
                elif t == "done":
                    print()
                    answer = data.get("answer")
                    if answer is None:
                        answer = "".join(answer_parts)
                    print(answer)
                    out = {"sources": data.get("sources", []), "session_id": data.get("session_id")}
                    print(json.dumps(out, ensure_ascii=False))
                    return 0
                else:
                    print(json.dumps(data, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({"type": "client_error", "error": str(e)}, ensure_ascii=False))
        return 2

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://localhost:8000/ws")
    parser.add_argument("--question", required=True)
    parser.add_argument("--session-id", default=str(uuid.uuid4()))
    args = parser.parse_args()
    code = asyncio.run(run(args.url, args.question, args.session_id))
    sys.exit(code)

if __name__ == "__main__":
    main()
