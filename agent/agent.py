import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Qdrant
import qdrant_client
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import AIMessage, HumanMessage
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from qdrant_client.http.models import ScoredPoint
from langchain_core.documents import Document
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from agent.tools import get_rules_section_tool, get_course_data_tool, get_programme_structure_tool, query_sqlite_db_tool, generate_degree_plan_tool
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from shared import config

# Load environment variables from a .env file
load_dotenv('../.env')
load_dotenv('.env')

# This class fills page_content with a json string of the entire document
# This is required for the ccourses collection
class QdrantWithObjectPayload(Qdrant):
    def _document_from_scored_point(
        self,
        scored_point: ScoredPoint,
        collection_name,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> Document:
        """Overrides the base method to handle object-like payloads."""
        payload = scored_point.payload or {}
        page_content = payload.get(content_payload_key)
                # If the content is a dictionary (i.e., a JSON object), serialize it
        if isinstance(page_content, dict):
            page_content = json.dumps(page_content, indent=2)        
        metadata = payload.get(metadata_payload_key) or {}
        # Add score and id to metadata, similar to the base class implementation
        metadata["_score"] = scored_point.score
        metadata["_id"] = scored_point.id
        return Document(
            page_content=page_content or "", # Ensure page_content is not None
            metadata=metadata,
        )

# Initialize the LLM and Embeddings model
llm = ChatGoogleGenerativeAI(model=config.llm_model, temperature=0,google_api_key=os.environ.get("GOOGLE_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)


# Connect to Existing Qdrant RAG Data Sources

client = qdrant_client.QdrantClient(url=config.qdrant_url)

rules_vector_store = QdrantWithObjectPayload(
    client=client,
    collection_name="rules",
    embeddings=embeddings,
    content_payload_key='content',
    metadata_payload_key='metadata'
)
rules_retriever = rules_vector_store.as_retriever(search_kwargs={'k': config.rag_top_k})

courses_vector_store = QdrantWithObjectPayload(
    client=client,
    collection_name="courses",
    embeddings=embeddings,
    content_payload_key='description',
    metadata_payload_key='metadata'
)
courses_retriever = courses_vector_store.as_retriever(search_kwargs={'k': config.rag_top_k})

# Reranking step
model = HuggingFaceCrossEncoder(model_name=config.reranking_model)
# top n documents from reranking scores
compressor = CrossEncoderReranker(model=model, top_n=config.rerank_top_n)

# these retrievers implement reranking
rules_compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=rules_retriever
)
courses_compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=courses_retriever
)


rules_tool = create_retriever_tool(
    rules_compression_retriever,
    "search_iitd_rules",
    """
    Use this tool to semantically search for certain queries about IIT Delhi's rules for undergraduate or postgraduate students.
    Use this tool when you cannot determine the section of rules to look up, or when the user query is more general.
    """,
)

courses_tool = create_retriever_tool(
    courses_compression_retriever,
    "search_iitd_courses",
    """
    Use this tool to find information about specific courses offered at IIT Delhi, but you want to search by topic or keywords rather than course code.
    If the course code is known, use the get_course_data_tool instead.
    """,
)

tools = [rules_tool, courses_tool, get_rules_section_tool, get_course_data_tool, get_programme_structure_tool, query_sqlite_db_tool, generate_degree_plan_tool]


with open(config.system_prompt_path, 'r') as file:
    system_prompt = file.read()
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def invoke_agent(input_dict):
    return agent_executor.invoke(input_dict)

runnable_agent = RunnableLambda(invoke_agent)
runnable_agent_with_history = RunnableWithMessageHistory(
    # runnable_agent,
    agent_executor,
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string=config.messages_conn_string
    ),
    input_messages_key="input",
    history_messages_key="chat_history"
)

def invoke_memory_agent(input_dict, session_id=None):
    if not session_id:
        return runnable_agent.invoke(input_dict)
    session_id = str(session_id)
    agent_config = {"configurable": {"session_id": session_id}}
    return runnable_agent_with_history.invoke(input_dict, config=agent_config)

# Initialize chat history
chat_history = []


def main():
    """Runs the agent in a conversational command-line loop."""
    print("--- IIT Delhi Academic Chatbot Initialized ---")
    print("Ask me about courses or institute rules.")
    print("Type 'quit' to exit.")

    # A unique session ID for the command-line interaction
    session_id = "cli_session"
    
    # --- 5. Run the Agent in a Conversational Loop ---
    while True:
        query = input("You: ")
        if query.lower() == "quit":
            break

        # The agent now takes both the input and a session_id
        response = invoke_memory_agent({
            "input": query
        }, session_id=session_id)

        print(f"Assistant: {response['output']}")
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
