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
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from qdrant_client.http.models import ScoredPoint
from langchain_core.documents import Document
import json
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import get_rules_section_tool, get_course_data_tool


# Load environment variables from a .env file
load_dotenv('../.env')
load_dotenv('.env')

class QdrantWithObjectPayload(Qdrant):
    """
    Custom Qdrant vector store class that handles object payloads.
    It serializes the 'page_content' into a JSON string if it is a dictionary.
    """
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
            # Using indent for better readability for the LLM
            page_content = json.dumps(page_content, indent=2)
        
        metadata = payload.get(metadata_payload_key) or {}
        
        # Add score and id to metadata, similar to the base class implementation
        metadata["_score"] = scored_point.score
        metadata["_id"] = scored_point.id

        return Document(
            page_content=page_content or "", # Ensure page_content is not None
            metadata=metadata,
        )


# --- 0. Setup ---
# Set up the necessary API keys. You will only need a Google API Key for Gemini.
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

# Initialize the LLM and Embeddings model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# --- 1. Connect to Existing Qdrant RAG Data Sources ---

# Initialize the Qdrant client to connect to your local instance.
client = qdrant_client.QdrantClient(url="http://localhost:6333")

# Connect to the existing 'rules' collection
rules_vector_store = QdrantWithObjectPayload(
    client=client,
    collection_name="rules",
    embeddings=embeddings,
    content_payload_key='content',
    metadata_payload_key='metadata'
)
rules_retriever = rules_vector_store.as_retriever()

# Connect to the existing 'courses' collection
courses_vector_store = QdrantWithObjectPayload(
    client=client,
    collection_name="courses",
    embeddings=embeddings,
    content_payload_key='description',
    metadata_payload_key='metadata'
)
courses_retriever = courses_vector_store.as_retriever()

# --- 2. Add a Free, Self-Run Reranking Step ---

# Initialize a free, self-run cross-encoder model from HuggingFace.
# The first time you run this, it will download the model weights (~227MB).
# This model runs locally on your machine (CPU or GPU if available).
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

# The compressor uses the cross-encoder model to rerank documents.
# It returns the top 3 most relevant documents.
compressor = CrossEncoderReranker(model=model, top_n=3)

# Create compression retrievers that will use the local reranker.
rules_compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=rules_retriever
)
courses_compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=courses_retriever
)


# --- 3. Define Tools with Refined Descriptions ---

# The descriptions are crucial. They guide the agent on when to use each tool.
# We will now use the compression retrievers in our tools.

rules_tool = create_retriever_tool(
    rules_compression_retriever,
    "search_iitd_rules",
    """
    Use this tool to answer questions about IIT Delhi's rules for undergraduate or postgraduate students.
    The documents contain headers such as:
    - Grading Policy
    - Degree Requirements
    - Code of Conduct
    - Academic Integrity
    - Hostel Regulations
    """,
)

courses_tool = create_retriever_tool(
    courses_compression_retriever,
    "search_iitd_courses",
    """
    Use this tool to find information about specific courses offered at IIT Delhi, but you want to search by topic or keywords rather than course code.
    """,
)

tools = [rules_tool, courses_tool, get_rules_section_tool, get_course_data_tool]


# --- 4. Create the Conversational Agent ---

# We'll use a prompt that supports chat history. This is suitable for tool-calling models like Gemini.
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant meant to assist students of Indian Institute of Technology Delhi with their academic queries. Respond to all irrelevant questions with 'Sorry, I can only respond to academic queries.' Prioritise gathering new context by calling tools. NEVER ASSUME ANY INFORMATION."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


print("--- IIT Delhi Academic Chatbot Initialized (Model: Gemini Flash, Reranker: BAAI/bge-reranker-base) ---")
print("Ask me about courses or institute rules.")
print("Type 'quit' to exit.")

# Initialize chat history
chat_history = []

# --- 5. Run the Agent in a Conversational Loop ---
while True:
    query = input("You: ")
    if query.lower() == "quit":
        break

    # The agent now takes both the input and the chat history
    response = agent_executor.invoke({
        "input": query,
        "chat_history": chat_history
    })

    # Append the latest interaction to the chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))

    print(f"Assistant: {response['output']}")
    print("\n" + "-"*50 + "\n")