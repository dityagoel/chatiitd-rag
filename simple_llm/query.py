from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import litellm
import config

# Connect to services
qdrant = QdrantClient(url=config.QDRANT_URL)
embedder = SentenceTransformer(config.EMBED_MODEL)

def query_bot(user_query: str, top_k: int = 5, show_sources: bool = False) -> str:
    """
    Query pipeline:
    1. Embed user query with SentenceTransformer
    2. Search in Qdrant
    3. Send context + query to Gemini (via LiteLLM)
    """
    # Step 1: embed query
    query_vec = embedder.encode(user_query).tolist()

    # Step 2: search Qdrant
    results = qdrant.search(
        collection_name=config.QDRANT_COLLECTION,
        query_vector=query_vec,
        limit=top_k
    )

    context_chunks = [r.payload for r in results]
    context = str(context_chunks) if context_chunks else "No relevant context found."

    # Step 3: generate answer with Gemini
    prompt = f"""
You are a helpful college assistant. 
Use the context below to answer the user's question.

Context:
{context}

User query: {user_query}

Answer clearly and, if possible, point to specific sources.
"""
    print('PROMPT ------\n', prompt)
    response = litellm.completion(
        model=config.LITELLM_MODEL,   # e.g. "gemini/gemini-1.5-flash"
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["choices"][0]["message"]["content"]  # type: ignore
    print(answer)

    if show_sources:
        sources = [r.payload for r in results]
        return answer + "\n\n---\n📚 Sources:\n" + "\n".join(str(s) for s in sources)

    return answer