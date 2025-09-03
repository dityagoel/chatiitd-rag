from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
from sentence_transformers import SentenceTransformer
import litellm
import config
import re

# Connect to services
qdrant = QdrantClient(url=config.QDRANT_URL)
embedder = SentenceTransformer(config.EMBED_MODEL)

def query_bot(user_query: str, top_k: int = 5, show_sources: bool = False) -> str:
    print('Received query')
    """
    Query pipeline:
    1. Make the LLM write queries for
    """
    # Step 1: embed query
    query_vec = embedder.encode(user_query).tolist()

    # Step 2: search Qdrant
    results = qdrant.search(
        collection_name=config.QDRANT_COLLECTION,
        query_vector=query_vec,
        limit=top_k,
    )

    # look for course names in query and look them up directly
    course_pattern = r'\b([A-Z]{3}\d{3,4})\b'
    course_matches = re.findall(course_pattern, user_query)
    course_context = []
    for course in course_matches:
        course_results = qdrant.search(
            collection_name=config.QDRANT_COLLECTION,
            query_vector=embedder.encode(course).tolist(),
            query_filter= Filter(
                must=[
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=course)
                    )
                ]
            )
        )
        if course_results:
            course_context.append(course_results[0].payload)

    context = str([r.payload for r in results]) if results else "No relevant context found."
    course_context = str(course_context) if results else "No relevant context found."
    print('collected context')

    # Step 3: generate answer with Gemini
    # read prompt from prompt.txt
    prompt = ''
    with open('prompt.txt') as file:
        prompt = file.read().replace('{context}', context).replace('{course_context}', course_context).replace('{query}', user_query)

    # print('PROMPT ------\n', prompt)
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