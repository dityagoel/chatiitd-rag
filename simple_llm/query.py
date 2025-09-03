from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue
from sentence_transformers import SentenceTransformer
import litellm
import config
import re
import json

# Connect to services
qdrant = QdrantClient(url=config.QDRANT_URL)
embedder = SentenceTransformer(config.EMBED_MODEL)

def query_bot(user_query: str, top_k: int = 5, show_sources: bool = False) -> str:
    print('Received query')
    # Generate search terms using LLM
    init_prompt = ''
    with open('init_prompt.txt') as file:
        init_prompt = file.read()#.replace('{user_query}', user_query)

    # print('PROMPT ------\n', prompt)
    response = litellm.completion(
        model=config.LITELLM_MODEL,   # e.g. "gemini/gemini-1.5-flash"
        messages=[{"role": "system", "content": init_prompt}, {"role": "user", "content": user_query}]
    )
    lines = str(response["choices"][0]["message"]["content"]).split('\n')
    lines = [l for l in lines if l.strip() != '' and not l.startswith('```')]

    rules_query = ''
    courses_query = ''
    try:
        search_terms = json.loads(''.join(lines))
        print("Generated search terms:", search_terms)
        if search_terms.get('irrelevant_query', False):
            return "The question is not relevant to IIT Delhi academics."
        rules_query = search_terms.get('cos_query', None)
        courses_query = search_terms.get('course_search', None)
        
    except json.JSONDecodeError as e:
        err = "Error parsing JSON from LLM response:" + str(e)
        print(err)
        return err

    # Embed queries and search in the database
    q_rules = embedder.encode(rules_query).tolist() if rules_query else []
    q_courses = embedder.encode(courses_query).tolist() if courses_query else []

    # search Qdrant
    rules_results = qdrant.search(
        collection_name='rules',
        query_vector=q_rules,
        limit=top_k,
    ) if rules_query else None
    courses_results = qdrant.search(
        collection_name='courses',
        query_vector=q_courses,
        limit=top_k,
    ) if courses_query else None

    rules_context = str([r.payload for r in rules_results]) if rules_results else "No relevant context found."
    courses_context = str([r.payload for r in courses_results]) if courses_results else "No relevant context found."
    print('collected context')

    # generate answer with Gemini
    # read prompt from prompt.txt
    prompt = ''
    with open('prompt.txt') as file:
        prompt = file.read().replace('{courses_context}', courses_context).replace('{rules_context}', rules_context).replace('{query}', user_query)

    # print('PROMPT ------\n', prompt)
    response = litellm.completion(
        model=config.LITELLM_MODEL,   # e.g. "gemini/gemini-1.5-flash"
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["choices"][0]["message"]["content"]  # type: ignore
    print(answer)

    return answer