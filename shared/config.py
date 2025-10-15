# Secrets are stored in .env, check that out as well

# AGENT CONFIG

llm_model = "gemini-2.5-flash"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
reranking_model = "BAAI/bge-reranker-base"
qdrant_url = "http://localhost:6333"
rag_top_k = 25
rerank_top_n = 5
system_prompt_path = 'agent/system_prompt.txt'
messages_conn_string = "sqlite:///agent/messages.db"
all_rules_path = 'agent/sources/jsonl/all_rules.jsonl'
courses_jsonl_path = 'agent/sources/jsonl/courses.jsonl'
offered_jsonl_path = 'agent/sources/jsonl/courses_offered.jsonl'
programme_structures_folder_path = 'agent/sources/programme_structures'
courses_db_conn_string = 'file:agent/courses.sqlite?mode=ro'