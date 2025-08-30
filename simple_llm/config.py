import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv('../.env')

# Load environment variables or set default values
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# LiteLLM
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "gemini/gemini-1.5-flash")

# Add other configuration variables as needed