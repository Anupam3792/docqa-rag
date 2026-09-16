import os
from dotenv import load_dotenv

load_dotenv()

# Local LLM via Ollama — free, runs on your machine, no API key needed.
# Requires `ollama serve` running and the model pulled: `ollama pull llama3.2`
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
# When running in Docker, this points at the `ollama` service in docker-compose;
# locally it defaults to your machine's Ollama server.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # free, local, 384-dim
VECTOR_STORE_DIR = "vectorstore_index"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4
