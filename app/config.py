import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI is used only for the final answer-generation step.
# Everything else (embeddings, chunking, retrieval) runs fully local/free.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # free, local, 384-dim
VECTOR_STORE_DIR = "vectorstore_index"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4
