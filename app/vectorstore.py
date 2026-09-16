from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR, TOP_K

_embeddings = None
_vector_store = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


def get_vector_store():
    """Loads the persisted FAISS index once and caches it in memory."""
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISS.load_local(
            VECTOR_STORE_DIR,
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vector_store


def retrieve(query: str, k: int = TOP_K):
    """Returns the top-k most similar chunks for a query, with scores."""
    store = get_vector_store()
    results = store.similarity_search_with_score(query, k=k)
    return [
        {"content": doc.page_content, "source": doc.metadata.get("source", "unknown"), "score": float(score)}
        for doc, score in results
    ]
