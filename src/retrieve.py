"""
Similarity search against the ChromaDB collection built by ingest.py.
"""

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "data/chroma_db"
COLLECTION_NAME = "arxiv_papers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_embedder = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def retrieve(query: str, top_k: int = 5):
    """
    Returns a list of dicts: [{"text": ..., "metadata": ..., "distance": ...}, ...]
    sorted by relevance (closest first).
    """
    embedder = _get_embedder()
    collection = _get_collection()

    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "metadata": meta, "distance": dist})
    return hits
