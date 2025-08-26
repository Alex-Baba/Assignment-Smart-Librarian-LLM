from __future__ import annotations
import os, shutil
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions

from .data import load_summaries

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROMA_DIR = str(ROOT / "db_store")
CHROMA_DIR = os.environ.get("CHROMA_DIR", DEFAULT_CHROMA_DIR)

def client():
    """Create and return a persistent ChromaDB client for vector storage."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)

def collection(embed_model: str, name: str = "books"):
    """
    Get or create a ChromaDB collection for storing book embeddings.
    Uses OpenAI embedding model and API key from environment.
    """
    api_key = os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    ef = embedding_functions.OpenAIEmbeddingFunction(model_name=embed_model, api_key=api_key)
    c = client()
    try:
        return c.get_collection(name=name, embedding_function=ef)
    except Exception:
        return c.create_collection(name=name, embedding_function=ef)

def index_books(embed_model: str) -> int:
    """
    Index all books from the summaries file into the ChromaDB collection.
    Returns the number of books indexed.
    """
    data = load_summaries()
    col = collection(embed_model)
    ids, docs, metas = [], [], []
    for i, (title, summary) in enumerate(data.items()):
        ids.append(f"id_{i}")
        docs.append(summary)
        metas.append({"title": title, "summary": summary})
    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(ids)

def ensure_index(embed_model: str) -> int:
    """
    Ensure the ChromaDB collection is indexed.
    If empty, index all books. Returns the count of indexed books.
    """
    col = collection(embed_model)
    try:
        n = col.count()
    except Exception:
        n = 0
    if not n:
        return index_books(embed_model)
    return n

def reset_db() -> None:
    """
    Delete the ChromaDB storage directories to reset the vector database.
    """
    for d in [Path(CHROMA_DIR), ROOT / "chroma_db"]:
        if d.exists() and d.is_dir():
            shutil.rmtree(d, ignore_errors=True)

def reset_and_rebuild(embed_model: str) -> int:
    """
    Reset the vector database and rebuild the index from scratch.
    Returns the number of books indexed.
    """
    reset_db()
    return index_books(embed_model)

def search_books(query: str, k: int, embed_model: str) -> List[Dict[str, Any]]:
    """
    Search for the top-k most relevant books using semantic similarity.
    Returns a list of dicts with title, summary, document text, and score.
    """
    col = collection(embed_model)
    res = col.query(query_texts=[query], n_results=k, include=["documents", "metadatas", "distances"])
    hits: List[Dict[str, Any]] = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        doc  = res["documents"][0][i]
        dist = res.get("distances", [[None]])[0][i]
        hits.append({
            "title": meta.get("title", ""),
            "summary": meta.get("summary", ""),
            "doc": doc,
            "score": (1.0 - (dist or 0.0)) if dist is not None else None
        })
    return hits