import os, json
from pathlib import Path
from typing import Dict, List, Any
import chromadb
from chromadb.utils import embedding_functions

# rag.py: Retrieval-Augmented Generation (RAG) module for Smart Librarian LLM
# This module provides functions for document retrieval, embedding, and context generation for chatbot responses.

# Paths & env
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "book_summaries.json"
PERSIST_DIR = os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "db_store"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# Helper functions for ChromaDB client and collection

def _client():
    """
    Create and return a persistent ChromaDB client.
    """
    return chromadb.PersistentClient(path=PERSIST_DIR)

def _collection(name: str = "books"):
    """
    Get or create a ChromaDB collection with OpenAI embedding function.
    """
    ef = embedding_functions.OpenAIEmbeddingFunction(model_name=EMBED_MODEL)
    c = _client()
    try:
        return c.get_collection(name=name, embedding_function=ef)
    except Exception:
        return c.create_collection(name=name, embedding_function=ef)

def load_summaries(path: Path = DATA_PATH) -> Dict[str, str]:
    """
    Load book summaries from a JSON file.
    Args:
        path (Path): Path to the JSON file.
    Returns:
        Dict[str, str]: Dictionary of title to summary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def index_books(json_path: Path = DATA_PATH, collection_name: str = "books") -> int:
    """
    Index all book summaries into the ChromaDB collection.
    Args:
        json_path (Path): Path to the JSON file.
        collection_name (str): Name of the ChromaDB collection.
    Returns:
        int: Number of indexed books.
    """
    data = load_summaries(json_path)
    col = _collection(collection_name)
    ids, docs, metas = [], [], []
    for i, (title, summary) in enumerate(data.items()):
        ids.append(f"id_{i}")
        docs.append(summary)
        metas.append({"title": title, "summary": summary})
    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(ids)

def search_books(query: str, k: int = 3, collection_name: str = "books") -> List[Dict[str, Any]]:
    """
    Search for the top-k most relevant book summaries for a given query using ChromaDB.
    Args:
        query (str): User query.
        k (int): Number of top relevant summaries to return.
        collection_name (str): Name of the ChromaDB collection.
    Returns:
        List[Dict[str, Any]]: List of dictionaries with book info and relevance score.
    """
    col = _collection(collection_name)
    res = col.query(query_texts=[query], n_results=k, include=["documents", "metadatas", "distances"])
    hits: List[Dict[str, Any]] = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i]
        doc = res["documents"][0][i]
        dist = res.get("distances", [[None]])[0][i]
        hits.append({
            "title": meta["title"],
            "summary": meta["summary"],
            "doc": doc,
            "score": (1.0 - (dist or 0.0)) if dist is not None else None
        })
    return hits