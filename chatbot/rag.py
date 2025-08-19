import os, json
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

# Load environment variables from .env file
load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "book_summaries.json"
PERSIST_DIR = os.getenv("CHROMA_DIR", str(PROJECT_ROOT / "db_store"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

def _client():
    """
    Returns a persistent ChromaDB client for vector database operations.
    """
    return chromadb.PersistentClient(path=PERSIST_DIR)

def _collection(name: str = "books"):
    """
    Gets or creates a ChromaDB collection for storing book embeddings.
    Uses OpenAI embeddings with the specified model and API key.
    """
    api_key = os.getenv("CHROMA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    ef = embedding_functions.OpenAIEmbeddingFunction(
        model_name=EMBED_MODEL,
        api_key=api_key,
    )
    c = _client()
    try:
        return c.get_collection(name=name, embedding_function=ef)
    except Exception:
        return c.create_collection(name=name, embedding_function=ef)

# --- NEW: robust normalization for many JSON shapes ---
def _from_list(lst: List[Any]) -> Dict[str, str]:
    """
    Converts a list of book items (dicts or strings) to a normalized dict {title: summary}.
    Handles various possible JSON shapes for book data.
    """
    out: Dict[str, str] = {}
    for i, item in enumerate(lst):
        if isinstance(item, dict):
            # Try to extract title and summary from known keys
            title = item.get("title") or item.get("name") or f"Book {i+1}"
            summary = (
                item.get("summary")
                or item.get("description")
                or item.get("synopsis")
                or ""
            )
            if not summary:
                # Fallback: use any long string value as summary
                for v in item.values():
                    if isinstance(v, str) and len(v.split()) > 5:
                        summary = v; break
        else:
            # If not a dict, treat as string
            title = f"Book {i+1}"
            summary = str(item)
        out[title] = summary or "(no summary provided)"
    return out

def load_summaries(path: Path = DATA_PATH) -> Dict[str, str]:
    """
    Loads and normalizes book summaries from a JSON file.
    Supports dicts of title->summary, lists of dicts, or lists of strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Case 1: already dict of title->summary
    if isinstance(raw, dict):
        # Sometimes wrapped like {"books": [...]}
        if "books" in raw and isinstance(raw["books"], list):
            return _from_list(raw["books"])
        # Otherwise assume {title: summary}
        return {str(k): str(v) for k, v in raw.items()}

    # Case 2: list of items
    if isinstance(raw, list):
        return _from_list(raw)

    raise ValueError("Unsupported JSON shape for book_summaries.json")

def index_books(json_path: Path = DATA_PATH, collection_name: str = "books") -> int:
    """
    Indexes all books from the summaries file into the ChromaDB collection.
    Returns the number of books indexed.
    """
    data = load_summaries(json_path)  # normalized dict {title: summary}
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
    Searches for the top-k most relevant books using semantic similarity.
    Returns a list of dicts with title, summary, document text, and score.
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

# --- NEW: safe helper your UI can call before first query ---
def ensure_index(collection_name: str = "books") -> int:
    """
    Ensures the ChromaDB collection is indexed.
    If empty, indexes all books. Returns the count of indexed books.
    """
    col = _collection(collection_name)
    try:
        count = col.count()
    except Exception:
        count = 0
    if not count:
        return index_books(collection_name=collection_name)
    return count