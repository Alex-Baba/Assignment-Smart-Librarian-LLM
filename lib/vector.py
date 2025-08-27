from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json, hashlib, re, os

import chromadb
from chromadb.utils import embedding_functions

from lib.config import AppConfig, CHROMA_OPENAI_API_KEY

# ---- basics ----
def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-") or "untitled"

def _stable_id(title: str) -> str:
    return hashlib.md5(title.encode("utf-8")).hexdigest()

def _load_summaries(p: Path) -> List[Dict[str, str]]:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return [{"Title": k, "Summary": v} for k, v in data.items()]
    out = []
    for it in data:
        if isinstance(it, dict):
            t = it.get("Title") or it.get("title")
            s = it.get("Summary") or it.get("summary")
            if t and s:
                out.append({"Title": t, "Summary": s})
    return out

# ---- chroma 1.x wiring ----
def _client(cfg: AppConfig) -> chromadb.PersistentClient:
    cfg.db_dir.mkdir(parents=True, exist_ok=True)
    # 1.x recommended local client
    return chromadb.PersistentClient(path=str(cfg.db_dir))

def _embedder(model_name: str):
    if not CHROMA_OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY / CHROMA_OPENAI_API_KEY not set")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=CHROMA_OPENAI_API_KEY,
        model_name=model_name,
    )

def _collection(cli: chromadb.PersistentClient, model_name: str):
    name = f"books_{_slug(model_name)}"
    # 1.x has get_or_create_collection
    return cli.get_or_create_collection(name=name, embedding_function=_embedder(model_name))

# ---- public API ----
def reset_db(cfg: AppConfig) -> None:
    # Clear the local store (duckdb+parquet under the dir)
    for p in cfg.db_dir.glob("*"):
        if p.is_file():
            p.unlink(missing_ok=True)
        else:
            for q in p.glob("**/*"):
                try: q.unlink()
                except: pass
            try: p.rmdir()
            except: pass

def index_books(cfg: AppConfig) -> int:
    cli = _client(cfg)
    col = _collection(cli, cfg.embed_model)
    items = _load_summaries(cfg.data_file)

    ids, docs, metas = [], [], []
    for it in items:
        title = it["Title"].strip()
        summary = it["Summary"].strip()
        ids.append(_stable_id(title))
        docs.append(summary)
        metas.append({"title": title, "summary": summary})

    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(ids)

def ensure_index(cfg: AppConfig) -> int:
    cli = _client(cfg)
    col = _collection(cli, cfg.embed_model)
    count = 0
    try:
        count = col.count()
    except Exception:
        count = 0
    if not count:
        return index_books(cfg)
    return count

def search_books(query: str, k: int, cfg: AppConfig) -> List[Dict[str, Any]]:
    cli = _client(cfg)
    col = _collection(cli, cfg.embed_model)
    res = col.query(
        query_texts=[query.strip()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    mds = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i in range(len(ids)):
        md = mds[i] or {}
        out.append({
            "id": ids[i],
            "title": md.get("title") or "(untitled)",
            "summary": md.get("summary") or docs[i],
            "score": dists[i],
        })
    out.sort(key=lambda r: r["score"])  # lower = closer
    return out

def reset_and_rebuild(cfg: AppConfig) -> int:
    reset_db(cfg)
    return index_books(cfg)