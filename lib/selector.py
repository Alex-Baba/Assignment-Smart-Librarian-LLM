from __future__ import annotations
import json, re
from typing import Any, Dict, List
from openai import OpenAI

SYSTEM_PROMPT = """You are Smart Librarian.
Given RAG_CONTEXT (a list of results with metadatas.title), pick ONE book.
Return ONLY JSON: {"title": "...", "why": "..."}.
- "title" MUST be copied VERBATIM from a metadatas.title in RAG_CONTEXT.
- "why" = 1–3 sentences using the user's request and context.
- Do not invent titles. If unsure, pick the first result's title verbatim.
"""

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

def snap_to_hits(candidate_title: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return candidate_title or ""
    allowed = [h["title"] for h in hits if h.get("title")]
    if not allowed:
        return candidate_title or ""
    if candidate_title in allowed:
        return candidate_title
    cand_n = _normalize(candidate_title)
    for t in allowed:
        if _normalize(t) == cand_n:
            return t
    for t in allowed:
        tn = _normalize(t)
        if tn in cand_n or cand_n in tn:
            return t
    return allowed[0]

def llm_select(user_query: str, hits: List[Dict[str, Any]], model_name: str) -> Dict[str, str]:
    client = OpenAI()
    resp = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
            {"role": "system", "content": "RAG_CONTEXT: " + json.dumps({"results": hits})},
        ],
    )
    txt = resp.output_text
    m = re.search(r"\{.*\}", txt, flags=re.S)
    data: Dict[str, Any] = {}
    if m:
        try:
            data = json.loads(m.group(0))
        except Exception:
            pass
    candidate = (data.get("title") if isinstance(data, dict) else "") or ""
    title = snap_to_hits(candidate, hits)
    why = (data.get("why") if isinstance(data, dict) else None) or "Top semantic match from your query."
    return {"title": title, "why": why}

__all__ = ["llm_select", "snap_to_hits"]