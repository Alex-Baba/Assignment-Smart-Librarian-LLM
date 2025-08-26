from __future__ import annotations
from typing import Any, Dict
from openai import OpenAI

# Categories that usually indicate profanity/insults/slurs
BAD_WORD_KEYS = {
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "abusive",
}

def moderate_text(text: str) -> Dict[str, Any]:
    """
    Run OpenAI Moderation on the input text.
    Returns: {"flagged": bool, "categories": dict, "scores": dict, "raw": resp}
    """
    client = OpenAI()
    resp = client.moderations.create(model="omni-moderation-latest", input=text or "")
    result = resp.results[0] if getattr(resp, "results", None) else None
    if not result:
        return {"flagged": False, "categories": {}, "scores": {}, "raw": resp}
    return {
        "flagged": bool(result.flagged),
        "categories": dict(result.categories or {}),
        "scores": dict(result.category_scores or {}),
        "raw": resp,
    }

def looks_like_bad_words(categories: Dict[str, Any]) -> bool:
    """Heuristic: treat harassment/hate-like categories as 'bad words'."""
    return any(bool(categories.get(k)) for k in BAD_WORD_KEYS)

__all__ = ["moderate_text", "looks_like_bad_words", "BAD_WORD_KEYS"]