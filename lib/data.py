from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
DATA_JSON = ROOT / "data" / "book_summaries.json"

def load_summaries() -> Dict[str, str]:
    """
    Load exactly what's in data/book_summaries.json, without altering titles.
    Supports:
      A) {"Title": "Summary", ...}
      B) [{"Title": "...", "Summary": "..."}, ...]  (case-insensitive)
    """
    raw = json.loads(DATA_JSON.read_text(encoding="utf-8"))

    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}

    if isinstance(raw, list):
        out: Dict[str, str] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            lower = {str(k).lower(): v for k, v in item.items()}
            t, s = lower.get("title"), lower.get("summary")
            if t is None or s is None:
                continue
            out[str(t)] = str(s)
        return out

    raise ValueError("Unsupported JSON shape in book_summaries.json")