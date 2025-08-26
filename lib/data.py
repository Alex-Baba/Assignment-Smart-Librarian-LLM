from __future__ import annotations
import json
from pathlib import Path
from typing import Dict

# Set project root and path to book summaries JSON file
ROOT = Path(__file__).resolve().parents[1]
DATA_JSON = ROOT / "data" / "book_summaries.json"

def load_summaries() -> Dict[str, str]:
    """
    Load exactly what's in data/book_summaries.json, without altering titles.
    Supports:
      A) {"Title": "Summary", ...}  # Dict mapping titles to summaries
      B) [{"Title": "...", "Summary": "..."}, ...]  # List of dicts (case-insensitive)
    Returns:
        Dictionary mapping book titles to summaries.
    Raises:
        ValueError if the JSON shape is not supported.
    """
    raw = json.loads(DATA_JSON.read_text(encoding="utf-8"))

    # Case A: JSON is a dict of title -> summary
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}

    # Case B: JSON is a list of dicts with "title" and "summary" keys
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

    # If JSON shape is not recognized, raise an error
    raise ValueError("Unsupported JSON shape in book_summaries.json")