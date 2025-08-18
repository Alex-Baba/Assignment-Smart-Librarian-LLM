import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "book_summaries.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    _SUMMARIES = json.load(f)

def get_summary_by_title(title: str) -> str:
    """Return the full summary for an exact title match."""
    return _SUMMARIES.get(title) or "No exact match found."

# OpenAI Responses API tool schema (function calling)
openai_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_summary_by_title",
            "description": "Return the full summary for an exact book title.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Exact book title"}
                },
                "required": ["title"]
            }
        }
    }
]