from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

# --- Project root (works locally and in Docker: WORKDIR=/app) ---
ROOT = Path(__file__).resolve().parents[1]

# --- API keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Chroma can use its own key if provided; otherwise it falls back to OPENAI_API_KEY
CHROMA_OPENAI_API_KEY = os.getenv("CHROMA_OPENAI_API_KEY", OPENAI_API_KEY)

# --- Paths (Docker compose maps /app/data and /app/db_store) ---
DATA_FILE_DEFAULT = Path(os.getenv("DATA_FILE", "/app/data/book_summaries.json"))
DB_DIR_DEFAULT    = Path(os.getenv("CHROMA_PERSIST_DIR", "/app/db_store"))

# --- Model defaults ---
EMBED_MODEL_DEFAULT = os.getenv("EMBED_MODEL", "text-embedding-3-small")
TEXT_MODEL_DEFAULT  = os.getenv("TEXT_MODEL",  "gpt-4o-mini")
TTS_MODEL_DEFAULT   = os.getenv("TTS_MODEL",   "gpt-4o-mini-tts")

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() not in {"0", "false", "no"}

@dataclass
class AppConfig:
    # Core knobs
    api_key: str = OPENAI_API_KEY
    embed_model: str = EMBED_MODEL_DEFAULT
    text_model: str = TEXT_MODEL_DEFAULT
    top_k: int = int(os.getenv("TOP_K", "5"))
    use_llm: bool = _env_bool("USE_LLM", True)

    # App behavior
    auto_reset: bool = _env_bool("AUTO_RESET", True)          # reindex once on first run
    moderation_on: bool = _env_bool("MODERATION_ON", True)    # run moderation
    moderation_block: bool = _env_bool("MODERATION_BLOCK", True)  # block recos if flagged

    # Extras
    tts_model: str = TTS_MODEL_DEFAULT
    data_file: Path = DATA_FILE_DEFAULT
    db_dir: Path = DB_DIR_DEFAULT
    admin: bool = _env_bool("SMARTLIB_ADMIN", False)          # show admin/debug UI