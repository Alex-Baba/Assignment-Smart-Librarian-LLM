from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st

# --- Make root importable (Docker + local) ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]  # /app
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- App imports ---
from lib.config import AppConfig
from lib.vector import ensure_index, search_books, reset_and_rebuild, _client, _collection
from lib.moderation import moderate_text, looks_like_bad_words

from lib.tts import tts_to_file                        
from lib.imagegen import generate_book_image   
from lib.selector import llm_select        

# ---------- Constants / Paths ----------
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/.cache"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOICE_DESCRIPTIONS = {
    "alloy":   "Balanced and warm narrator; great default.",
    "verse":   "Soft, poetic cadence—lyrical feel.",
    "charlie": "Bright, upbeat, friendly.",
    "sage":    "Calm, reflective, slightly formal.",
    "nova":    "Energetic, crisp, modern.",
}

# Styles supported by your imagegen._style_phrase()
IMAGE_STYLE_OPTIONS = [
    "Default",
    "Watercolor",
    "Dark fantasy",
    "Whimsical",
    "Sci-fi neon",
    "Minimalist",
]

DEFAULT_IMAGE_SIZE = "1024x1024"
DEFAULT_IMAGE_QUALITY = "medium"


# ---------- Small helpers ----------
def _ensure_key(k: str, default):
    if k not in st.session_state:
        st.session_state[k] = default
    return st.session_state[k]

def recommend_top_book(query: str, cfg: AppConfig) -> Optional[Dict[str, Any]]:
    """RAG search, then optionally use LLM to choose among top-k."""
    ensure_index(cfg)
    hits = search_books(query, k=cfg.top_k, cfg=cfg)
    if not hits:
        return None

    # Default (vector-only) choice:
    chosen = {**hits[0], "why": "Top semantic match from your query."}

    if getattr(cfg, "use_llm", True):
        try:
            sel = llm_select(user_query=query, hits=hits, model_name=cfg.text_model)
            # snap selected title to actual hit
            title = sel.get("title") or chosen["title"]
            match = next((h for h in hits if h.get("title") == title), hits[0])
            chosen = {**match, "why": sel.get("why") or chosen["why"]}
        except Exception:
            # silently fall back to vector choice
            pass

    return chosen

def _save_image_bytes(b: bytes, basename: str) -> str:
    """Persist image bytes to disk and return path."""
    out = OUTPUT_DIR / f"{basename}.png"
    with open(out, "wb") as f:
        f.write(b)
    return str(out)


# ---------- UI ----------
def main() -> None:
    cfg = AppConfig()
    st.set_page_config(page_title="Smart Librarian", layout="wide")
    st.title("📚 Smart Librarian")

    admin = cfg.admin  # show admin/debug UI only if SMARTLIB_ADMIN=1

    # Conversation state
    _ensure_key("history", [])           # list of {"role": "user"/"assistant", "content": "..."}
    _ensure_key("assets", {})            # book_id -> {"image": path, "audio": path}
    _ensure_key("blocked_by_moderation", False)
    _ensure_key("last_reco", None)       # {"id","title","summary","score"}

    # Auto reset once per session (quiet for normal users)
    if cfg.auto_reset and "auto_reset_done" not in st.session_state:
        with st.spinner("Preparing the library…"):
            try:
                _ = reset_and_rebuild(cfg)
            except Exception as e:
                st.warning(f"Index prepare notice: {e}")
        st.session_state["auto_reset_done"] = True

    # Input
    prompt = st.chat_input("Tell me what you’re in the mood for (genre, vibe, themes)…")
    if prompt:
        # Moderation—block until next clean message
        flagged = False
        try:
            if cfg.moderation_on:
                mod = moderate_text(prompt)
                flagged = bool(mod.get("flagged", False))
        except Exception:
            flagged = looks_like_bad_words(prompt)

        if cfg.moderation_block and flagged:
            st.session_state["blocked_by_moderation"] = True
            st.chat_message("assistant").markdown(
                "⚠️ I sense rough language in your request. "
                "Please rephrase and try again."
            )
        else:
            st.session_state["blocked_by_moderation"] = False
            st.session_state["history"].append({"role": "user", "content": prompt})
            with st.spinner("Consulting the stacks…"):
                reco = recommend_top_book(prompt, cfg)
            if reco:
                st.session_state["last_reco"] = reco
                st.session_state["history"].append({
                    "role": "assistant",
                    "content": f"**{reco['title']}**\n\n{reco['summary']}\n\n*Why this book:* {reco.get('why','')}"
                })
            if not reco:
                st.chat_message("assistant").markdown(
                    "I find no worthy match yet. Try another genre, theme, or author?"
                )
            

    # Render history
    for turn in st.session_state["history"]:
        st.chat_message(turn["role"]).markdown(turn["content"])

    # Stop if blocked by moderation—no recos, no extras
    if st.session_state["blocked_by_moderation"]:
        return

    # Extras after a recommendation
    reco = st.session_state.get("last_reco")
    if reco:
        st.divider()
        st.subheader("Enhance your recommendation")

        colA, colB = st.columns(2, vertical_alignment="top")

        # --- TTS panel (uses your tts.py) ---
        with colA:
            st.markdown("**Listen to the summary**")
            voice = st.selectbox(
                "Choose a voice",
                list(VOICE_DESCRIPTIONS.keys()),
                index=0,
                help="Affects tone & cadence",
                key=f"voice_{reco['id']}",
            )
            st.caption(f"*{voice}*: {VOICE_DESCRIPTIONS[voice]}")

            if st.button("🔊 Generate audio", key=f"tts_btn_{reco['id']}"):
                text = f"{reco['title']}. {reco['summary']}"
                try:
                    audio_path = tts_to_file(text=text, voice=voice)
                    assets = st.session_state["assets"].get(reco["id"], {})
                    assets["audio"] = audio_path
                    st.session_state["assets"][reco["id"]] = assets
                    st.success("Audio ready.")
                except Exception as e:
                    st.error(f"TTS failed: {e}")

            assets = st.session_state["assets"].get(reco["id"], {})
            if "audio" in assets:
                st.audio(assets["audio"])

        # --- Image panel (uses your imagegen.py) ---
        with colB:
            st.markdown("**Generate a cover concept**")
            style = st.selectbox(
                "Cover style",
                IMAGE_STYLE_OPTIONS,
                index=0,
                key=f"imgstyle_{reco['id']}",
                help="Art direction for the generated cover"
            )
            st.caption("Image is generated by the AI; no real typography or logos are added.")

            if st.button("🖼️ Generate image", key=f"img_btn_{reco['id']}"):
                try:
                    raw = generate_book_image(
                        title=reco["title"],
                        summary=reco["summary"],
                        style=style,
                        size=DEFAULT_IMAGE_SIZE,
                        quality=DEFAULT_IMAGE_QUALITY,
                    )
                    img_path = _save_image_bytes(raw, basename=f"cover_{reco['id']}")
                    assets = st.session_state["assets"].get(reco["id"], {})
                    assets["image"] = img_path
                    st.session_state["assets"][reco["id"]] = assets
                    st.success("Image ready.")
                except Exception as e:
                    st.error(f"Image generation failed: {e}")

            assets = st.session_state["assets"].get(reco["id"], {})
            if "image" in assets:
                st.image(assets["image"], use_container_width=True)

    # Admin / debug (hidden by default)
    if admin:
        cli = _client(cfg)
        col = _collection(cli, cfg.embed_model)
    
        with st.expander("Admin · Debug"):
            st.write("Session keys:", list(st.session_state.keys()))
            st.json({
                "last_reco": st.session_state.get("last_reco"),
                "assets": list(st.session_state.get("assets", {}).keys()),
                "index_count": col.count(),
            })
            st.caption("Set SMARTLIB_ADMIN=0 to hide this section.")


if __name__ == "__main__":
    main()