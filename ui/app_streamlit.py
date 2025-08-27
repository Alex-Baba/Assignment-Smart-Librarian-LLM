from __future__ import annotations

import os
import io
import base64
from pathlib import Path
from typing import Dict, Any, Optional
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]  # /app
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Our project imports ---
from lib.config import AppConfig
from lib.vector import ensure_index, search_books, reset_and_rebuild
from lib.moderation import moderate_text, looks_like_bad_words

# Optional (if present). We'll fail gracefully if missing.
try:
    from lib.tts import tts_to_file as project_tts_to_file  # returns output filepath
except Exception:
    project_tts_to_file = None

# ---------- OpenAI client (for TTS & image generation fallbacks) ----------
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ---------- Constants / Paths ----------
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/.cache"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOICE_DESCRIPTIONS = {
    "alloy":   "Balanced, warm narrator. Neutral-lively tone; good for general summaries.",
    "verse":   "Soft, poetic cadence with gentle emphasis—ideal for lyrical blurbs.",
    "charlie": "Bright and upbeat; friendly guide vibe.",
    "sage":    "Calm and thoughtful, slightly formal; great for reflective prose.",
    "nova":    "Energetic modern tone; crisp pronunciation.",
}

IMAGE_STYLES = {
    "Cinematic": "cinematic lighting, shallow depth of field, dramatic mood",
    "Illustrated": "storybook illustration, ink and watercolor, whimsical",
    "Minimalist": "clean minimalist poster, bold shapes, limited palette",
    "Vintage": "retro book cover, aged paper texture, classic typography",
}

DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_IMAGE_MODEL = "gpt-image-1"
DEFAULT_IMAGE_SIZE = "1024x1024"


# ---------- Helpers ----------
def _ensure_key(k: str, default):
    if k not in st.session_state:
        st.session_state[k] = default
    return st.session_state[k]


def tts_to_file_openai_fallback(text: str, voice: str = "alloy", basename: Optional[str] = None) -> str:
    """Generate TTS using OpenAI directly; returns path to MP3 file."""
    basename = basename or "tts_output"
    out_path = OUTPUT_DIR / f"{basename}.mp3"
    # Streaming is robust in Streamlit/dockers
    with client.audio.speech.with_streaming_response.create(
        model=DEFAULT_TTS_MODEL,
        voice=voice,
        input=text,
        format="mp3",
    ) as response:
        response.stream_to_file(str(out_path))
    return str(out_path)


def generate_image_cover(prompt: str, style: str, basename: Optional[str] = None) -> str:
    """Generate a cover image using gpt-image-1; returns path to PNG file."""
    basename = basename or "cover"
    out_path = OUTPUT_DIR / f"{basename}.png"
    styled_prompt = f"Book cover concept. {prompt}\nStyle: {IMAGE_STYLES.get(style, style)}"
    resp = client.images.generate(
        model=DEFAULT_IMAGE_MODEL,
        prompt=styled_prompt,
        size=DEFAULT_IMAGE_SIZE,
    )
    b64 = resp.data[0].b64_json
    img_bytes = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    return str(out_path)


def recommend_top_book(query: str, cfg: AppConfig) -> Optional[Dict[str, Any]]:
    """RAG search then pick the closest match."""
    ensure_index(cfg)
    hits = search_books(query, k=3, cfg=cfg)
    if not hits:
        return None
    # hits already sorted by score asc (0=best)
    return hits[0]


# ---------- UI ----------
def main() -> None:
    cfg = AppConfig()
    st.set_page_config(page_title="Smart Librarian", layout="wide")
    st.title("📚 Smart Librarian")

    # Hide admin/debug by default. Enable with SMARTLIB_ADMIN=1
    admin = os.getenv("SMARTLIB_ADMIN", "0") == "1"

    # Auto reset once per session (quiet for normal users)
    if cfg.auto_reset and "auto_reset_done" not in st.session_state:
        with st.spinner("Preparing the library…"):
            try:
                _ = reset_and_rebuild(cfg)
            except Exception as e:
                st.warning(f"Index prepare notice: {e}")
        st.session_state["auto_reset_done"] = True

    # Conversation state
    _ensure_key("history", [])           # list of {"role": "user"/"assistant", "content": "..."}
    _ensure_key("assets", {})            # book_id -> {"image": path, "audio": path}
    _ensure_key("blocked_by_moderation", False)
    _ensure_key("last_reco", None)       # {"id","title","summary","score"}

    # Input area
    prompt = st.chat_input("Tell me what you’re in the mood for (genre, vibe, themes)…")
    if prompt:
        # Moderation: block recommendations if content is flagged
        flagged = False
        try:
            mod = moderate_text(prompt)
            flagged = bool(mod.get("flagged", False))
        except Exception:
            # Fallback heuristic if API not available
            flagged = looks_like_bad_words(prompt)

        if flagged:
            st.session_state["blocked_by_moderation"] = True
            st.chat_message("assistant").markdown(
                "⚠️ I sense rough language in thy request. "
                "Pray, phrase thy wish more gently, and I shall fetch thee a fitting tome."
            )
        else:
            st.session_state["blocked_by_moderation"] = False
            st.session_state["history"].append({"role": "user", "content": prompt})
            with st.spinner("Consulting the stacks…"):
                reco = recommend_top_book(prompt, cfg)
            if not reco:
                st.chat_message("assistant").markdown("I find no worthy match just yet. Try a different theme or author?")
            else:
                st.session_state["last_reco"] = reco
                st.session_state["history"].append(
                    {"role": "assistant",
                     "content": f"**{reco['title']}**\n\n{reco['summary']}"}
                )

    # Render history
    for turn in st.session_state["history"]:
        st.chat_message(turn["role"]).markdown(turn["content"])

    # If we’re blocked, stop here until a new, clean prompt arrives
    if st.session_state["blocked_by_moderation"]:
        return

    # After a recommendation, offer TTS & Image buttons
    reco = st.session_state.get("last_reco")
    if reco:
        st.divider()
        st.subheader("Enhance your recommendation")

        colA, colB = st.columns(2)

        # --- TTS panel ---
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
                basename = f"tts_{reco['id']}"
                try:
                    if project_tts_to_file:
                        audio_path = project_tts_to_file(text=text, voice=voice)
                    else:
                        audio_path = tts_to_file_openai_fallback(text=text, voice=voice, basename=basename)
                    # persist
                    assets = st.session_state["assets"].get(reco["id"], {})
                    assets["audio"] = audio_path
                    st.session_state["assets"][reco["id"]] = assets
                    st.success("Audio ready.")
                except Exception as e:
                    st.error(f"TTS failed: {e}")

            # If we already have audio, show it
            assets = st.session_state["assets"].get(reco["id"], {})
            if "audio" in assets:
                st.audio(assets["audio"])

        # --- Image panel ---
        with colB:
            st.markdown("**Generate a cover concept**")
            style = st.selectbox(
                "Cover style",
                list(IMAGE_STYLES.keys()),
                index=0,
                key=f"imgstyle_{reco['id']}",
            )
            st.caption(f"*{style}*: {IMAGE_STYLES[style]}")

            if st.button("🖼️ Generate image", key=f"img_btn_{reco['id']}"):
                # Build a concise image prompt from the summary
                prompt_img = (
                    f"Title: {reco['title']}. "
                    f"Design a striking book cover that evokes key themes from this summary: "
                    f"{reco['summary']}"
                )
                try:
                    img_path = generate_image_cover(prompt_img, style=style, basename=f"cover_{reco['id']}")
                    # persist
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
        with st.expander("Admin · Debug"):
            st.write("Session keys:", list(st.session_state.keys()))
            st.json({
                "last_reco": st.session_state.get("last_reco"),
                "assets": list(st.session_state.get("assets", {}).keys()),
            })
            st.caption("Set SMARTLIB_ADMIN=0 to hide this section.")


if __name__ == "__main__":
    main()