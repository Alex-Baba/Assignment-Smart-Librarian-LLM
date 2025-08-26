from __future__ import annotations

# --- make project root importable when running from /ui ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# Import app modules for configuration, retrieval, selection, TTS, moderation, and image generation
from lib.config import setup_page, sidebar_config, is_admin_mode
from lib.vector import ensure_index, search_books, reset_and_rebuild
from lib.selector import llm_select
from lib.tts import tts_to_file
from lib.moderation import moderate_text, looks_like_bad_words
from lib.imagegen import generate_book_image

# ---------- helpers & state ----------

@st.cache_data(show_spinner=False)
def _img_cache(title: str, summary: str, style: str, size: str) -> bytes:
    """Cache image bytes to avoid regenerating the same prompt/style/size."""
    return generate_book_image(title, summary, style=style, size=size)

def _init_state() -> None:
    """Initialize Streamlit session state variables for recommendations and generated media."""
    st.session_state.setdefault("rec_ready", False)
    st.session_state.setdefault("rec_hits", [])
    st.session_state.setdefault("rec_title", "")
    st.session_state.setdefault("rec_why", "")
    st.session_state.setdefault("rec_summary", "")
    # persist generated media
    st.session_state.setdefault("tts_path", None)
    st.session_state.setdefault("img_bytes", None)
    st.session_state.setdefault("img_caption", "")

def _store_recommendation(hits, title, why) -> None:
    """Save the chosen recommendation and reset any previous media."""
    pick = next((h for h in hits if h.get("title") == title), hits[0] if hits else {"summary": ""})
    st.session_state.rec_hits = hits
    st.session_state.rec_title = title
    st.session_state.rec_why = why
    st.session_state.rec_summary = pick.get("summary", "")
    st.session_state.rec_ready = True
    # reset media for the fresh pick
    st.session_state.tts_path = None
    st.session_state.img_bytes = None
    st.session_state.img_caption = ""

def _render_actions() -> None:
    """
    Render post-recommendation actions:
    - Text-to-Speech audio generation
    - AI-generated cover image
    """
    title = st.session_state.rec_title
    why = st.session_state.rec_why
    summary_for_img = st.session_state.rec_summary

    st.markdown("### Do more with this pick:")
    col1, col2 = st.columns(2)

    # ---- TTS ----
    with col1:
        with st.expander("🔊 Text-to-Speech", expanded=False):
            voice = st.selectbox(
                "Voice",
                ["alloy", "verse", "charlie", "sage", "nova", "atticus"],
                index=0,
                key=f"voice_{title}",
            )
            if st.button("Generate audio", key=f"btn_tts_{title}"):
                try:
                    path = tts_to_file(f"My pick is {title}. {why}", voice=voice)
                    st.session_state.tts_path = path  # persist path
                except Exception as e:
                    st.caption(f"TTS unavailable: {e}")

            # Always render from state if present
            if st.session_state.tts_path:
                st.audio(st.session_state.tts_path)

    # ---- Image ----
    with col2:
        with st.expander("🎨 Generate cover image", expanded=False):
            style = st.selectbox(
                "Style",
                ["Default", "Watercolor", "Dark fantasy", "Whimsical", "Sci-fi neon", "Minimalist"],
                index=0,
                key=f"img_style_{title}",
            )
            size = st.selectbox(
                "Size",
                ["512x512", "768x768", "1024x1024"],
                index=2,
                key=f"img_size_{title}",
            )

            if st.button("Generate image", key=f"btn_img_{title}"):
                try:
                    with st.spinner("Generating cover image…"):
                        img_bytes = _img_cache(title, summary_for_img, style, size)
                    st.session_state.img_bytes = img_bytes
                    st.session_state.img_caption = f'AI-generated cover for “{title}” ({style}, {size})'
                except Exception as e:
                    st.caption(f"Image generation unavailable: {e}")

            # Always render from state if present
            if st.session_state.img_bytes:
                st.image(
                    st.session_state.img_bytes,
                    caption=st.session_state.img_caption,
                    use_container_width=True,
                )

# ---------- main app ----------

def main() -> None:
    """
    Main entry point for the Smart Librarian Streamlit app.
    Handles setup, query input, safety checks, recommendation logic, and post-actions.
    """
    setup_page()
    cfg, reset_now = sidebar_config()
    admin = is_admin_mode()
    _init_state()

    # Manual reset (admin-only visible via sidebar_config)
    if reset_now:
        with st.spinner("Resetting vector DB & reindexing…"):
            n = reset_and_rebuild(cfg.embed_model)
        st.success(f"Rebuilt index with {n} books.")
        # clear any stale recommendation + media
        st.session_state.rec_ready = False
        st.session_state.tts_path = None
        st.session_state.img_bytes = None
        st.session_state.img_caption = ""

    # Auto reset once per session (hidden default for users)
    if cfg.auto_reset and "auto_reset_done" not in st.session_state:
        with st.spinner("Resetting vector DB & reindexing…"):
            n = reset_and_rebuild(cfg.embed_model)
            if admin:
                st.success(f"Rebuilt index with {n} books.")
                st.session_state["auto_reset_done"] = True
                st.success(f"Fresh index built with {n} books.")

    # Query UI: user enters a reading preference or genre
    q = st.text_input("What do you feel like reading?", key="query_input")
    go = st.button("Recommend", key="btn_recommend")

    if go:
        # Validate input
        if not q.strip():
            st.warning("Type a query first.")
            st.stop()

        # Safety filter — always block if flagged; details only for admins
        if cfg.moderation_on:
            mod = moderate_text(q)
            if mod.get("flagged"):
                if looks_like_bad_words(mod.get("categories", {})):
                    st.error("⚠️ Heads-up: your message contains bad words. Please rephrase.")
                else:
                    st.warning("⚠️ Your message was flagged by our safety filter.")
                if admin:
                    with st.expander("Why was it flagged? (details)"):
                        st.write(mod.get("categories", {}))
                        st.caption("Powered by OpenAI Moderation.")
                st.stop()

        # Retrieve recommendations using semantic search
        ensure_index(cfg.embed_model)
        hits = search_books(q, k=cfg.top_k, embed_model=cfg.embed_model)

        # Show raw matches for admins
        if admin:
            st.write("**Top matches (RAG):**")
            st.json(hits, expanded=False)

        # Use LLM to select best match if enabled
        if cfg.use_llm:
            sel = llm_select(q, hits, cfg.text_model)
            title, why = sel["title"], sel["why"]
        else:
            title, why = hits[0]["title"], "Top semantic match from your query."

        st.markdown(f"## Recommendation: {title}")
        st.write(why)

        # Save to state so the post-actions work after rerun
        _store_recommendation(hits, title, why)

        # Render actions immediately on this run too
        _render_actions()

    # Reruns (e.g., clicked Generate audio/image): restore last rec & actions
    elif st.session_state.rec_ready:
        hits = st.session_state.rec_hits
        if admin:
            st.write("**Top matches (RAG):**")
            st.json(hits, expanded=False)

        title = st.session_state.rec_title
        why = st.session_state.rec_why
        st.markdown(f"## Recommendation: {title}")
        st.write(why)

        _render_actions()


if __name__ == "__main__":
    main()