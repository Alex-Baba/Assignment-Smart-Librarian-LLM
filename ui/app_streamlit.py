from __future__ import annotations

# Make repo root importable when running from /ui
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from lib.config import setup_page, sidebar_config
from lib.vector import ensure_index, search_books, reset_and_rebuild
from lib.selector import llm_select
from lib.tts import tts_to_file
from lib.moderation import moderate_text, looks_like_bad_words


def main() -> None:
    setup_page()
    cfg, reset_now = sidebar_config()

    # Manual reset
    if reset_now:
        with st.spinner("Resetting vector DB & reindexing…"):
            n = reset_and_rebuild(cfg.embed_model)
        st.success(f"Rebuilt index with {n} books.")

    # Auto reset once per session
    if cfg.auto_reset and "auto_reset_done" not in st.session_state:
        with st.spinner("Resetting vector DB & reindexing…"):
            n = reset_and_rebuild(cfg.embed_model)
        st.session_state["auto_reset_done"] = True
        st.success(f"Fresh index built with {n} books.")

    # Query UI
    q = st.text_input(
        "What do you feel like reading? (e.g., 'found family fantasy')",
        key="query_input",
    )
    go = st.button("Recommend", key="btn_recommend")
    if not go:
        return
    if not q.strip():
        st.warning("Type a query first.")
        return

    # Safety filter: warn/block if the user used bad words
    mod = None
    if getattr(cfg, "moderation_on", True):
        mod = moderate_text(q)
        if mod.get("flagged"):
            if looks_like_bad_words(mod.get("categories", {})):
                st.error("⚠️ Heads-up: your message contains bad words. Please rephrase.")
            else:
                st.warning("⚠️ Your message was flagged by our safety filter.")
            with st.expander("Why was it flagged? (details)"):
                st.write(mod.get("categories", {}))
                st.caption("Powered by OpenAI Moderation.")

            # Clear any prior recommendation and STOP this run
            # result_area.empty()  # Uncomment if result_area is defined
            st.stop()

    # Ensure index & retrieve
    ensure_index(cfg.embed_model)
    hits = search_books(q, k=cfg.top_k, embed_model=cfg.embed_model)

    st.write("**Top matches (RAG):**")
    st.json(hits, expanded=False)

    # Choose best (English output)
    if cfg.use_llm:
        sel = llm_select(q, hits, cfg.text_model)
        title, why = sel["title"], sel["why"]
    else:
        title, why = hits[0]["title"], "Top semantic match from your query."

    st.markdown(f"## Recommendation: {title}")
    st.write(why)

    # TTS
    if cfg.use_tts:
        try:
            path = tts_to_file(f"My pick is {title}. {why}", voice=cfg.voice)
            st.audio(path)
        except Exception as e:
            st.caption(f"TTS unavailable: {e}")


if __name__ == "__main__":
    main()