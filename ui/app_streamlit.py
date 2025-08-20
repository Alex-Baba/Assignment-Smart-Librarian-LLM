from __future__ import annotations
import streamlit as st

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.config import setup_page, sidebar_config, AppConfig
from lib.vector import ensure_index, search_books, reset_and_rebuild
from lib.selector import llm_select
from lib.tts import tts_to_file

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
    q = st.text_input("What do you feel like reading? (e.g., 'found family fantasy')", key="query_input")
    go = st.button("Recommend", key="btn_recommend")
    if not go:
        return
    if not q.strip():
        st.warning("Type a query first.")
        return

    ensure_index(cfg.embed_model)

    # RAG search
    hits = search_books(q, k=cfg.top_k, embed_model=cfg.embed_model)
    st.write("**Top matches (RAG):**")
    st.json(hits, expanded=False)

    # Choose best
    if cfg.use_llm:
        sel = llm_select(q, hits, cfg.text_model)
        title, why = sel["title"], sel["why"]
    else:
        title, why = hits[0]["title"], "Top semantic match from your query."

    st.markdown(f"## Recommendation: {title}")
    st.write(why)

    # Optional TTS
    if cfg.use_tts:
        try:
            path = tts_to_file(f"My pick is {title}. {why}", voice=cfg.voice)
            st.audio(path)
        except Exception as e:
            st.caption(f"TTS unavailable: {e}")

if __name__ == "__main__":
    main()