from __future__ import annotations
import os
from dataclasses import dataclass
import streamlit as st
from dotenv import load_dotenv

def setup_page() -> None:
    """Set Streamlit page metadata and header."""
    load_dotenv()
    st.set_page_config(page_title="Smart Librarian", page_icon="📚")
    st.title("📚 Smart Librarian")
    st.caption("RAG with ChromaDB + OpenAI — reads your JSON exactly as-is.")

@dataclass
class AppConfig:
    api_key: str
    embed_model: str
    text_model: str
    top_k: int
    use_llm: bool
    use_tts: bool
    voice: str
    auto_reset: bool

def sidebar_config() -> tuple[AppConfig, bool]:
    """Render sidebar controls and return (config, reset_now_clicked)."""
    with st.sidebar:
        st.subheader("Setup")
        api = st.text_input(
            "OPENAI_API_KEY",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            key="api_key_input",
        )
        if api:
            os.environ["OPENAI_API_KEY"] = api
            os.environ["CHROMA_OPENAI_API_KEY"] = os.getenv("CHROMA_OPENAI_API_KEY", api)

        st.markdown("### Retrieval & Selection")
        top_k = st.slider("Top-K results", 1, 5, 3, key="opt_top_k")
        use_llm = st.checkbox("Use LLM to choose among top-K", value=True, key="opt_use_llm")
        text_model = st.text_input("Text model", os.getenv("GPT_MODEL", "gpt-4o-mini"), key="opt_model_text")
        embed_model = st.text_input("Embedding model", os.getenv("EMBED_MODEL", "text-embedding-3-small"), key="opt_embed_model")

        st.markdown("### Optional: Text-to-Speech")
        use_tts = st.checkbox("Speak the recommendation", value=False, key="opt_use_tts")
        voice = st.selectbox("Voice", ["alloy", "verse", "charlie", "sage", "nova", "atticus"], index=0, key="opt_voice")

        st.markdown("### Index control")
        auto_reset = st.checkbox("Auto-reset & reindex on app start (once per session)", value=True, key="opt_autoreset")
        reset_now = st.button("🔁 Reset index now (delete DB & rebuild)", key="btn_reset_now")

    cfg = AppConfig(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        embed_model=embed_model,
        text_model=text_model,
        top_k=top_k,
        use_llm=use_llm,
        use_tts=use_tts,
        voice=voice,
        auto_reset=auto_reset,
    )
    return cfg, reset_now