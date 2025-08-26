from __future__ import annotations
import os
from dataclasses import dataclass
import streamlit as st
from dotenv import load_dotenv

def is_admin_mode() -> bool:
    """
    Determine if admin mode is enabled.
    Advanced controls are visible if SMARTLIB_ADMIN=1 (env) or URL has ?admin=1.
    Returns True if admin mode is active, else False.
    """
    try:
        params = st.query_params()
        qp = (params.get("admin") or ["0"])[0]
    except Exception:
        qp = "0"
    env_flag = os.getenv("SMARTLIB_ADMIN", "0").lower() in ("1", "true", "yes", "on")
    url_flag = qp.lower() in ("1", "true", "yes", "on")
    return bool(env_flag or url_flag)

def setup_page() -> None:
    """
    Load environment variables and set up the Streamlit page.
    Sets page title, icon, and header.
    """
    load_dotenv()
    st.set_page_config(page_title="Smart Librarian", page_icon="📚")
    st.title("📚 Smart Librarian")
    #st.caption("RAG with ChromaDB + OpenAI — reads your JSON exactly as-is.")

@dataclass
class AppConfig:
    """
    Configuration dataclass for app settings.
    Stores API keys, model names, retrieval options, and safety settings.
    """
    api_key: str
    embed_model: str
    text_model: str
    top_k: int
    use_llm: bool
    auto_reset: bool
    # safety (hidden unless admin)
    moderation_on: bool
    moderation_block: bool

def sidebar_config() -> tuple[AppConfig, bool]:
    """
    Render sidebar controls for configuration.
    Shows advanced options only for admins.
    Returns the AppConfig and a flag for manual index reset.
    """
    admin = is_admin_mode()

    # Defaults for non-admin users (no sidebar UI shown)
    api = os.getenv("OPENAI_API_KEY", "")
    top_k = 5
    use_llm = True
    text_model = os.getenv("GPT_MODEL", "gpt-4o-mini")
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    moderation_on = True
    moderation_block = True
    auto_reset = True
    reset_now = False

    if admin:
        # Only admins see the sidebar sections
        with st.sidebar:
            st.subheader("Setup")
            api = st.text_input(
                "OPENAI_API_KEY",
                type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
                key="api_key_input",
            )
            st.caption("Admin mode: advanced options are visible.")

            st.markdown("### Retrieval & Selection")
            top_k = st.slider("Top-K results", 1, 5, 5, key="opt_top_k")
            use_llm = st.checkbox("Use LLM to choose among top-K", value=True, key="opt_use_llm")
            text_model = st.text_input("Text model", text_model, key="opt_model_text")
            embed_model = st.text_input("Embedding model", embed_model, key="opt_embed_model")

            st.markdown("### Safety")
            moderation_on = st.checkbox(
                "Enable safety filter (OpenAI Moderation)",
                value=True,
                key="opt_moderation_on"
            )
            moderation_block = st.checkbox(
                "Block request when flagged",
                value=True,
                key="opt_moderation_block"
            )

            st.markdown("### Index control")
            auto_reset = st.checkbox(
                "Auto-reset & reindex on app start (once per session)",
                value=True,
                key="opt_autoreset"
            )
            reset_now = st.button(
                "🔁 Reset index now (delete DB & rebuild)",
                key="btn_reset_now"
            )

    # Apply key for both modes (from env or admin input)
    if api:
        os.environ["OPENAI_API_KEY"] = api
        os.environ["CHROMA_OPENAI_API_KEY"] = os.getenv("CHROMA_OPENAI_API_KEY", api)

    cfg = AppConfig(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        embed_model=embed_model,
        text_model=text_model,
        top_k=top_k,
        use_llm=use_llm,
        auto_reset=auto_reset,
        moderation_on=moderation_on,
        moderation_block=moderation_block,
    )
    return cfg, reset_now