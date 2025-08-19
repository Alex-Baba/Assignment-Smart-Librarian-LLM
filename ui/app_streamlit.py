# ui/app_streamlit.py
import os, re, json, tempfile, sys
from pathlib import Path
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# --- make sure .env is loaded (project root) ---
load_dotenv()

# --- ensure project root is importable (so we can import `chatbot.*` and `tools.*`) ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- our project imports ---
from chatbot.rag import search_books, index_books, ensure_index
from tools.tools import get_summary_by_title
from chatbot.extras import (
    sanitized_or_warning,
    tts_to_file_openai,
    stt_from_file,
    generate_cover_png,
    VOICE_DESCRIPTIONS,
)

# ---------------- UI ----------------
st.set_page_config(page_title="Smart Librarian", page_icon="📚")
st.title("📚 Smart Librarian")
st.caption("RAG (ChromaDB) + OpenAI + Tool call + TTS/STT & cover image")

with st.sidebar:
    st.subheader("Setup")

    # 1) API key input (mask like password) → set BOTH env vars
    api = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if api:
        os.environ["OPENAI_API_KEY"] = api
        os.environ["CHROMA_OPENAI_API_KEY"] = api  # Chroma embedding helper expects this

    # small status line so you can see they are set
    st.caption(
        f"Keys → OPENAI: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'} | "
        f"CHROMA: {'✅' if os.getenv('CHROMA_OPENAI_API_KEY') else '❌'}"
    )

    # 2) Model choice for chat (optional; we still use gpt-4o-mini by default)
    st.text_input("GPT model", value=os.getenv("GPT_MODEL", "gpt-4o-mini"), key="model")

    # 3) Voice picker (OpenAI TTS)
    st.markdown("### Voice")
    voices = list(VOICE_DESCRIPTIONS.keys())
    default_voice = os.getenv("VOICE", "alloy")
    idx = voices.index(default_voice) if default_voice in voices else 0
    voice_choice = st.selectbox(
        "Choose a voice",
        voices,
        index=idx,
        format_func=lambda v: f"{v} — {VOICE_DESCRIPTIONS.get(v,'')}",
    )
    os.environ["VOICE"] = voice_choice
    st.caption(f"Selected: **{voice_choice}** — {VOICE_DESCRIPTIONS.get(voice_choice,'')}")

    # 4) Image settings (budget-first)
    st.markdown("---")
    st.markdown("### Images (budget)")
    model_choice = st.selectbox("Image model", ["dall-e-3", "gpt-image-1"], index=0)
    size_choice = st.selectbox("Image size", ["512x512", "1024x1024"], index=0)
    quality_choice = st.selectbox("DALL·E 3 quality", ["standard", "hd"], index=0)
    st.caption("Tip: 512x512 + standard = lowest cost.")
    os.environ["IMAGE_MODEL"] = model_choice
    os.environ["IMAGE_SIZE"] = size_choice
    os.environ["IMAGE_QUALITY"] = quality_choice

    # 5) Ensure the index exists (no-op if already built)
    try:
        ct = ensure_index()
        st.caption(f"Indexed docs: {ct}")
    except Exception as e:
        st.caption(f"Index error: {e}")

    # 6) Manual reindex button
    st.markdown("---")
    if st.button("(Re)Index books"):
        n = index_books()
        st.success(f"Indexed/updated {n} books.")

# ---------------- main interaction ----------------
# If a voice transcript exists from below, pre-fill the box
prefill = st.session_state.get("q_voice", "")
q = st.text_input("What do you feel like reading? (e.g., 'friendship and magic')", value=prefill)

if st.button("Recommend"):
    if not q.strip():
        st.warning("Type a query first.")
    else:
        clean = sanitized_or_warning(q)
        if clean is None:
            st.info("Let us keep a courteous tongue. Try a kinder query.")
            st.stop()
        q = clean

        # double-check index just before search
        ensure_index()

        # ---- RAG search
        hits = search_books(q, k=3)
        st.write("**Top matches (RAG):**", hits)

        client = OpenAI()
        system = (
            "You are Smart Librarian. Recommend a single best-fit book given RAG search results. "
            "Return JSON with keys 'title' and 'why'."
        )
        resp = client.responses.create(
            model=os.getenv("GPT_MODEL", "gpt-4o-mini"),
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": q},
                {"role": "system", "content": "RAG_CONTEXT: " + json.dumps({"results": hits})},
            ],
        )
        txt = resp.output_text
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = {"title": hits[0]["title"], "why": "Top semantic match from your query."} if hits else {"title": "", "why": ""}
        else:
            data = {"title": hits[0]["title"], "why": "Top semantic match from your query."} if hits else {"title": "", "why": ""}

        title = data.get("title", "")
        why = data.get("why", "")
        st.subheader(f"Recommendation: {title or '—'}")
        st.write(why or "No rationale available.")

        # ---- TTS to MP3, then play
        if title:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                try:
                    voice = os.getenv("VOICE", "alloy")
                    tts_to_file_openai(f"My pick is {title}. {why}", tf.name, voice=voice)
                    st.audio(tf.name)
                    st.caption(f"Voice: **{voice}** — {VOICE_DESCRIPTIONS.get(voice,'')}")
                except Exception as e:
                    st.caption(f"TTS unavailable (optional): {e}")

        # ---- Generate cover image (optional, on demand)
        if title and st.button("Generate cover image"):
            with st.spinner("Creating your cover..."):
                try:
                    out = generate_cover_png(title, why, out_path=f"cover_{title.replace(' ','_')}.png")
                    st.image(out, caption="AI-generated cover")
                    st.download_button("Download cover", data=open(out, "rb"), file_name=Path(out).name)
                except Exception as e:
                    st.warning(f"Could not generate cover image (optional): {e}")

st.markdown("---")
st.subheader("🎙️ Ask by voice (upload a WAV/MP3)")
audio_file = st.file_uploader("Upload audio", type=["wav", "mp3"])
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix="."+audio_file.name.split(".")[-1]) as af:
        af.write(audio_file.read())
        path = af.name
    try:
        q_voice = stt_from_file(path)
        st.success(f"Transcribed: {q_voice}")
        st.session_state["q_voice"] = q_voice
    except Exception as e:
        st.error(f"Transcription failed. Check OPENAI_API_KEY. Details: {e}")