import os, re, json, tempfile
from pathlib import Path
import streamlit as st
from openai import OpenAI

from chatbot.rag import search_books, index_books
from chatbot.tools import get_summary_by_title
from chatbot.extras import (
    sanitized_or_warning,
    tts_to_file_openai,
    stt_from_file,
    generate_cover_png,
    VOICE_DESCRIPTIONS,
)

st.set_page_config(page_title="Smart Librarian", page_icon="📚")
st.title(" Smart Librarian")
st.caption("RAG (ChromaDB) + OpenAI + Tool call + TTS/STT & cover image")

with st.sidebar:
    st.subheader("Setup")
    api = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if api:
        os.environ["OPENAI_API_KEY"] = api

    st.text_input("GPT model", value=os.getenv("GPT_MODEL", "gpt-4o-mini"), key="model")

    # ---- Voice selector with descriptions
    st.markdown("### Voice")
    voices = list(VOICE_DESCRIPTIONS.keys())
    default_voice = os.getenv("VOICE", "alloy")
    idx = voices.index(default_voice) if default_voice in voices else 0
    voice_choice = st.selectbox(
        "Choose a voice",
        voices,
        index=idx,
        format_func=lambda v: f"{v} — {VOICE_DESCRIPTIONS.get(v,'')}"
    )
    os.environ["VOICE"] = voice_choice
    st.caption(f"Selected: **{voice_choice}** — {VOICE_DESCRIPTIONS.get(voice_choice,'')}")

    st.markdown("---")
    st.markdown("### Images (budget)")
    model_choice = st.selectbox("Image model", ["dall-e-3", "gpt-image-1"], index=0)
    size_choice = st.selectbox("Image size", ["512x512", "1024x1024"], index=0)
    quality_choice = st.selectbox("DALL·E 3 quality", ["standard", "hd"], index=0)
    st.caption("Tip: 512x512 + standard = lowest cost.")
    os.environ["IMAGE_MODEL"] = model_choice
    os.environ["IMAGE_SIZE"] = size_choice
    os.environ["IMAGE_QUALITY"] = quality_choice

    st.markdown("---")
    if st.button("(Re)Index books"):
        n = index_books()
        st.success(f"Indexed/updated {n} books.")

q = st.text_input("What do you feel like reading? (e.g., 'friendship and magic')")
if st.button("Recommend"):
    if not q.strip():
        st.warning("Type a query first.")
    else:
        clean = sanitized_or_warning(q)
        if clean is None:
            st.info("Let us keep a courteous tongue. Try a kinder query.")
            st.stop()
        q = clean

        hits = search_books(q, k=3)
        st.write("**Top matches (RAG):**", hits)

        client = OpenAI()
        system = ("You are Smart Librarian. Recommend a single best-fit book given RAG search results. "
                  "Return JSON with keys 'title' and 'why'.")
        resp = client.responses.create(
            model=os.getenv("GPT_MODEL", "gpt-4o-mini"),
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": q},
                {"role": "system", "content": "RAG_CONTEXT: " + json.dumps({"results": hits})}
            ]
        )
        txt = resp.output_text
        m = re.search(r"\{.*\}", txt, flags=re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = {"title": hits[0]["title"], "why": "Top semantic match from your query."}
        else:
            data = {"title": hits[0]["title"], "why": "Top semantic match from your query."}

        title = data.get("title", "")
        why = data.get("why", "")
        st.subheader(f"Recommendation: {title}")
        st.write(why)

        # ---- OpenAI TTS to MP3, then play
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
            try:
                voice = os.getenv("VOICE", "alloy")
                tts_to_file_openai(f"My pick is {title}. {why}", tf.name, voice=voice)
                st.audio(tf.name)
                st.caption(f"Voice: **{voice}** — {VOICE_DESCRIPTIONS.get(voice,'')}")
            except Exception:
                st.caption("TTS unavailable (optional).")

        # ---- Generate cover image (optional, on demand)
        if st.button("Generate cover image"):
            with st.spinner("Creating your cover..."):
                try:
                    out = generate_cover_png(title, why, out_path=f"cover_{title.replace(' ','_')}.png")
                    st.image(out, caption="AI-generated cover")
                    st.download_button("Download cover", data=open(out, "rb"), file_name=Path(out).name)
                except Exception:
                    st.warning("Could not generate cover image (optional).")

st.markdown("---")
st.subheader(" Ask by voice (upload a WAV/MP3)")
audio_file = st.file_uploader("Upload audio", type=["wav", "mp3"])
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix="."+audio_file.name.split(".")[-1]) as af:
        af.write(audio_file.read())
        path = af.name
    try:
        q_voice = stt_from_file(path)
        st.success(f"Transcribed: {q_voice}")
        st.session_state["q_voice"] = q_voice
    except Exception:
        st.error("Transcription failed. Check OPENAI_API_KEY.")

if st.session_state.get("q_voice"):
    st.info("Copy the transcribed text into the input above and click Recommend.")