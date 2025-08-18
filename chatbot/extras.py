import os, base64, tempfile
from typing import Optional

# --- Light politeness filter ---
BAD_WORDS = {"idiot", "stupid", "hate", "moron", "dumb"}

def is_impolite(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in BAD_WORDS)

def sanitized_or_warning(text: str) -> Optional[str]:
    return None if is_impolite(text) else text

# ---------- OPENAI TTS ----------
# Voices & short descriptions for the UI
VOICE_DESCRIPTIONS = {
    "alloy":   "Balanced, warm, neutral narrator; clean and clear for most content.",
    "verse":   "Lighter, friendly and expressive; great for casual or upbeat tones.",
    "charlie": "Softer, intimate, slightly breathy; nice for reflective passages.",
    "sage":    "Calm, measured, confident; good for factual or instructional text.",
    "nova":    "Bright and energetic; adds a touch of enthusiasm.",
    "atticus": "Deeper baritone; authoritative and steady.",
}

DEFAULT_VOICE = os.getenv("VOICE", "alloy")

# --- Text-to-Speech ---
def tts_to_file_openai(text: str, out_path: str = "speech.mp3", voice: str = None) -> str:
    """
    Generate speech using OpenAI TTS (gpt-4o-mini-tts) to a local file (mp3).
    """
    from openai import OpenAI
    client = OpenAI()
    voice = voice or os.getenv("VOICE", "alloy")

    # You may also set format="wav" if you prefer WAV
    resp = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text
    )
    with open(out_path, "wb") as f:
        f.write(resp.read())
    return out_path

def tts_say_cli(text: str):
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# --- Speech-to-Text ---
def stt_from_file(file_path: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    with open(file_path, "rb") as f:
        tr = client.audio.transcriptions.create(model="whisper-1", file=f)
    return tr.text.strip()

# --- Image generation ---
def generate_cover_png(title: str, why: str, out_path: str = None) -> str:
    """
    Budget-first image gen.
    - Uses DALL·E 3 by default (cheaper/predictable).
    - Defaults to 512x512.
    - Skips API call if file already exists.
    """
    from openai import OpenAI
    client = OpenAI()

    # Env-configurable
    model = os.getenv("IMAGE_MODEL", "dall-e-3")
    size = os.getenv("IMAGE_SIZE", "1024x1024")
    quality = os.getenv("IMAGE_QUALITY", "standard")  # 'standard' or 'hd'

    # Simple cache: reuse if already generated for this title
    safe = title.replace(" ", "_")
    out_path = out_path or f"cover_{safe}.png"
    if os.path.exists(out_path):
        return out_path

    prompt = (
        f"Minimal, modern book-cover style poster for '{title}'. "
        f"Symbolic imagery guided by: {why}. "
        "Clean composition; tasteful title text only; no extra body text."
    )

    if model == "dall-e-3":
        # DALL·E 3 supports quality + size
        img = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size=size,
            quality=quality
        )
    else:
        # fallback to gpt-image-1
        img = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size
        )

    b64 = img.data[0].b64_json
    raw = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(raw)
    return out_path