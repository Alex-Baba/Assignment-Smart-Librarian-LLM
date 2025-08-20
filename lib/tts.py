from __future__ import annotations
import tempfile
from openai import OpenAI

def tts_to_file(text: str, voice: str = "alloy") -> str:
    client = OpenAI()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    # Stream to disk (works across client versions)
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts", voice=voice, input=text
    ) as r:
        r.stream_to_file(tmp.name)
    return tmp.name