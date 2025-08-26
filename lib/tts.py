from __future__ import annotations
import tempfile
from openai import OpenAI

def tts_to_file(text: str, voice: str = "alloy") -> str:
    """
    Generate speech audio from text using OpenAI's TTS API.
    Streams the audio to a temporary MP3 file and returns its path.

    Args:
        text: The text to synthesize.
        voice: The voice model to use (default: "alloy").

    Returns:
        Path to the generated MP3 file.
    """
    client = OpenAI()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.close()
    # Stream audio response to disk (compatible with OpenAI client versions)
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts", voice=voice, input=text
    ) as r:
        r.stream_to_file(tmp.name)
    return tmp.name