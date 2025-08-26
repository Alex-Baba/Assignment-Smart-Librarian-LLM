from __future__ import annotations
import base64
from typing import Optional
from openai import OpenAI

def _style_phrase(style: str) -> str:
    styles = {
        "Default": "",
        "Watercolor": "in a soft watercolor illustration style",
        "Dark fantasy": "in a dark fantasy cinematic concept-art style",
        "Whimsical": "in a cozy whimsical storybook illustration style",
        "Sci-fi neon": "as retro-futuristic sci-fi with neon lighting and glossy materials",
        "Minimalist": "as minimalist vector art with simple geometric shapes",
    }
    return styles.get(style, "")

def generate_book_image(
    title: str,
    summary: Optional[str] = "",
    style: str = "Default",
    size: str = "1024x1024",
    quality: str = "medium",
) -> bytes:
    """
    Returns raw image bytes for Streamlit (st.image accepts bytes).
    """
    client = OpenAI()

    prompt = (
        f"Book cover concept art for the novel '{title}'. "
        f"{('Plot gist: ' + summary[:500]) if summary else ''} "
        f"{_style_phrase(style)}. "
        "Beautiful, detailed, evocative. "
        "NO words, NO letters, NO typography, NO logos. "
        "Square composition, cinematic lighting."
    )

    resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size,
    )
    b64 = resp.data[0].b64_json
    return base64.b64decode(b64)