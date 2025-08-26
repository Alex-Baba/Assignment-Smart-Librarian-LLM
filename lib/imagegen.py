from __future__ import annotations
import base64
from typing import Optional
from openai import OpenAI

def _style_phrase(style: str) -> str:
    """
    Map a style name to a descriptive phrase for the image prompt.
    Used to guide the AI image generation for different artistic styles.
    """
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
    Generate a book cover image using OpenAI's image API.
    Returns raw image bytes suitable for display in Streamlit.

    Args:
        title: Book title for the cover.
        summary: Optional plot summary for context.
        style: Artistic style for the image.
        size: Image size (e.g., "1024x1024").
        quality: Image quality (default "medium").

    Returns:
        Raw image bytes (PNG/JPEG) decoded from base64.
    """
    client = OpenAI()

    # Compose the prompt for the image generation model
    prompt = (
        f"Book cover concept art for the novel '{title}'. "
        f"{('Plot gist: ' + summary[:500]) if summary else ''} "
        f"{_style_phrase(style)}. "
        "Beautiful, detailed, evocative. "
        "NO words, NO letters, NO typography, NO logos. "
        "Square composition, cinematic lighting."
    )

    # Call OpenAI's image generation API
    resp = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size,
    )
    b64 = resp.data[0].b64_json
    return base64.b64decode(b64)