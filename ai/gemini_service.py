from __future__ import annotations

import base64
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def gemini_api_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or "").strip()


def gemini_model() -> str:
    return (os.getenv("GEMINI_IMAGE_MODEL") or "gemini-3.1-flash-image").strip()


def generate_image(prompt: str, output_path: str) -> str:
    """Generate an image with Gemini native image output and save it as PNG."""
    key = gemini_api_key()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured in .env")

    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed. Run: pip install -r requirements.txt") from exc

    client = genai.Client(api_key=key)
    interaction = client.interactions.create(
        model=gemini_model(),
        input=prompt,
        response_format={
            "type": "image",
            "mime_type": "image/jpeg",
            "aspect_ratio": os.getenv("GEMINI_IMAGE_ASPECT_RATIO", "4:3"),
        },
    )

    image_data = getattr(getattr(interaction, "output_image", None), "data", None)
    if not image_data:
        raise RuntimeError("Gemini did not return image data.")

    raw = base64.b64decode(image_data)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return str(path)
