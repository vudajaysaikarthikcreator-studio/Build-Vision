from __future__ import annotations

import os
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

PLACEHOLDER_KEYS = {"", "your_openai_api_key_here", "your-api-key", "changeme"}


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def openai_api_key() -> str:
    key = _env("OPENAI_API_KEY")
    return "" if key.lower() in PLACEHOLDER_KEYS else key


def openai_model() -> str:
    return _env("OPENAI_MODEL", "gpt-5")


def openai_status() -> Dict[str, Any]:
    return {"configured": bool(openai_api_key()), "model": openai_model()}


def analyze_house_requirements(
    *,
    plot_length: float,
    plot_width: float,
    bedrooms: int,
    bathrooms: int,
    kitchens: int,
    floors: int,
    parking: bool,
    vastu_enabled: bool,
    entrance: str,
    road_direction: str,
    budget: float,
    planning_type: str,
    rooms: list[str],
) -> Dict[str, Any]:
    """Use OpenAI for natural-language design analysis and suggestions.

    This service deliberately does not replace the deterministic ML/Vastu/layout
    engines. It adds an AI explanation/suggestion layer on top of them.
    """
    api_key = openai_api_key()
    if not api_key:
        return {
            "enabled": False,
            "model": openai_model(),
            "summary": "OpenAI is not configured. Add OPENAI_API_KEY to .env.",
            "suggestions": [],
        }

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI SDK is not installed. Run: pip install -r requirements.txt") from exc

    client = OpenAI(api_key=api_key)
    prompt = f"""
You are the AI planning assistant inside BuildVision, a residential floor-plan
planning application. Analyze the supplied structured requirements and the
outputs of deterministic engines. Give concise, practical design suggestions.
Do not invent rooms or dimensions. Do not claim architectural approval or
structural safety. Treat Vastu as a user-selected planning preference, not as a
scientific guarantee.

INPUTS
plot: {plot_length} ft x {plot_width} ft
bedrooms: {bedrooms}
bathrooms: {bathrooms}
kitchens: {kitchens}
floors: {floors}
parking: {parking}
road_direction: {road_direction}
main_entrance: {entrance}
vastu_enabled: {vastu_enabled}
budget_rupees: {budget}
ML planning type: {planning_type}
rooms: {rooms}

Return exactly three headings:
1. ANALYSIS
2. SUGGESTIONS
3. IMAGE_INSTRUCTION

Under IMAGE_INSTRUCTION, write a compact instruction for Gemini to create a
clean black-and-white top-down residential architectural blueprint. Do not add
furniture, people, decorative 3D perspective, or rooms that are not listed.
""".strip()

    response = client.responses.create(model=openai_model(), input=prompt)
    text = (response.output_text or "").strip()
    if not text:
        text = "OpenAI returned an empty response."

    return {
        "enabled": True,
        "model": openai_model(),
        "summary": text,
        "suggestions": _extract_suggestions(text),
    }


def _extract_suggestions(text: str) -> list[str]:
    suggestions: list[str] = []
    in_section = False
    for raw in text.splitlines():
        line = raw.strip()
        upper = line.upper()
        if upper.startswith("2. SUGGESTIONS"):
            in_section = True
            continue
        if in_section and upper.startswith("3. IMAGE_INSTRUCTION"):
            break
        if in_section and line:
            cleaned = line.lstrip("-•0123456789. ").strip()
            if cleaned:
                suggestions.append(cleaned)
    return suggestions[:8]
