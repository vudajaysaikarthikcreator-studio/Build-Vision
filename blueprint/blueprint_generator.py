"""Gemini image generation with an exact Pillow fallback.

Gemini 3.1 Flash Image is used when BLUEPRINT_RENDERER is auto/gemini and a
valid API key is configured. The deterministic local renderer is used as the
fallback so the pipeline never depends on an image service for correctness.
"""
from __future__ import annotations
import base64, binascii, json, logging, os, re, time
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from PIL import Image
from blueprint.local_renderer import render_local_blueprint
from blueprint.door_engine import build_doors
from planning.building_programs import property_title

log = logging.getLogger(__name__)
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
FLOOR_TITLES = {0:"GROUND FLOOR",1:"FIRST FLOOR",2:"SECOND FLOOR",3:"THIRD FLOOR",4:"FOURTH FLOOR"}
FLOOR_SLUGS = {0:"ground_floor",1:"first_floor",2:"second_floor",3:"third_floor",4:"fourth_floor"}
DEFAULT_MODEL = "gemini-3.1-flash-image"
PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here", "your-api-key", "changeme"}

def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()

def renderer_mode() -> str:
    # Accept both the new names and the older project env names.
    # Exact dimensions are guaranteed by the deterministic renderer. Gemini can
    # be enabled explicitly with BLUEPRINT_RENDERER=gemini when a conceptual
    # AI-rendered image is preferred over dimension-authoritative CAD-style output.
    value = (_env("BLUEPRINT_RENDERER") or _env("GEMINI_RENDERER_MODE", "local")).lower()
    return value if value in {"auto", "gemini", "local"} else "auto"

def gemini_api_key() -> str:
    key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
    return "" if key.lower() in PLACEHOLDER_KEYS else key

def gemini_model() -> str:
    # GEMINI_MODEL is supported for compatibility with existing .env files.
    return _env("GEMINI_IMAGE_MODEL") or _env("GEMINI_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL

def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "project"

def _layout_prompt(layout: Dict, entrance_direction: str, floor_title: str) -> str:
    plot = layout["plot"]
    rooms = layout.get("rooms", [])
    resolved_doors = build_doors(
        rooms, entrance_direction=entrance_direction,
        plot_length=float(plot.get("length", 40)), plot_width=float(plot.get("width", 30)),
        building_envelope=layout.get("building_envelope"),
        main_door_position=layout.get("main_door_position", "center"),
        allow_main_entrance=(int(layout.get("floor", 0) or 0) == 0),
    )
    door_by_room = {d["room"]: d for d in resolved_doors}
    room_data = [{
        "label": r.get("label"),
        "room_type": r.get("room_type"),
        "x_ft": round(float(r.get("x", 0)), 2),
        "y_ft": round(float(r.get("y", 0)), 2),
        "depth_ft": round(float(r.get("w", 0)), 2),
        "frontage_ft": round(float(r.get("h", 0)), 2),
        "dimension_text": f"{float(r.get('h', 0)):.2f} ft × {float(r.get('w', 0)):.2f} ft",
        "zone": r.get("zone"),
        "door": {
            "wall": door_by_room.get(r.get("label"), {}).get("wall"),
            "position_ft": door_by_room.get(r.get("label"), {}).get("position"),
            "width_ft": door_by_room.get(r.get("label"), {}).get("width"),
            "main_entrance": bool(door_by_room.get(r.get("label"), {}).get("is_main_entrance")),
            "door_actual_wall": door_by_room.get(r.get("label"), {}).get("wall"),
        },
    } for r in rooms]
    data = {
        "building_type": layout.get("building_type", "custom"),
        "building_title": property_title(layout.get("building_type", "custom")),
        "plot_length_ft": round(float(plot["length"]), 2),
        "plot_width_ft": round(float(plot["width"]), 2),
        "floor": floor_title,
        "road_direction": layout.get("road_direction", entrance_direction),
        "main_entrance": entrance_direction,
        "setback_ft": layout.get("setback_ft"),
        "building_envelope": layout.get("building_envelope"),
        "wall_width_ft": layout.get("wall_width_ft", 0.5),
        "room_size_details": layout.get("room_size_details", True),
        "appearance": layout.get("appearance", "technical"),
        "main_door_position": layout.get("main_door_position", "center"),
        "parking_position": layout.get("parking_position", "road_side"),
        "staircase_position": layout.get("staircase_position", "road_side"),
        "rooms": room_data,
    }
    return f"""Create an original professional Indian architectural floor-plan drawing for the specified building type.
TOP-DOWN ORTHOGRAPHIC 2D ONLY — NOT A 3D HOUSE, PERSPECTIVE, PHOTOREALISTIC EXTERIOR, OR INFOGRAPHIC.
Use the structured JSON as the geometry source of truth. Preserve the exact plot aspect ratio, every requested room, room names, relative positions, proportions, and supplied dimensions. Treat each dimension_text, depth_ft, and frontage_ft as authoritative. Do not invent, remove, merge, duplicate, resize, or relocate spaces.

DRAWING STYLE: technical CAD-like architectural plan on a white background; use the requested appearance; dark exterior walls with the requested wall width, lighter internal partitions, clear room labels with feet/inches dimensions, door openings with swing arcs using the supplied door geometry, windows, staircase treads with UP direction, parking/open area with entry, setback/open-space indication, road edge and a clearly marked MAIN DOOR only where the supplied exterior-wall geometry permits it, north arrow, overall plot dimensions, and a room-size schedule. Use subtle light fills only to separate rooms; no gradients or decorative art. Keep text horizontal and legible.

PLANNING STYLE: practical functional Indian building planning similar to professionally presented residential floor-plan sheets: clear circulation, connected living/dining/kitchen zones, bedrooms placed as private spaces, bathrooms beside service/private areas, staircase grouped logically, parking near the road-facing side, and open spaces shown clearly. Respect the requested Vastu zone assignments when supplied, but never sacrifice non-overlap or plot boundaries.

IMPORTANT: This is an original BuildVision layout. Do not copy or reproduce any specific website's exact drawing, branding, watermark, furniture arrangement, or proprietary layout.

Use only this floor title: {floor_title}.
LAYOUT JSON:
{json.dumps(data, indent=2)}"""

def _paths(project_id: str, floor_idx: int) -> Tuple[str, Optional[str]]:
    safe = _safe_filename(project_id); slug = FLOOR_SLUGS.get(floor_idx, f"floor{floor_idx}")
    path = os.path.join(OUTPUT_DIR, f"{safe}_{slug}.png")
    legacy = os.path.join(OUTPUT_DIR, f"{safe}.png") if floor_idx == 0 else None
    return path, legacy

def _coerce_image_bytes(data) -> Optional[bytes]:
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
        if raw.startswith((b"\x89PNG", b"RIFF", b"\xff\xd8")):
            return raw
        try:
            decoded = base64.b64decode(raw, validate=True)
            return decoded if decoded.startswith((b"\x89PNG", b"RIFF", b"\xff\xd8")) else raw
        except (binascii.Error, ValueError):
            return raw
    if isinstance(data, str):
        try: return base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError): return None
    return None

def _save_image_bytes(image_bytes: bytes, project_id: str, floor_idx: int) -> str:
    path, legacy = _paths(project_id, floor_idx)
    with Image.open(BytesIO(image_bytes)) as image:
        image.convert("RGB").save(path, "PNG")
    if legacy:
        with Image.open(path) as image: image.save(legacy, "PNG")
    return path

def _generate_with_gemini(prompt: str, aspect_ratio: str | None = None) -> bytes:
    api_key = gemini_api_key()
    if not api_key: raise RuntimeError("GEMINI_API_KEY is not configured in .env")
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("google-genai is not installed. Run: pip install -r requirements.txt") from exc
    client = genai.Client(api_key=api_key)
    attempts = max(1, int(_env("GEMINI_MAX_ATTEMPTS", "2") or 2))
    last = None
    for n in range(attempts):
        try:
            # Gemini Interactions image output currently accepts image/jpeg here.
            interaction = client.interactions.create(
                model=gemini_model(), input=prompt,
                response_format={
                    "type": "image",
                    "mime_type": "image/jpeg",
                    "aspect_ratio": aspect_ratio or _env("GEMINI_IMAGE_ASPECT_RATIO", "4:3"),
                    "image_size": _env("GEMINI_IMAGE_SIZE", "2K"),
                },
            )
            data = getattr(getattr(interaction, "output_image", None), "data", None)
            result = _coerce_image_bytes(data)
            if result: return result
            for step in getattr(interaction, "steps", None) or []:
                for block in getattr(step, "content", None) or []:
                    if getattr(block, "type", None) == "image":
                        result = _coerce_image_bytes(getattr(block, "data", None))
                        if result: return result
            raise RuntimeError("Gemini returned no image data.")
        except Exception as exc:
            last = exc
            if any(t in str(exc).lower() for t in ("api key", "permission", "unauthorized", "invalid argument", "not found", "404", "403", "400")): break
            if n < attempts-1: time.sleep(1.2*(n+1))
    raise RuntimeError(f"Gemini image request failed: {last}")

def generate_gemini_blueprint(layout: Dict, entrance_direction: str, project_id: str, floor_idx: int = 0) -> str:
    title = FLOOR_TITLES.get(floor_idx, f"FLOOR {floor_idx}")
    entrance = entrance_direction if floor_idx == 0 else "Internal staircase access"
    plot = layout.get("plot", {})
    L = float(plot.get("length", 40) or 40)
    W = float(plot.get("width", 30) or 30)
    # Portrait is a better match for depth-first residential plots such as
    # 25x40; wider plots use landscape output. The current Gemini image model
    # supports multiple aspect ratios, so this is selected per plan.
    aspect = "3:4" if L >= W * 1.15 else "4:3"
    prompt = _layout_prompt(layout, entrance, title)
    return _save_image_bytes(_generate_with_gemini(prompt, aspect_ratio=aspect), project_id, floor_idx)

def generate_local_blueprint(layout: Dict, entrance_direction: str, project_id: str, floor_idx: int = 0) -> str:
    path, legacy = _paths(project_id, floor_idx)
    render_local_blueprint(
        layout,
        entrance_direction if floor_idx == 0 else "Internal staircase access",
        path,
        FLOOR_TITLES.get(floor_idx, f"FLOOR {floor_idx}"),
        building_type=layout.get("building_type", "custom"),
        road_direction=layout.get("road_direction", "South"),
    )
    if legacy:
        with Image.open(path) as image: image.save(legacy, "PNG")
    return path

def generate_blueprint(layout, entrance_direction, project_id, floor_title=None):
    return generate_floor_blueprints(layout, entrance_direction, project_id, floors=1)[0]["path"]

def generate_floor_blueprints(layout: Dict, entrance_direction: str, project_id: str, floors: int = 1, floor_layouts: list[Dict] | None = None) -> List[dict]:
    mode, results = renderer_mode(), []
    max_floors=max(1,min(int(floors),5))
    floor_layouts=floor_layouts or [layout for _ in range(max_floors)]
    for floor_idx in range(max_floors):
        floor_layout=floor_layouts[floor_idx] if floor_idx < len(floor_layouts) else layout
        path=None; warning=None; engine='local'
        if mode in ('auto','gemini'):
            try:
                path=generate_gemini_blueprint(floor_layout,entrance_direction,project_id,floor_idx); engine='gemini'
            except Exception as exc:
                warning=str(exc)
                if mode=='gemini': raise
                log.warning('Gemini unavailable; using Pillow fallback: %s',exc)
        if path is None: path=generate_local_blueprint(floor_layout,entrance_direction,project_id,floor_idx)
        results.append({'floor':floor_idx,'title':FLOOR_TITLES.get(floor_idx,f'FLOOR {floor_idx}'),'path':path,'filename':os.path.basename(path),'engine':engine,'warning':warning})
    return results

def gemini_status() -> dict:
    return {"configured": bool(gemini_api_key()), "model": gemini_model(), "renderer_mode": renderer_mode()}
