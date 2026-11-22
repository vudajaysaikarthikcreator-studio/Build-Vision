"""
backend/schemas.py

Pydantic models for request validation and response shaping. Mirrors the
house-planning form fields from the brief's section 4.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, EmailStr, ConfigDict


class HouseRequirementIn(BaseModel):
    building_type: str = "custom"
    space_count: int = Field(1, ge=1, le=100, description="Primary repeated spaces for the selected building type")
    plot_length: Optional[float] = Field(None, gt=5, description="Plot length (N-S) in feet; optional if plot_sides/plot_traverse is given")
    plot_width: Optional[float] = Field(None, gt=5, description="Plot width (E-W) in feet; optional if plot_sides/plot_traverse is given")
    bedrooms: int = Field(1, ge=0, le=100)
    bathrooms: int = Field(1, ge=1, le=100)
    kitchens: int = Field(1, ge=1, le=20)
    living_room: bool = True
    dining_room: bool = True
    pooja_room: bool = False
    study_room: bool = False
    store_room: bool = False
    parking: bool = True
    floors: int = Field(1, ge=1, le=5)
    staircase: bool = True
    balcony: bool = False
    garden: bool = False
    budget: float = Field(0, ge=0)
    road_direction: str = "North"
    main_entrance_direction: str = "North"
    hall_shape: str = "auto"   # auto | rectangle | l_shape | u_shape (hall-centred residential plans)
    vastu_preference: bool = True
    construction_category: str = "standard"  # basic | standard | premium
    family_size: Optional[int] = None
    project_name: Optional[str] = None
    # Optional civil-engineering inputs (backend/civil/structural_engineering.py).
    # Safe defaults mean existing callers that don't send these keep working.
    soil_type: str = "Medium clay / sandy clay"
    # Presentation / architectural detailing controls.
    wall_width_ft: float = Field(0.5, ge=0.25, le=1.5, description="Indicative wall thickness in feet")
    room_size_details: bool = True
    appearance: str = "technical"  # technical | warm | minimal
    main_door_position: str = "center"  # left | center | right
    parking_position: str = "road_side"  # road_side | left | center | right
    staircase_position: str = "road_side"  # road_side | left | center | right
    design_notes: Optional[str] = None

    # ---- user-requirements module (lengths + degrees for every direction) ----
    # Side lengths in feet: {"north": 38, "east": 62, "south": 42, "west": 58}
    plot_sides: Optional[Dict[str, float]] = None
    # Interior corner angles in degrees: {"SW": 90, "SE": 88, "NE": 92, "NW": 90}
    plot_angles: Optional[Dict[str, float]] = None
    # Surveyor traverse: [{"length": 50, "bearing": 90}, ...] (bearing clockwise from true north)
    plot_traverse: Optional[List[Dict[str, float]]] = None
    # How far the plot's drawn north edge is rotated from true north (degrees, clockwise)
    plot_bearing_deg: float = Field(0.0, ge=-360, le=360)
    plot_alignment: str = "auto"               # auto | compass | plot
    road_side: Optional[str] = None            # side of the plot the road touches, in plot terms
    road_bearing_deg: Optional[float] = Field(None, ge=0, le=360)   # true bearing the road faces
    # Custom spaces and sizes
    custom_rooms: Optional[List[Dict]] = None  # {type|name, count, width_ft, length_ft, zone, floor, ...}
    room_size_overrides: Optional[Dict[str, Dict]] = None   # {"bedroom": {"width_ft": 12, "length_ft": 14}}
    custom_rooms_only: bool = False            # plan ONLY custom_rooms, skip the building-type program
    size_flexibility: float = Field(0.85, ge=0.5, le=1.0)   # custom room minimum = size x this
    setback_ft: Optional[float] = Field(None, ge=0)
    setbacks: Optional[Dict[str, float]] = None              # per side: north/east/south/west


class VastuRow(BaseModel):
    room_type: str
    label: str
    preferred_zone: str
    zone: str
    matched: Optional[bool]


class RoomOut(BaseModel):
    room_type: str
    label: str
    zone: str
    preferred_zone: str
    matched: Optional[bool]
    x: float
    y: float
    w: float
    h: float
    door_side: str
    color: Optional[str] = None
    # Which floor this room sits on (0 = ground, 1 = first, ...). Was being
    # silently dropped by response_model filtering even though the pipeline
    # always sets it (backend/pipeline.py::_floor_layouts) — multi-floor
    # rooms all came back looking like floor 0 to the frontend.
    floor: int = 0


class CostOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    plot_area: float
    builtup_area: float
    floors: int
    num_rooms: int
    rate: float
    material: float
    labour: float
    total: float
    model_predicted_total: Optional[float] = None
    model_predicted_total_challenger: Optional[float] = None


class FloorPlanOut(BaseModel):
    floor: str            # "Ground Floor" | "First Floor" | ...
    image_url: str
    engine: Optional[str] = None      # "gemini" | "local"
    warning: Optional[str] = None     # why Gemini was skipped, if it was


class GeneratePlanOut(BaseModel):
    project_id: str
    name: str
    building_type: str = "custom"
    planning_type: str
    planning_type_challenger: Optional[str] = None
    category: str
    plot: dict
    floors: int
    vastu_enabled: bool
    vastu_match_count: int = 0
    vastu_acceptable_count: int = 0
    vastu_score: float = 0.0
    vastu_match_percent: float = 0.0
    entrance: str
    rooms: List[RoomOut]
    collisions: int
    boundary_violations: int
    cost: CostOut
    blueprint_url: str
    # Per-floor professional blueprint images (brief section 12). Additive
    # field — existing frontends that only read `blueprint_url` (the
    # ground-floor image) keep working unchanged.
    floor_plans: List[FloorPlanOut] = Field(default_factory=list)
    # Which renderer actually produced the images, whether the plan was
    # persisted, and any non-fatal problems the UI should surface.
    blueprint_engine: Optional[str] = None
    feasibility: Optional[dict] = None
    # Preliminary footing/column/beam/slab sizing, material quantities and
    # setback/FAR compliance — see civil/structural_engineering.py.
    structural: Optional[dict] = None
    saved_to_database: bool = True
    warnings: List[str] = Field(default_factory=list)
    algorithm_trace: List[dict] = Field(default_factory=list)
    vastu_audit: Optional[dict] = None
    requirements_report: Optional[dict] = None


class RegisterIn(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginIn(BaseModel):
    email: EmailStr
    password: str
