"""Turn a raw requirement (pydantic model, dict or namespace) into a validated,
engine-ready request plus a human-readable report of everything that was
interpreted, repaired or refused.

Nothing here raises for problems it can repair; it raises ``RequirementError``
(a ValueError with a clear message) only when the input cannot describe a real
building at all.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from planning.building_programs import canonical_type
from vastu.vastu_engine import entrance_advice
from .plot_geometry import PlotError, resolve_plot, compass_to_bearing, snap_cardinal
from .room_specs import RoomSpecError, parse_custom_rooms, parse_size_overrides

PLOT_KEYS = ("plot_sides", "plot_angles", "plot_traverse", "road_side", "road_bearing_deg")


class RequirementError(ValueError):
    """The requirement cannot be turned into a building; message is user-facing."""


def _as_dict(req: Any) -> Dict[str, Any]:
    if isinstance(req, dict):
        return dict(req)
    for attr in ("model_dump", "dict"):
        fn = getattr(req, attr, None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:  # noqa: BLE001
                pass
    return dict(vars(req))


def _cardinal_from_name(name, default="North"):
    if not name:
        return default
    try:
        return snap_cardinal(compass_to_bearing(name))[0]
    except PlotError:
        return default


def normalize_requirements(req: Any) -> Tuple[SimpleNamespace, dict]:
    """Return (engine-ready request, report)."""
    d = _as_dict(req)
    warnings: List[str] = []
    adjustments: List[str] = []

    # ---- plot: lengths, angles, orientation ------------------------------
    has_plot_extras = any(d.get(k) for k in PLOT_KEYS) or abs(float(d.get("plot_bearing_deg") or 0.0)) > 1e-9
    try:
        plot = resolve_plot(
            plot_length=d.get("plot_length"), plot_width=d.get("plot_width"),
            plot_sides=d.get("plot_sides"), plot_angles=d.get("plot_angles"),
            plot_traverse=d.get("plot_traverse"), plot_bearing_deg=d.get("plot_bearing_deg") or 0.0,
            road_direction=d.get("road_direction"), road_side=d.get("road_side"),
            road_bearing_deg=d.get("road_bearing_deg"), alignment=d.get("plot_alignment") or "auto",
        )
    except PlotError as exc:
        raise RequirementError(str(exc)) from exc
    warnings += plot["warnings"]
    d["plot_length"] = plot["plot_length"]
    d["plot_width"] = plot["plot_width"]
    if float(plot["plot_length"]) != float(_num(req, "plot_length", plot["plot_length"])) or \
            float(plot["plot_width"]) != float(_num(req, "plot_width", plot["plot_width"])):
        adjustments.append(
            f"Plot resolved to a {plot['plot_width']:g} ft (E-W) x {plot['plot_length']:g} ft (N-S) building block "
            f"inside a {plot['area_sqft']:g} sq.ft plot.")
    d["road_direction"] = plot["road_direction"]
    entrance_given = d.get("main_entrance_direction")
    if d.get("road_side") or d.get("road_bearing_deg") not in (None, ""):
        d["main_entrance_direction"] = plot["road_direction"]
    else:
        snapped = _cardinal_from_name(entrance_given, plot["road_direction"])
        if entrance_given and snapped != str(entrance_given).strip().title():
            adjustments.append(f"Entrance '{entrance_given}' was aligned to the nearest main direction: {snapped}.")
        d["main_entrance_direction"] = snapped

    # ---- scalars ---------------------------------------------------------
    d["floors"] = int(min(5, max(1, int(d.get("floors") or 1))))
    d["bedrooms"] = int(max(1, int(d.get("bedrooms") or 1)))
    d["bathrooms"] = int(max(1, int(d.get("bathrooms") or 1)))
    d["kitchens"] = int(max(1, int(d.get("kitchens") or 1)))
    d["budget"] = float(d.get("budget") or 0.0)
    d["wall_width_ft"] = float(min(1.5, max(0.25, float(d.get("wall_width_ft") or 0.5))))
    d["building_type"] = canonical_type(d.get("building_type"))
    if d.get("vastu_preference") is None:
        d["vastu_preference"] = True
    sb = d.get("setback_ft")
    if sb not in (None, ""):
        sb = float(sb)
        if not (0.0 <= sb <= min(plot["plot_length"], plot["plot_width"]) / 3.0):
            raise RequirementError(f"setback_ft must be between 0 and {min(plot['plot_length'], plot['plot_width']) / 3.0:.1f} ft for this plot.")
        d["setback_ft"] = sb
    else:
        d["setback_ft"] = None
    sbs = d.get("setbacks")
    if sbs:
        clean = {}
        for k, v in dict(sbs).items():
            key = str(k).strip().lower()
            if key not in ("north", "east", "south", "west"):
                raise RequirementError(f"Unknown setback side '{k}'. Use north, east, south, west.")
            v = float(v)
            if v < 0 or v > 0.4 * max(plot["plot_length"], plot["plot_width"]):
                raise RequirementError(f"Setback for the {key} side ({v:g} ft) is out of range for this plot.")
            clean[key] = v
        d["setbacks"] = clean
    else:
        d["setbacks"] = None

    plot_area = plot["plot_length"] * plot["plot_width"]
    if d.get("parking") and plot_area < 450:
        d["parking"] = False
        adjustments.append("Parking was left out: the buildable area is under 450 sq.ft.")

    # ---- entrance door position (Vastu auto) -----------------------------
    pos = str(d.get("main_door_position") or "auto").lower()
    advice = entrance_advice(d["road_direction"])
    if pos == "auto":
        pos = advice["recommended_door_position"] if d.get("vastu_preference") else "center"
        adjustments.append(f"Main door position set automatically to '{pos}' ({advice['facing']}-facing Vastu guidance).")
    if pos not in ("left", "center", "right"):
        raise RequirementError("main_door_position must be left, center, right or auto.")
    d["main_door_position"] = pos

    # ---- custom rooms and sizes -----------------------------------------
    try:
        flex = float(d.get("size_flexibility", 0.85) or 0.85)
        if not (0.5 <= flex <= 1.0):
            raise RequirementError("size_flexibility must be between 0.5 and 1.0.")
        specs = parse_custom_rooms(d.get("custom_rooms"), flex=flex)
        overrides = parse_size_overrides(d.get("room_size_overrides"), flex=flex)
    except RoomSpecError as exc:
        raise RequirementError(str(exc)) from exc
    d["_custom_specs"] = specs
    d["_size_overrides"] = overrides
    d["custom_rooms_only"] = bool(d.get("custom_rooms_only"))
    if d["custom_rooms_only"] and not specs:
        raise RequirementError("custom_rooms_only is on, but no custom_rooms were given.")
    d["custom_mode"] = bool(specs or overrides or d["setback_ft"] is not None or d["setbacks"])
    d["north_offset_deg"] = plot["vastu_north_offset_deg"]
    d["auto_adjust"] = d.get("auto_adjust", True) is not False

    out = SimpleNamespace(**d)
    report = {
        "plot": plot,
        "custom_room_count": len(specs),
        "size_overrides": {k: {"dims": list(v["dims"])} for k, v in overrides.items()},
        "custom_mode": d["custom_mode"],
        "entrance": advice,
        "adjustments": adjustments,
        "warnings": warnings,
    }
    return out, report


def _num(req, name, default):
    try:
        v = req.get(name) if isinstance(req, dict) else getattr(req, name, None)
        return default if v in (None, "") else v
    except Exception:  # noqa: BLE001
        return default
