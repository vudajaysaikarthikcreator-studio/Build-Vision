"""User-requirements module: plot geometry (lengths + degrees for every
direction), custom room specifications, and requirement normalisation."""
from .plot_geometry import PlotError, resolve_plot, bearing_to_compass, compass_to_bearing  # noqa: F401
from .room_specs import RoomSpecError, parse_custom_rooms, parse_size_overrides  # noqa: F401
from .normalizer import normalize_requirements, RequirementError  # noqa: F401
