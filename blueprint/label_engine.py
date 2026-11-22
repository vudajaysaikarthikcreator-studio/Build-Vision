"""
blueprint/label_engine.py

Room labels + dimensions (brief section 5), the floor title (section 8),
and the North compass indicator (section 10). Kept separate from wall/
door/window drawing so text layout can be tuned independently.
"""

from blueprint.dimension_engine import feet_to_arch_string


def _autoscale_font(room_w: float, room_h: float, base: float, lo: float, hi: float) -> float:
    """Scales label font size to the room's smaller dimension so text
    never overruns small rooms (e.g. bathrooms) or looks tiny in big
    ones (e.g. the living room)."""
    smallest = min(room_w, room_h)
    size = base * (smallest / 10.0)
    return max(lo, min(hi, size))


def draw_room_label(ax, room: dict, text_color: str, zorder: int = 7):
    """
    Room name on one line, dimensions on the next — e.g.

        BEDROOM 1
        12'-0" x 14'-0"

    Font size auto-scales with room size; both lines are centered inside
    the room so they never overlap the walls.
    """
    cx = room["x"] + room["w"] / 2
    cy = room["y"] + room["h"] / 2
    smallest = min(room["w"], room["h"])
    name_size = _autoscale_font(room["w"], room["h"], base=11, lo=6, hi=11)
    dim_size = _autoscale_font(room["w"], room["h"], base=8, lo=5, hi=8)

    if smallest < 3.2:
        # too little vertical room for two lines without touching the
        # walls — collapse to a single small centered label
        ax.text(cx, cy, room["label"].upper(), ha="center", va="center",
                 fontsize=min(name_size, 6.5), fontweight="bold", color=text_color, zorder=zorder)
        return

    gap = smallest * 0.12
    ax.text(cx, cy + gap, room["label"].upper(), ha="center", va="center",
             fontsize=name_size, fontweight="bold", color=text_color, zorder=zorder)
    ax.text(cx, cy - gap, f'{feet_to_arch_string(room["w"])} x {feet_to_arch_string(room["h"])}',
             ha="center", va="center", fontsize=dim_size, color=text_color, zorder=zorder)


def draw_floor_title(ax, plot_length: float, plot_width: float, title: str,
                      text_color: str, top_offset: float = 2.6):
    ax.text(0, plot_width + top_offset, title.upper(), ha="left", va="bottom",
             fontsize=15, fontweight="bold", color=text_color)


def draw_compass(ax, cx: float, cy: float, size: float, color: str):
    """A simple architectural N/E/S/W cross-compass indicator."""
    ax.plot([cx - size, cx + size], [cy, cy], color=color, linewidth=1.2, zorder=9)
    ax.plot([cx, cx], [cy - size, cy + size], color=color, linewidth=1.2, zorder=9)
    ax.plot(cx, cy, marker="+", markersize=1, color=color)
    pad = size * 0.42
    ax.text(cx, cy + size + pad, "N", ha="center", va="bottom", fontsize=8, fontweight="bold", color=color, zorder=9)
    ax.text(cx, cy - size - pad, "S", ha="center", va="top", fontsize=8, fontweight="bold", color=color, zorder=9)
    ax.text(cx + size + pad, cy, "E", ha="left", va="center", fontsize=8, fontweight="bold", color=color, zorder=9)
    ax.text(cx - size - pad, cy, "W", ha="right", va="center", fontsize=8, fontweight="bold", color=color, zorder=9)
