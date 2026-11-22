"""
blueprint/dimension_engine.py

Architectural dimension system (brief section 6): overall building
dimensions along the bottom and left edges, drawn as a standard
dimension line with extension lines and tick/arrow ends, labeled in
feet-inches notation (e.g. 40'-0").
"""


def feet_to_arch_string(value_ft: float) -> str:
    """12.5 -> 12'-6\" (rounded to the nearest inch)."""
    total_inches = round(value_ft * 12)
    feet, inches = divmod(total_inches, 12)
    return f'{feet}\'-{inches}"'


def draw_overall_dimensions(ax, plot_length: float, plot_width: float, color: str,
                             offset: float = 1.6, ext: float = 0.35):
    """Bottom dimension = overall length, left dimension = overall width."""
    # bottom (length)
    y = -offset
    ax.plot([0, 0], [0, y - ext / 2], color=color, linewidth=0.6, zorder=9)
    ax.plot([plot_length, plot_length], [0, y - ext / 2], color=color, linewidth=0.6, zorder=9)
    ax.annotate("", xy=(plot_length, y), xytext=(0, y),
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=0.8), zorder=9)
    ax.text(plot_length / 2, y - 0.45, feet_to_arch_string(plot_length),
            ha="center", va="top", fontsize=8.5, color=color, zorder=9)

    # left (width)
    x = -offset
    ax.plot([0, x - ext / 2], [0, 0], color=color, linewidth=0.6, zorder=9)
    ax.plot([0, x - ext / 2], [plot_width, plot_width], color=color, linewidth=0.6, zorder=9)
    ax.annotate("", xy=(x, plot_width), xytext=(x, 0),
                arrowprops=dict(arrowstyle="<->", color=color, linewidth=0.8), zorder=9)
    ax.text(x - 0.45, plot_width / 2, feet_to_arch_string(plot_width),
            ha="right", va="center", fontsize=8.5, color=color, rotation=90, zorder=9)
