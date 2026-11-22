from dataclasses import dataclass

@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float

def clamp_to_boundary(rect: Rect, plot_length: float, plot_width: float) -> Rect:
    w = max(1.0, min(rect.w, plot_length))
    h = max(1.0, min(rect.h, plot_width))
    x = min(max(0.0, rect.x), max(0.0, plot_length - w))
    y = min(max(0.0, rect.y), max(0.0, plot_width - h))
    return Rect(x, y, w, h)
