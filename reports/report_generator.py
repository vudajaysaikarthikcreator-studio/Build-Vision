"""
reports/report_generator.py

Builds a professional PDF report with ReportLab, per the brief's section
22: project identity, plot dimensions, requirements, Vastu arrangement,
room positions, the blueprint image, cost estimation and floor count.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image as PILImage, ImageDraw, ImageFont
from blueprint.door_engine import build_doors

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# One type scale for the whole report (Helvetica is built into every PDF reader, so the font never falls back).
FONT, FONT_B, FONT_I = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"
SZ_TITLE, SZ_H, SZ_BODY, SZ_META, SZ_TABLE, SZ_SMALL = 22, 14, 10, 9.5, 9, 8.5

INK = colors.HexColor("#16233D")
ORANGE = colors.HexColor("#E1622C")
GRAPHITE = colors.HexColor("#5B6472")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("BVTitle", parent=ss["Title"], fontName=FONT_B, fontSize=SZ_TITLE, leading=SZ_TITLE + 4,
                          textColor=INK, spaceAfter=2, alignment=0))
    ss.add(ParagraphStyle("BVMeta", parent=ss["Normal"], fontName=FONT, fontSize=SZ_META, leading=SZ_META + 3.5,
                          textColor=GRAPHITE))
    ss.add(ParagraphStyle("BVHeading", parent=ss["Heading2"], fontName=FONT_B, fontSize=SZ_H, leading=SZ_H + 4,
                          textColor=INK, spaceBefore=14, spaceAfter=6))
    ss.add(ParagraphStyle("BVBody", parent=ss["Normal"], fontName=FONT, fontSize=SZ_BODY, leading=SZ_BODY + 4,
                          textColor=INK))
    ss.add(ParagraphStyle("BVCell", parent=ss["Normal"], fontName=FONT, fontSize=SZ_TABLE, leading=SZ_TABLE + 3,
                          textColor=INK))
    ss.add(ParagraphStyle("BVDisclaimer", parent=ss["Normal"], fontName=FONT_I, fontSize=SZ_SMALL, leading=SZ_SMALL + 3,
                          textColor=colors.HexColor("#7A5A1E")))
    return ss


def _font_style(header=True):
    """Shared TableStyle commands: same family and size in every table, bold header row."""
    cmds = [("FONTNAME", (0, 0), (-1, -1), FONT), ("FONTSIZE", (0, 0), (-1, -1), SZ_TABLE),
            ("LEADING", (0, 0), (-1, -1), SZ_TABLE + 3), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]
    if header:
        cmds += [("FONTNAME", (0, 0), (-1, 0), FONT_B)]
    return cmds


def _safe(text):
    """Escape for Paragraph and keep to characters the built-in PDF font can draw."""
    t = str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return t.translate({0x2192: "->", 0x2265: ">=", 0x2264: "<=", 0x2713: "OK", 0x26A0: "!", 0x2022: "-",
                        0x2013: "-", 0x2014: "-", 0x00B2: "2"})


def _kv_table(rows, col_widths=(62 * mm, 112 * mm)):
    cell = _styles()["BVCell"]
    rows = [[r[0], Paragraph(_safe(r[1]), cell)] if len(r) == 2 else r for r in rows]
    t = Table(rows, colWidths=list(col_widths))
    t.setStyle(TableStyle(_font_style(header=False) + [
        ("FONTNAME", (0, 0), (0, -1), FONT_B),
        ("TEXTCOLOR", (0, 0), (0, -1), GRAPHITE),
        ("TEXTCOLOR", (1, 0), (1, -1), INK),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#EFE9DA")),
    ]))
    return t


def _hex_to_rgb(value, fallback=(220, 225, 232)):
    try:
        text = str(value or "").strip().lstrip("#")
        if len(text) == 6:
            return tuple(int(text[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        pass
    return fallback


def _shade(rgb, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in rgb)


def _report_font(size, bold=False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _room_color(room_type):
    palette = {
        "master_bedroom": "#DCE9F7",
        "bedroom": "#E5EEF8",
        "living_room": "#F7E9D8",
        "dining_room": "#F2E6CF",
        "kitchen": "#E5F0E8",
        "bathroom": "#DDEDF0",
        "toilet": "#DDEDF0",
        "pooja_room": "#F7F0D9",
        "study_room": "#E9E4F6",
        "store_room": "#E7E7E3",
        "staircase": "#E1E6ED",
        "parking": "#D8DEE8",
        "balcony": "#DCE9F7",
    }
    return palette.get(str(room_type or "").lower(), "#E9EDF2")


def generate_3d_visualization(project: dict, floor_layouts: list | None = None, out_path: str | None = None) -> str | None:
    """Create a clean static isometric massing image for inclusion in PDF reports.

    This is intentionally independent of Matplotlib/GUI state so backend PDF
    generation works on a normal FastAPI server. It mirrors the same room
    coordinates used by the interactive Three.js viewer.
    """
    plot = project.get("plot") or {}
    W = float(plot.get("length") or 40)
    D = float(plot.get("width") or 30)
    floors = max(1, int(project.get("floors") or 1))
    floor_h = 10.5
    wall_h = 9.5

    if floor_layouts:
        rooms = [r for fl in floor_layouts for r in (fl.get("rooms") or [])]
    else:
        rooms = list(project.get("rooms") or [])

    if not rooms:
        return None

    width, height = 1500, 900
    image = PILImage.new("RGB", (width, height), "#F3F7FB")
    draw = ImageDraw.Draw(image)
    title_font = _report_font(30, bold=True)
    sub_font = _report_font(16, bold=False)
    label_font = _report_font(14, bold=True)

    # Isometric scale is computed from the complete plot diagonal so the
    # visualization remains readable for small homes and large plots alike.
    # Zoom to the building itself (not the whole plot) so walls, doors and windows are readable.
    env0 = None
    if floor_layouts:
        env0 = floor_layouts[0].get("building_envelope") or floor_layouts[0].get("occupied_envelope")
    if env0:
        ex0, ey0 = float(env0["x"]) - W / 2 - 1.0, float(env0["y"]) - D / 2 - 1.0
        ex1, ey1 = ex0 + float(env0["w"]) + 2.0, ey0 + float(env0["h"]) + 2.0
        span_w, span_d = ex1 - ex0, ey1 - ey0
    else:
        ex0, ey0, ex1, ey1 = -W / 2, -D / 2, W / 2, D / 2
        span_w, span_d = W, D
    sx = min(16.0, max(2.6, 1000.0 / max(span_w + span_d, 1.0)))
    sz = sx * 0.50
    sy = min(sx * 1.35, max(4.0, 380.0 / max(floors * floor_h + 4.0, 1.0)))

    def raw_proj(x, z, y):
        return ((x - z) * sx, (x + z) * sz - y * sy)

    total_h = floors * floor_h
    pts_b = [(ex0, ey0, 0), (ex1, ey0, 0), (ex1, ey1, 0), (ex0, ey1, 0),
             (ex0, ey0, total_h), (ex1, ey0, total_h), (ex1, ey1, total_h), (ex0, ey1, total_h)]
    bound_points = [raw_proj(*p) for p in pts_b]
    min_x = min(p[0] for p in bound_points)
    max_x = max(p[0] for p in bound_points)
    min_y = min(p[1] for p in bound_points)
    max_y = max(p[1] for p in bound_points)
    target_cx, target_cy = width / 2 + 170, 515
    ox = target_cx - (min_x + max_x) / 2
    oy = target_cy - (min_y + max_y) / 2

    def proj(x, z, y):
        px, py = raw_proj(x, z, y)
        return (px + ox, py + oy)

    def poly(points):
        return [proj(x, z, y) for x, z, y in points]

    # Header.
    draw.text((58, 42), "3D Building Visualization", fill="#0B2545", font=title_font)
    draw.text((60, 82), f"{project.get('name', 'BuildVision Plan')}  |  {W:.0f} x {D:.0f} ft  |  {floors} floor(s)", fill="#5B6472", font=sub_font)

    # Plot base and road-side/front cue.
    base = poly([(-W/2, -D/2, 0), (W/2, -D/2, 0), (W/2, D/2, 0), (-W/2, D/2, 0)])
    draw.polygon(base, fill="#E1E8F0", outline="#0B2545")
    road_side = poly([(-W/2, -D/2, 0), (W/2, -D/2, 0), (W/2, -D/2, 0.15), (-W/2, -D/2, 0.15)])
    draw.line(road_side[:2], fill="#C1502E", width=7)

    # Real walls with door/window openings, painter-sorted back to front, plus numbered tags
    # (D1, W1 ...) with a legend so every door, window and wall is named on the page.
    try:
        from blueprint.model3d import build_floor_model
    except Exception:  # noqa: BLE001
        build_floor_model = None
    legend = []
    tag_font = _report_font(11, bold=True)
    room_font = _report_font(13, bold=True)
    pieces = []          # (depth, polygon points, fill, outline)
    tags = []            # (depth, x, z, y, text, bg)
    DOOR_H, SILL, WIN_TOP = 6.9, 2.6, 6.6
    ent = str(project.get("entrance") or "North")
    ext_c, ext_o = (233, 220, 195), "#9A8E75"
    int_c, int_o = (246, 242, 234), "#B9B2A3"
    for fi in range(floors):
        fl = None
        if floor_layouts and fi < len(floor_layouts):
            fl = floor_layouts[fi]
        base_y = fi * floor_h + 0.25
        # floor slabs
        for r in (fl.get("rooms") if fl else [q for q in rooms if int(q.get("floor", 0) or 0) == fi]) or []:
            x0 = float(r["x"]) - W / 2; z0 = float(r["y"]) - D / 2
            x1 = x0 + float(r["w"]); z1 = z0 + float(r["h"])
            rgb = _hex_to_rgb(r.get("color"), _hex_to_rgb(_room_color(r.get("room_type"))))
            pieces.append((-1e6 + fi, poly([(x0, z0, base_y), (x1, z0, base_y), (x1, z1, base_y), (x0, z1, base_y)]),
                           _shade(rgb, 1.0), "#7C8AA0"))
            tags.append((1e6, (x0 + x1) / 2, (z0 + z1) / 2, base_y + 0.2, str(r.get("label") or r.get("room_type") or "Room"), "room"))
        if fl is None or build_floor_model is None:
            continue
        m = build_floor_model(fl, ent, is_ground_floor=(fi == 0))
        openings = m["doors"] + m["windows"]
        for wl in m["walls"]:
            t = wl["thickness"] / 2.0
            horiz = wl["axis"] == "x"
            mine = sorted([o for o in openings if o["axis"] == wl["axis"] and
                           abs((o["cy"] if horiz else o["cx"]) - wl["c"]) < 0.45 and
                           wl["a"] - 0.01 < (o["cx"] if horiz else o["cy"]) < wl["b"] + 0.01],
                          key=lambda o: o["cx"] if horiz else o["cy"])
            col, outl = (ext_c, ext_o) if wl["exterior"] else (int_c, int_o)

            def block(s0, s1, y0, y1, fill=None, outline=None):
                if s1 - s0 < 0.02 or y1 - y0 < 0.02:
                    return
                if horiz:
                    xa, xb, za, zb = s0 - W / 2, s1 - W / 2, wl["c"] - D / 2 - t, wl["c"] - D / 2 + t
                else:
                    xa, xb, za, zb = wl["c"] - W / 2 - t, wl["c"] - W / 2 + t, s0 - D / 2, s1 - D / 2
                ya, yb = base_y + y0, base_y + y1
                depth = (xa + xb) / 2 + (za + zb) / 2 + fi * 0.01
                f = fill or col
                o = outline or outl
                # visible faces from the isometric camera: top, +x face, +z face
                pieces.append((depth, poly([(xa, za, yb), (xb, za, yb), (xb, zb, yb), (xa, zb, yb)]), _shade(f, 1.06) if isinstance(f, tuple) else f, o))
                pieces.append((depth, poly([(xb, za, ya), (xb, zb, ya), (xb, zb, yb), (xb, za, yb)]), _shade(f, 0.80) if isinstance(f, tuple) else f, o))
                pieces.append((depth, poly([(xa, zb, ya), (xb, zb, ya), (xb, zb, yb), (xa, zb, yb)]), _shade(f, 0.66) if isinstance(f, tuple) else f, o))

            cur = wl["a"]
            for o in mine:
                al = o["cx"] if horiz else o["cy"]
                s0, s1 = max(wl["a"], al - o["width"] / 2), min(wl["b"], al + o["width"] / 2)
                block(cur, s0, 0, wall_h)
                if o["kind"] == "door":
                    block(s0, s1, DOOR_H, wall_h)
                    block(s0, s1, 0, DOOR_H - 0.1, fill=(139, 90, 43), outline="#4B2E12")
                    key = f"D{sum(1 for g in legend if g[0].startswith('D')) + 1}"
                    legend.append((key, o["label"]))
                    tags.append((1e6, (o["cx"] - W / 2), (o["cy"] - D / 2), base_y + DOOR_H + 0.8, key, "door"))
                else:
                    block(s0, s1, 0, SILL)
                    block(s0, s1, WIN_TOP, wall_h)
                    block(s0, s1, SILL, WIN_TOP, fill=(156, 207, 242), outline="#2B6CB0")
                    key = f"W{sum(1 for g in legend if g[0].startswith('W')) + 1}"
                    legend.append((key, o["label"]))
                    tags.append((1e6, (o["cx"] - W / 2), (o["cy"] - D / 2), base_y + WIN_TOP + 0.8, key, "window"))
                cur = max(cur, s1)
            block(cur, wl["b"], 0, wall_h)
    pieces.sort(key=lambda p: p[0])
    for _, pts, fill, outl in pieces:
        draw.polygon(pts, fill=fill, outline=outl)
    for _, x, z, y, text, kind in tags:
        cx, cy = proj(x, z, y)
        bg = {"room": "#0B2545", "door": "#7A3E0E", "window": "#1565C0"}[kind]
        font = room_font if kind == "room" else tag_font
        tw = draw.textbbox((0, 0), text, font=font)[2]
        th = 22 if kind == "room" else 18
        draw.rounded_rectangle((cx - tw / 2 - 7, cy - th / 2, cx + tw / 2 + 7, cy + th / 2), radius=5, fill=bg)
        draw.text((cx - tw / 2, cy - th / 2 + (2 if kind == "room" else 1)), text, fill="#FFFFFF", font=font)
    # legend for the numbered doors/windows
    lx, ly = 60, 130
    draw.text((lx, ly - 24), "Doors (D) and windows (W)", fill="#0B2545", font=_report_font(15, bold=True))
    for i, (key, name) in enumerate(legend[:28]):
        col_i, row_i = divmod(i, 14)
        draw.text((lx + col_i * 250, ly + row_i * 20), f"{key}  {name}", fill="#16233D", font=_report_font(13))

    # North arrow and simple legend.
    draw.line((1440, 120, 1440, 60), fill="#0B2545", width=5)
    draw.polygon([(1440, 45), (1428, 65), (1452, 65)], fill="#C1502E")
    draw.text((1426, 128), "N", fill="#0B2545", font=label_font)
    draw.text((60, 850), "Static view of the BuildVision 3D model (roof removed). Walls, doors (D) and windows (W) are named; rotate the live model in the web app.", fill="#5B6472", font=_report_font(14))

    if out_path is None:
        out_path = os.path.join(OUTPUT_DIR, f"{project.get('id', 'buildvision')}-3d.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    image.save(out_path, "PNG", optimize=True)
    return out_path


def generate_report(project: dict, blueprint_image_path: str, floor_images: list = None, three_d_image_path: str | None = None) -> str:
    """
    project: dict with keys
        id, name, created_at, plot(length,width,area), floors,
        planning_type, category, vastu_enabled, entrance, rooms (list of
        dicts with label/zone/preferred_zone/matched/w/h), cost (dict with
        builtup_area, rate, material, labour, total)

    blueprint_image_path: the ground-floor PNG (kept for backward
        compatibility — always shown under "Blueprint").
    floor_images: optional list of {"title": "First Floor", "path": "..."}
        for multi-floor projects (brief section 8); each gets its own
        page section, in addition to the ground-floor image above.
    """
    ss = _styles()
    out_path = os.path.join(OUTPUT_DIR, f"{project['id']}-report.pdf")
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                             topMargin=18 * mm, bottomMargin=16 * mm,
                             leftMargin=18 * mm, rightMargin=18 * mm)

    story = []
    story.append(Paragraph("BuildVision", ss["BVTitle"]))
    story.append(Paragraph(project.get("name", "House Plan"), ss["BVMeta"]))
    created = project.get("created_at") or datetime.now().isoformat()
    story.append(Paragraph(f"Project ID: {project['id']} &nbsp;&nbsp;·&nbsp;&nbsp; Generated: {created}", ss["BVMeta"]))

    story.append(Paragraph("Plot & Requirements", ss["BVHeading"]))
    plot = project["plot"]
    story.append(_kv_table([
        ["Plot Dimensions", f"{plot['length']:.0f} ft x {plot['width']:.0f} ft ({plot['area']:.0f} sqft)"],
        ["Floors", str(project.get("floors", 1))],
        ["Planning Type (AI recommendation)", project.get("planning_type", "-")],
        ["Construction Category", project.get("category", "-")],
        ["Vastu Arrangement", "Enabled" if project.get("vastu_enabled") else "Disabled"],
        ["Main Entrance", project.get("entrance", "-")],
    ]))

    story.append(Paragraph("Vastu Arrangement", ss["BVHeading"]))
    if project.get("vastu_enabled"):
        audit = project.get("vastu_audit") or {}
        score_text = audit.get("score_percent")
        if score_text is None and project.get("vastu_score") is not None:
            score_text = round(float(project.get("vastu_score")) * 100.0, 1)
        if score_text is not None:
            acceptable = project.get("vastu_acceptable_count")
            total_rooms = len([r for r in project.get("rooms", [])
                               if str(r.get("room_type", "")).lower() not in {"parking", "garden", "loading_bay"}])
            story.append(Paragraph(
                f"Vastu audit score: <b>{float(score_text):.1f}%</b> &nbsp;|&nbsp; acceptable placements: "
                f"{int(acceptable or 0)}/{max(total_rooms, 1)} &nbsp;|&nbsp; method: 9-zone + 16-zone bearing audit",
                ss["BVMeta"]
            ))
            story.append(Spacer(1, 4))
        rows = [["Room", "Preferred Direction", "Assigned Zone", "Result"]]
        for r in project["rooms"]:
            rows.append([r["label"], r.get("preferred_zone", "-"), r["zone"],
                         "Matched" if r.get("matched") else "Reassigned"])
        vt = Table(rows, colWidths=[55 * mm, 40 * mm, 35 * mm, 30 * mm])
        vt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            *_font_style(),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#EFE9DA")),
        ]))
        story.append(vt)
    else:
        story.append(Paragraph("Vastu Arrangement was disabled for this plan.", ss["BVMeta"]))

    # Door schedule is computed from the exact same authoritative geometry used
    # by the blueprint renderer, so the printable report cannot disagree with
    # the drawing about East/West/North/South wall labels or door positions.
    try:
        ground_rooms = [r for r in project.get("rooms", []) if int(r.get("floor", 0) or 0) == 0]
        door_rows = build_doors(
            ground_rooms,
            entrance_direction=project.get("entrance", "North"),
            plot_length=float(plot.get("length", 40)), plot_width=float(plot.get("width", 30)),
            building_envelope=project.get("building_envelope") or None,
            main_door_position=project.get("main_door_position", "center"),
        )
        if door_rows:
            story.append(Paragraph("Door Schedule", ss["BVHeading"]))
            rows = [["Room", "Wall", "Position", "Width", "Swing", "Main"]]
            for d in door_rows:
                rows.append([
                    str(d.get("room", "-")), str(d.get("wall", "-")).title(),
                    f'{d.get("position", 0):.2f} ft', f'{d.get("width", 0):.2f} ft',
                    str(d.get("swing_direction", "inside")).title(),
                    "Yes" if d.get("is_main_entrance") else "No",
                ])
            dt = Table(rows, colWidths=[49 * mm, 22 * mm, 25 * mm, 22 * mm, 24 * mm, 18 * mm], repeatRows=1)
            dt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                *_font_style(),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#EFE9DA")),
            ]))
            story.append(dt)
    except Exception:
        pass

    rows = [["Room", "Zone", "Size (ft)"]]
    for r in project["rooms"]:
        rows.append([r["label"], r["zone"], f'{r["w"]:.1f} x {r["h"]:.1f}'])
    rp = Table(rows, colWidths=[70 * mm, 40 * mm, 50 * mm])
    rp.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        *_font_style(),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#EFE9DA")),
    ]))
    # Keep the heading attached to its table so a page cannot end with a
    # dangling "Room Positions" heading while the actual rows start later.
    story.append(KeepTogether([Paragraph("Room Positions", ss["BVHeading"]), rp]))

    extra_floor_images = [f for f in (floor_images or [])
                          if f.get("path") and os.path.exists(f["path"])
                          and f["path"] != blueprint_image_path]

    if extra_floor_images:
        # Multi-floor project: one labeled section per floor plan.
        story.append(Paragraph("Floor Plans", ss["BVHeading"]))
        if blueprint_image_path and os.path.exists(blueprint_image_path):
            story.append(Paragraph("Ground Floor", ss["BVMeta"]))
            story.append(Image(blueprint_image_path, width=160 * mm, height=160 * mm * 0.72, kind="proportional"))
            story.append(Spacer(1, 8))
        for f in extra_floor_images:
            story.append(Paragraph(f["title"], ss["BVMeta"]))
            story.append(Image(f["path"], width=160 * mm, height=160 * mm * 0.72, kind="proportional"))
            story.append(Spacer(1, 8))
    else:
        story.append(Paragraph("Blueprint", ss["BVHeading"]))
        if blueprint_image_path and os.path.exists(blueprint_image_path):
            story.append(Image(blueprint_image_path, width=160 * mm, height=160 * mm * 0.72, kind="proportional"))

    if three_d_image_path and os.path.exists(three_d_image_path):
        story.append(Paragraph("3D Building Visualization", ss["BVHeading"]))
        story.append(Paragraph("Static isometric view generated from the same room geometry used by the BuildVision 3D viewer.", ss["BVMeta"]))
        # Keep the static isometric view compact enough that the structural
        # summary can normally finish on the same page.
        story.append(Image(three_d_image_path, width=100 * mm, height=100 * mm * 0.643, kind="proportional"))
        story.append(Spacer(1, 8))

    cost = project["cost"]
    cost_block = [
        Paragraph("Cost Estimation", ss["BVHeading"]),
        _kv_table([
            ["Built-up Area", f"{cost['builtup_area']:.0f} sqft"],
            ["Rate / Sqft", f"Rs. {cost['rate']:.0f}"],
            ["Material Cost (~60%)", f"Rs. {cost['material']:,.0f}"],
            ["Labour Cost (~40%)", f"Rs. {cost['labour']:,.0f}"],
            ["Total Estimated Cost", f"Rs. {cost['total']:,.0f}"],
        ]),
        Spacer(1, 6),
        Paragraph(
            "This is an approximate estimation and not a professional construction quotation.",
            ss["BVDisclaimer"]),
    ]
    # Keep the cost heading/table/disclaimer together so ReportLab cannot split
    # the final total onto a mostly blank trailing page.
    story.append(KeepTogether(cost_block))

    structural = project.get("structural")
    if structural:
        story.append(Paragraph("Structural & Compliance Estimate", ss["BVHeading"]))
        footing, column = structural["footing"], structural["column"]
        beam, slab = structural["beam"], structural["slab"]
        setback, far = structural["setback_compliance"], structural["far_compliance"]
        rows = [
            ["Soil Type", footing["soil_type"]],
            ["Safe Bearing Capacity", f"{footing['safe_bearing_capacity_kn_m2']:.0f} kN/m2"],
            ["Recommended Footing", f"{footing['footing_type']}, {footing['recommended_footing_size_ft']} ft"],
            ["Recommended Column", f"{column['recommended_size_in']} in ({column['recommended_size_mm']} mm)"],
            ["Recommended Beam", f"{beam['depth_in']:.0f} in deep x {beam['width_in']:.0f} in wide (span {beam['span_ft']:.0f} ft)"],
            ["Recommended Slab Thickness", f"{slab['thickness_in']:.1f} in ({slab['slab_type']})"],
            ["Setbacks (Front/Rear/Side)", f"{setback['front_setback_ft']:.0f} / {setback['rear_setback_ft']:.0f} / {setback['side_setback_ft']:.0f} ft"],
            ["Max Ground Coverage", f"{setback['max_ground_coverage_pct']:.0f}% ({setback['effective_max_ground_floor_footprint_sqft']:.0f} sqft)"],
            ["Permissible FAR / Built-up", f"{far['permissible_far']:.2f} / {far['permissible_total_builtup_sqft']:.0f} sqft"],
            ["FAR Compliance", "Within limit" if far["within_far_limit"] else "Exceeds typical limit"],
        ]
        if structural.get("staircase"):
            st = structural["staircase"]
            rows.append(["Staircase (Riser/Tread/Width)", f"{st['riser_mm']:.0f} / {st['tread_mm']:.0f} / {st['flight_width_mm']:.0f} mm"])
            rows.append(["Staircase Compliance", "OK" if st["compliant"] else "; ".join(st["issues"])])
        material = structural["material_estimate"]
        rows.append(["Est. Cement", f"{material['cement_bags']:,} bags"])
        rows.append(["Est. Steel", f"{material['steel_tons']:.2f} tons"])
        rows.append(["Est. Sand / Aggregate", f"{material['sand_cft']:,} / {material['aggregate_cft']:,} cft"])
        rows.append(["Est. Bricks", f"{material['bricks']:,}"])
        structural_table = _kv_table(rows)
        structural_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), SZ_TABLE),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        story.append(structural_table)
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "Preliminary concept-planning estimate only; verify the structural design and local bylaws before construction.",
            ss["BVDisclaimer"]
        ))

    doc.build(story)
    return out_path


if __name__ == "__main__":
    demo_project = {
        "id": "demo_project",
        "name": "3BHK - 38x26 Plot",
        "created_at": datetime.now().isoformat(),
        "plot": {"length": 38, "width": 26, "area": 988},
        "floors": 1,
        "planning_type": "Standard",
        "category": "standard",
        "vastu_enabled": True,
        "entrance": "North",
        "rooms": [
            {"label": "Living Room", "zone": "S", "preferred_zone": "N", "matched": False, "w": 11.9, "h": 7.9},
            {"label": "Master Bedroom", "zone": "SW", "preferred_zone": "SW", "matched": True, "w": 11.9, "h": 7.9},
        ],
        "cost": {"builtup_area": 1150, "rate": 1800, "material": 1242000, "labour": 828000, "total": 2070000},
    }
    path = generate_report(demo_project, os.path.join(HERE, "..", "blueprint", "output", "demo_project.png"))
    print("Saved:", path)
