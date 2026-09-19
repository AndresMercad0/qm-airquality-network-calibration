"""
Figure 3: dynamic R0 of the NO2 sensor at Node 3

Reads the columns rsc_NO2 and r0_NO2 of the preprocessed Node 3 file of
step s4, from 10 June to 1 July 2025. Draws, on a logarithmic axis, the
hourly mean of the corrected resistance, its values between 00:00 and
05:00 UTC and the daily R0. The PNG goes to figures/output/.
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd

from render import render_with_selenium

# -------------------------------------------------------------------------
# Paths and configuration
# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Default config next to the script; --config <path> overrides it
CONFIG_PATH = (Path(sys.argv[sys.argv.index("--config") + 1]).resolve()
               if "--config" in sys.argv else SCRIPT_DIR / "fig03_r0_dynamic_config.json")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------
def load_series(cfg):
    """Return the window limits, the hourly resistance, the daily R0 and the nocturnal samples."""
    csv_path = Path(cfg["paths"]["input_csv"])
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path

    sensor = cfg["data"]["sensor_column"]
    rsc_col = f"rsc_{sensor}"
    r0_col = f"r0_{sensor}"

    df = pd.read_csv(csv_path, usecols=["timestamp_utc", rsc_col, r0_col],
                     parse_dates=["timestamp_utc"])
    df = df.set_index("timestamp_utc").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    start = pd.Timestamp(cfg["data"]["start"])
    end = pd.Timestamp(cfg["data"]["end"])
    df = df.loc[start:end]

    # Hourly mean of the corrected resistance and one R0 value per day
    rs = df[rsc_col].resample(cfg["data"].get("resample_rule", "1h")).mean().dropna()
    r0 = df[r0_col].resample("D").first().dropna()

    n_start, n_end = cfg["data"]["nocturnal_hours"]
    # Samples inside the nocturnal window (hours in UTC)
    mask_night = (rs.index.hour >= n_start) & (rs.index.hour < n_end)
    rs_night = rs[mask_night]

    return start, end, rs, r0, rs_night


# -------------------------------------------------------------------------
# Scales
# -------------------------------------------------------------------------
class Scales:
    """Map timestamps to x and resistance to a logarithmic y axis."""

    def __init__(self, start, end, y_min, y_max, margin, width, height):
        self.m = margin
        self.w = width
        self.h = height
        self.x0 = pd.Timestamp(start).value
        self.x1 = pd.Timestamp(end).value
        self.y0 = math.log10(y_min)
        self.y1 = math.log10(y_max)
        self.plot_left = margin["left"]
        self.plot_right = width - margin["right"]
        self.plot_top = margin["top"]
        self.plot_bottom = height - margin["bottom"]
        self.plot_w = self.plot_right - self.plot_left
        self.plot_h = self.plot_bottom - self.plot_top

    def sx(self, ts):
        ns = pd.Timestamp(ts).value
        f = (ns - self.x0) / (self.x1 - self.x0)
        return self.plot_left + f * self.plot_w

    def sy(self, v):
        lv = math.log10(max(v, 1e-9))
        f = (lv - self.y0) / (self.y1 - self.y0)
        return self.plot_bottom - f * self.plot_h


def make_polyline(xs, ys, scales):
    pts = [f"{scales.sx(x):.2f},{scales.sy(y):.2f}" for x, y in zip(xs, ys)]
    return " ".join(pts)


def make_path(xs, ys, scales):
    """Build the SVG path data of a polyline."""
    d = []
    first = True
    for x, y in zip(xs, ys):
        cx = scales.sx(x)
        cy = scales.sy(y)
        if first:
            d.append(f"M {cx:.2f} {cy:.2f}")
            first = False
        else:
            d.append(f"L {cx:.2f} {cy:.2f}")
    return " ".join(d)


# -------------------------------------------------------------------------
# SVG
# -------------------------------------------------------------------------
SWATCH_HALF_W = 22    # Half length of the legend line swatches (px)


def build_svg(cfg, start, end, rs, r0, rs_night):
    vp = cfg["viewport"]
    width = vp["width_px"]
    height = vp["height_px"]
    pal = cfg["palette"]
    ov = cfg["overlay"]
    typ = cfg["typography"]
    plt_cfg = cfg["plot"]

    # Axis limits from the tick values, with a margin below and above
    y_ticks = plt_cfg["y_tick_values"]
    y_min = min(y_ticks) * 0.9
    y_max = max(y_ticks) * 2.2

    scales = Scales(start, end, y_min, y_max, plt_cfg["margin"], width, height)

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Dynamic R0 example">'
    )

    parts.append(f'<rect width="{width}" height="{height}" fill="{pal["background"]}" />')

    for yv in y_ticks:
        gy = scales.sy(yv)
        parts.append(
            f'<line x1="{scales.plot_left:.2f}" y1="{gy:.2f}" '
            f'x2="{scales.plot_right:.2f}" y2="{gy:.2f}" '
            f'stroke="{pal["grid_line"]}" stroke-width="{ov["grid_linewidth"]}" '
            f'stroke-dasharray="1 4" />'
        )

    # Hourly resistance line
    rs_path = make_path(rs.index, rs.values, scales)
    parts.append(
        f'<path d="{rs_path}" fill="none" stroke="{pal["rs_line"]}" '
        f'stroke-width="{ov["rs_linewidth"]}" stroke-opacity="{ov["rs_opacity"]}" '
        f'stroke-linejoin="round" stroke-linecap="round" />'
    )

    # Nocturnal samples
    r = ov["night_marker_radius"]
    dots = []
    for t, v in rs_night.items():
        cx = scales.sx(t)
        cy = scales.sy(v)
        dots.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" '
            f'fill="{pal["night_dot"]}" fill-opacity="{ov["night_opacity"]}" '
            f'stroke="{pal["night_dot_edge"]}" stroke-width="0.6" />'
        )
    parts.append("".join(dots))

    # Daily R0, drawn at midday of each day
    r0_t = [t + pd.Timedelta(hours=12) for t in r0.index]
    r0_v = list(r0.values)
    r0_path = make_path(r0_t, r0_v, scales)
    parts.append(
        f'<path d="{r0_path}" fill="none" stroke="{pal["r0_line"]}" '
        f'stroke-width="{ov["r0_linewidth"]}" stroke-linejoin="round" '
        f'stroke-linecap="round" />'
    )
    rR0 = ov["r0_marker_radius"]
    for t, v in zip(r0_t, r0_v):
        cx = scales.sx(t)
        cy = scales.sy(v)
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{rR0}" '
            f'fill="{pal["r0_line"]}" stroke="#FFFFFF" stroke-width="1" />'
        )

    parts.append(
        f'<line x1="{scales.plot_left:.2f}" y1="{scales.plot_top:.2f}" '
        f'x2="{scales.plot_left:.2f}" y2="{scales.plot_bottom:.2f}" '
        f'stroke="{pal["axis_line"]}" stroke-width="{ov["axis_linewidth"]}" />'
    )
    parts.append(
        f'<line x1="{scales.plot_left:.2f}" y1="{scales.plot_bottom:.2f}" '
        f'x2="{scales.plot_right:.2f}" y2="{scales.plot_bottom:.2f}" '
        f'stroke="{pal["axis_line"]}" stroke-width="{ov["axis_linewidth"]}" />'
    )

    for yv in y_ticks:
        gy = scales.sy(yv)
        parts.append(
            f'<line x1="{scales.plot_left - 4:.2f}" y1="{gy:.2f}" '
            f'x2="{scales.plot_left:.2f}" y2="{gy:.2f}" '
            f'stroke="{pal["axis_line"]}" stroke-width="{ov["axis_linewidth"]}" />'
        )
        parts.append(
            f'<text x="{scales.plot_left - 12:.2f}" y="{gy + typ["tick_px"] * 0.35:.2f}" '
            f'text-anchor="end" fill="{pal["text_secondary"]}" '
            f'font-size="{typ["tick_px"]}">{yv:,}</text>'
        )

    # X ticks (days of June 2025)
    for d in plt_cfg["x_tick_days"]:
        ts = pd.Timestamp(f"2025-06-{d:02d}")
        gx = scales.sx(ts)
        parts.append(
            f'<line x1="{gx:.2f}" y1="{scales.plot_bottom:.2f}" '
            f'x2="{gx:.2f}" y2="{scales.plot_bottom + 4:.2f}" '
            f'stroke="{pal["axis_line"]}" stroke-width="{ov["axis_linewidth"]}" />'
        )
        parts.append(
            f'<text x="{gx:.2f}" y="{scales.plot_bottom + typ["tick_px"] + 8:.2f}" '
            f'text-anchor="middle" fill="{pal["text_secondary"]}" '
            f'font-size="{typ["tick_px"]}">{d} Jun</text>'
        )

    # Axis labels
    x_label_y = (scales.plot_bottom + typ["tick_px"] + 8
                 + typ["axis_label_px"] * 1.15)
    parts.append(
        f'<text x="{(scales.plot_left + scales.plot_right) / 2:.2f}" '
        f'y="{x_label_y:.2f}" text-anchor="middle" '
        f'fill="{pal["text_primary"]}" font-size="{typ["axis_label_px"]}" '
        f'font-style="italic">Date  (June 2025)</text>'
    )

    y_label_x = scales.plot_left - 78
    y_label_y = (scales.plot_top + scales.plot_bottom) / 2
    parts.append(
        f'<text x="{y_label_x:.2f}" y="{y_label_y:.2f}" '
        f'text-anchor="middle" fill="{pal["text_primary"]}" '
        f'font-size="{typ["axis_label_px"]}" font-style="italic" '
        f'transform="rotate(-90 {y_label_x:.2f} {y_label_y:.2f})">'
        f'Resistance  (Ω)</text>'
    )

    # Legend box
    lg = cfg["legend"]
    n_start, n_end = cfg["data"]["nocturnal_hours"]
    fs = typ["inline_caption_px"]
    note_fs = lg["note_px"]
    row_h = lg["row_h_px"]
    if lg.get("align") == "right":
        box_x = scales.plot_right - lg["width_px"] - lg["x_off_px"]
    else:
        box_x = scales.plot_left + lg["x_off_px"]
    box_y = lg["y_off_px"]
    box_w = lg["width_px"]
    box_h = 3 * row_h + note_fs * 1.45 + 16
    parts.append(
        f'<rect x="{box_x:.2f}" y="{box_y:.2f}" '
        f'width="{box_w}" height="{box_h:.2f}" rx="10" ry="10" '
        f'fill="#FFFFFF" fill-opacity="1.0" '
        f'stroke="{pal["legend_border"]}" stroke-width="1.2" />'
    )
    sw_cx = box_x + 16 + SWATCH_HALF_W
    tx = box_x + 16 + 2 * SWATCH_HALF_W + 18

    # Row 1: hourly resistance
    y1 = box_y + row_h * 0.66
    parts.append(
        f'<line x1="{sw_cx - SWATCH_HALF_W:.2f}" y1="{y1:.2f}" '
        f'x2="{sw_cx + SWATCH_HALF_W:.2f}" y2="{y1:.2f}" '
        f'stroke="{pal["rs_line"]}" stroke-width="{ov["rs_linewidth"]}" '
        f'stroke-opacity="{ov["rs_opacity"]}" stroke-linecap="round" />'
    )
    parts.append(
        f'<text x="{tx:.2f}" y="{y1 + fs * 0.34:.2f}" text-anchor="start" '
        f'fill="{pal["text_secondary"]}" font-size="{fs}" '
        f'font-style="italic">Hourly  RS<tspan baseline-shift="sub" '
        f'font-size="{fs * 0.8:.1f}">corr</tspan></text>'
    )

    # Row 2: nocturnal samples
    y2 = box_y + row_h * 1.66
    for dx_dot in (-14, 0, 14):
        parts.append(
            f'<circle cx="{sw_cx + dx_dot:.2f}" cy="{y2:.2f}" '
            f'r="{ov["night_marker_radius"]}" '
            f'fill="{pal["night_dot"]}" fill-opacity="{ov["night_opacity"]}" '
            f'stroke="{pal["night_dot_edge"]}" stroke-width="0.6" />'
        )
    parts.append(
        f'<text x="{tx:.2f}" y="{y2 + fs * 0.34:.2f}" text-anchor="start" '
        f'fill="{pal["night_text"]}" font-size="{fs}" font-style="italic">'
        f'Nocturnal samples ({n_start:02d}:00 – {n_end:02d}:00 UTC)</text>'
    )

    # Row 3: dynamic R0 and the method note
    y3 = box_y + row_h * 2.66
    parts.append(
        f'<line x1="{sw_cx - SWATCH_HALF_W:.2f}" y1="{y3:.2f}" '
        f'x2="{sw_cx + SWATCH_HALF_W:.2f}" y2="{y3:.2f}" '
        f'stroke="{pal["r0_line"]}" stroke-width="{ov["r0_linewidth"]}" '
        f'stroke-linecap="round" />'
    )
    parts.append(
        f'<circle cx="{sw_cx:.2f}" cy="{y3:.2f}" r="{ov["r0_marker_radius"]}" '
        f'fill="{pal["r0_line"]}" stroke="#FFFFFF" stroke-width="1" />'
    )
    parts.append(
        f'<text x="{tx:.2f}" y="{y3 + fs * 0.34:.2f}" text-anchor="start" '
        f'fill="{pal["r0_text"]}" font-size="{typ["inline_label_px"]}" '
        f'font-weight="700" font-style="italic">'
        f'Dynamic R<tspan baseline-shift="sub" '
        f'font-size="{typ["inline_label_px"] * 0.75:.1f}">0</tspan>(d)</text>'
    )
    y4 = y3 + note_fs * 1.5
    parts.append(
        f'<text x="{tx:.2f}" y="{y4 + note_fs * 0.34:.2f}" text-anchor="start" '
        f'fill="{pal["text_secondary"]}" font-size="{note_fs}" '
        f'font-style="italic">14-day nocturnal P10,  EWMA  α = 0.2</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


# -------------------------------------------------------------------------
# HTML wrapper
# -------------------------------------------------------------------------
def build_html(cfg, svg_markup):
    typ = cfg["typography"]
    pal = cfg["palette"]
    vp = cfg["viewport"]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dynamic R0</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Crimson+Pro:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: {pal["background"]};
  }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); }}
  body {{
    font-family: {typ["font_stack"]};
    color: {pal["text_primary"]};
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
  }}
  .figure-wrap {{
    width: {vp["width_px"]}px;
    background: var(--bg);
    padding: 0;
  }}
  svg text {{
    font-family: {typ["font_stack"]};
    letter-spacing: 0.01em;
  }}
</style>
</head>
<body>
  <div class="figure-wrap" id="figure-root">
    {svg_markup}
  </div>
</body>
</html>
"""


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def main():
    cfg = load_config(CONFIG_PATH)
    start, end, rs, r0, rs_night = load_series(cfg)

    svg = build_svg(cfg, start, end, rs, r0, rs_night)
    html = build_html(cfg, svg)

    html_out = REPO_ROOT / cfg["paths"]["output_html"]
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(html, encoding="utf-8")
    print(f"HTML: {html_out}")

    png_out = REPO_ROOT / cfg["paths"]["output_image"]
    png_out.parent.mkdir(parents=True, exist_ok=True)
    render_with_selenium(html_out, png_out, cfg, wait_s=2.2)
    print(f"PNG : {png_out}")


if __name__ == "__main__":
    main()
