"""
Figure 6: test R2 per calibration method at Nodes 3 and 5

Reads, for each node, metricas_por_nivel_de_costo_v6.csv (manufacturer
curves and OLS) and tabla_maestra_locked_v6.csv (ML model). Draws grouped
bars for PM2.5, CO and O3; bars below the axis minimum are cut and
labelled with their value. The PNG goes to figures/output/.
"""

import json
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
               if "--config" in sys.argv else SCRIPT_DIR / "fig06_r2_comparison_config.json")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------------
MODEL_DISPLAY = {
    "RF": "RF", "DNN": "DNN", "XGBoost": "XGB",
    "LightGBM": "LGBM", "LSTM": "LSTM",
}


def load_node_results(costs_csv: Path, ml_csv: Path) -> dict:
    """Return {gas: {manufacturer, ols, ml, ml_model, ml_config}} for one node."""
    costs = pd.read_csv(costs_csv)
    ml = pd.read_csv(ml_csv)

    out = {}
    for gas in ("PM2.5", "CO", "NO2", "O3"):
        gas_rows = costs[costs["gas"] == gas]
        # The manufacturer rows carry either of two labels in the result files
        mfg = gas_rows[gas_rows["nivel_costo"].isin(
            ["Low-cost (cal. fabricante)", "Low-cost curvas fab."]
        )]["r2"]
        ols = gas_rows[gas_rows["nivel_costo"] == "Low-cost OLS"]["r2"]

        # Row with the highest test R2 for this gas
        ml_gas = ml[ml["gas"] == gas].sort_values("R2_test", ascending=False)
        best = ml_gas.iloc[0]

        out[gas] = {
            "manufacturer": float(mfg.iloc[0]) if len(mfg) else float("nan"),
            "ols":          float(ols.iloc[0]) if len(ols) else float("nan"),
            "ml":           float(best["R2_test"]),
            "ml_model":     str(best["model"]),
            "ml_config":    str(best["config"]),
        }
    return out


# -------------------------------------------------------------------------
# Geometry
# -------------------------------------------------------------------------
class PanelScales:
    """Hold the pixel frame of one panel and map R2 values to y."""

    def __init__(self, x_left, x_right, y_top, y_bottom, y_min, y_max):
        self.x_left = x_left
        self.x_right = x_right
        self.y_top = y_top
        self.y_bottom = y_bottom
        self.y_min = y_min
        self.y_max = y_max
        self.width = x_right - x_left
        self.height = y_bottom - y_top

    def sy(self, value):
        f = (value - self.y_min) / (self.y_max - self.y_min)
        return self.y_bottom - f * self.height

    def y_zero(self):
        return self.sy(0.0)


def _bbox_overlap(a, b):
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


# -------------------------------------------------------------------------
# Label placement
# -------------------------------------------------------------------------
# Approximate glyph advance of the bold serif, as a fraction of the font size
CHAR_WIDTH_FACTOR = 0.52


def _label_bbox(x_center, y_baseline, text, font_size):
    """Return the approximate box (x0, y0, x1, y1) of a centred label, padding included."""
    n = max(len(text), 3)
    text_w = n * font_size * CHAR_WIDTH_FACTOR
    pad_x = 4
    pad_y = 2
    return (
        x_center - text_w / 2 - pad_x,
        y_baseline - font_size - pad_y,
        x_center + text_w / 2 + pad_x,
        y_baseline + pad_y,
    )


def _place_value_label(parts, x_center, y_default, bar_anchor_y,
                       text, font_size, color, weight,
                       occupied, leader_color,
                       lift_direction=-1, max_steps=10):
    """
    Shift a value label until it overlaps no other label; draw a leader if it moved.
    lift_direction is -1 to lift (positive bars) and +1 to push down (negative bars).
    """
    step = font_size * 1.05
    y = y_default
    bbox = _label_bbox(x_center, y, text, font_size)
    displaced = False

    for _ in range(max_steps):
        if not any(_bbox_overlap(bbox, o) for o in occupied):
            break
        y += lift_direction * step
        bbox = _label_bbox(x_center, y, text, font_size)
        displaced = True

    occupied.append(bbox)

    if displaced:
        if lift_direction == -1:
            leader_end = bbox[3] + 1
        else:
            leader_end = bbox[1] - 1
        parts.append(
            f'<line x1="{x_center:.2f}" y1="{bar_anchor_y:.2f}" '
            f'x2="{x_center:.2f}" y2="{leader_end:.2f}" '
            f'stroke="{leader_color}" stroke-width="1.0" '
            f'opacity="0.55" />'
        )

    parts.append(
        f'<text x="{x_center:.2f}" y="{y:.2f}" text-anchor="middle" '
        f'fill="{color}" font-size="{font_size}" '
        f'font-weight="{weight}">{text}</text>'
    )


# -------------------------------------------------------------------------
# Panel rendering
# -------------------------------------------------------------------------
PANEL_TITLES = {
    "node3": ("Node 3", "QMUL · Mile End Road · ref. TH2"),
    "node5": ("Node 5", "KEMP · Glamis Road · ref. TH7"),
}

POLLUTANT_LABELS = {
    "PM2.5": ('PM<tspan baseline-shift="sub" font-size="0.75em">2.5</tspan>', "PM2.5"),
    "CO":    ("CO", "CO"),
    "NO2":   ('NO<tspan baseline-shift="sub" font-size="0.75em">2</tspan>', "NO2"),
    "O3":    ('O<tspan baseline-shift="sub" font-size="0.75em">3</tspan>', "O3"),
}

SENSOR_NAMES = {
    "PM2.5": "SPS30", "CO": "GM-702B",
    "NO2":   "GM-102B", "O3": "MQ131",
}

METHOD_ORDER = ("manufacturer", "ols", "ml")
METHOD_LABEL = {
    "manufacturer": "Manufacturer curves",
    "ols":          "OLS  +  T/RH",
    "ml":           "Best ML",
}


def _format_truncated_value(r2):
    """Format the true value of a bar cut at the axis minimum."""
    av = abs(r2)
    if av >= 1000:
        return f"{r2:,.0f}"
    if av < 10:
        return f"{r2:.2f}"
    return f"{r2:.0f}"


def _add_panel(parts, panel_key, scales, panel_data, cfg, draw_y_axis=True):
    pal = cfg["palette"]
    typ = cfg["typography"]
    ov = cfg["overlay"]
    plt_cfg = cfg["plot"]

    bar_w = plt_cfg["bar_width_px"]
    intra = plt_cfg["intra_gap_px"]
    group_w = 3 * bar_w + 2 * intra
    pollutants = plt_cfg["pollutant_order"]
    n_groups = len(pollutants)
    inner_pad = plt_cfg.get("panel_inner_pad_px", 0)
    inner_left = scales.x_left + inner_pad
    inner_right = scales.x_right - inner_pad
    inner_w = inner_right - inner_left
    col_width = inner_w / n_groups
    group_centers = [
        inner_left + col_width * (i + 0.5) for i in range(n_groups)
    ]
    corner_r = plt_cfg.get("bar_corner_radius_px", 0)

    # Panel title and subtitle above the plot area
    layout = cfg.get("layout", {})
    title, sub = PANEL_TITLES[panel_key]
    title_x = (scales.x_left + scales.x_right) / 2
    title_y = layout.get(
        "title_row_y_px",
        scales.y_top - typ["panel_subtitle_px"] * 1.5 - 8,
    )
    parts.append(
        f'<text x="{title_x:.2f}" y="{title_y:.2f}" '
        f'text-anchor="middle" fill="{pal["panel_subtitle"]}" '
        f'font-size="{typ["panel_title_px"]}" font-weight="700" '
        f'font-style="italic">{title}</text>'
    )
    parts.append(
        f'<text x="{title_x:.2f}" y="{title_y + typ["panel_subtitle_px"] * 1.4:.2f}" '
        f'text-anchor="middle" fill="{pal["text_secondary"]}" '
        f'font-size="{typ["panel_subtitle_px"]}" font-style="italic">{sub}</text>'
    )

    # Horizontal grid at the major ticks
    for yv in plt_cfg["y_tick_values"]:
        gy = scales.sy(yv)
        parts.append(
            f'<line x1="{scales.x_left:.2f}" y1="{gy:.2f}" '
            f'x2="{scales.x_right:.2f}" y2="{gy:.2f}" '
            f'stroke="{pal["grid_line"]}" stroke-width="{ov["grid_linewidth"]}" '
            f'stroke-dasharray="1 4" />'
        )

    y_min = plt_cfg["y_min"]

    # First pass: bars, keeping their geometry for the labels
    bar_geoms = []
    for gi, gas in enumerate(pollutants):
        cx = group_centers[gi]
        x_first_bar_left = cx - group_w / 2
        for mi, method in enumerate(METHOD_ORDER):
            x = x_first_bar_left + mi * (bar_w + intra)
            r2 = panel_data[gas][method]
            color = pal[f"{method}_bar"]

            # Bars below the axis minimum are cut there and marked with an arrow
            truncated = r2 < y_min
            display = y_min if truncated else r2

            y0 = scales.y_zero()
            yv = scales.sy(display)
            top = min(y0, yv)
            bottom = max(y0, yv)
            height = bottom - top
            r = min(corner_r, height / 2.0) if corner_r else 0
            parts.append(
                f'<rect x="{x:.2f}" y="{top:.2f}" width="{bar_w:.2f}" '
                f'height="{height:.2f}" rx="{r:.2f}" ry="{r:.2f}" '
                f'fill="{color}" />'
            )

            if truncated:
                tcx = x + bar_w / 2
                tcy = scales.sy(y_min) + 7
                s = ov["trunc_arrow_size"]
                parts.append(
                    f'<polygon points="'
                    f'{tcx - s:.2f},{tcy:.2f} '
                    f'{tcx + s:.2f},{tcy:.2f} '
                    f'{tcx:.2f},{tcy + s * 1.15:.2f}" '
                    f'fill="{pal["manufacturer_text"]}" opacity="0.8" />'
                )

            bar_geoms.append({
                "x_center": x + bar_w / 2,
                "top": top,
                "bottom": bottom,
                "r2": r2,
                "method": method,
                "gi": gi,
                "truncated": truncated,
            })

    # Second pass: value labels, group by group
    method_label_specs = {
        "ml":           ("ml_text",        "value_label_ml_px",  "700"),
        "ols":          ("ols_text",       "value_label_ols_px", "600"),
        "manufacturer": ("manufacturer_text", "value_label_ols_px", "600"),
    }
    method_priority = {"ml": 0, "ols": 1, "manufacturer": 2}

    by_group = {}
    for g in bar_geoms:
        by_group.setdefault(g["gi"], []).append(g)

    for gi, geoms in by_group.items():
        geoms.sort(key=lambda g: method_priority[g["method"]])
        occupied = []
        for g in geoms:
            method = g["method"]
            r2 = g["r2"]
            color_key, font_key, weight = method_label_specs[method]
            font_size = typ[font_key]
            color = pal[color_key]

            # A truncated bar shows its true value inside the bar, just below zero
            if g["truncated"]:
                txt = _format_truncated_value(r2)
                inner_color = "#FFFFFF" if method == "ml" else pal["text_secondary"]
                label_y = scales.y_zero() + font_size + 4
                parts.append(
                    f'<text x="{g["x_center"]:.2f}" y="{label_y:.2f}" '
                    f'text-anchor="middle" fill="{inner_color}" '
                    f'font-size="{font_size}" font-weight="600">{txt}</text>'
                )
            elif r2 >= 0:
                txt = f"{(0.0 if abs(r2) < 0.005 else r2):.2f}"
                label_y = g["top"] - 6
                parts.append(
                    f'<text x="{g["x_center"]:.2f}" y="{label_y:.2f}" '
                    f'text-anchor="middle" fill="{color}" '
                    f'font-size="{font_size}" font-weight="{weight}">{txt}</text>'
                )
            else:
                txt = f"{(0.0 if abs(r2) < 0.005 else r2):.2f}"
                label_y = g["bottom"] + font_size + 2
                parts.append(
                    f'<text x="{g["x_center"]:.2f}" y="{label_y:.2f}" '
                    f'text-anchor="middle" fill="{color}" '
                    f'font-size="{font_size}" font-weight="{weight}">{txt}</text>'
                )

    # ML model name: right_top beside the bar, above_center_short above it with a dashed leader
    placements = cfg.get("ml_callouts", {})
    panel_overrides = cfg.get("ml_callouts_per_panel", {}).get(panel_key, {})
    if placements:
        callout_font = typ.get("ml_callout_px", 18)
        callout_color = pal["ml_text"]
        leader_color = ov.get("ml_callout_leader_color",
                              pal.get("callout_line", "#B8ACD1"))
        leader_opacity = ov.get("ml_callout_leader_opacity", 0.65)
        leader_dash = ov.get("ml_callout_leader_dash", "2 3")
        for gi, gas in enumerate(pollutants):
            ml_geom = next(
                (g for g in by_group.get(gi, []) if g["method"] == "ml"),
                None,
            )
            if ml_geom is None:
                continue
            spec = placements.get(gas)
            if not isinstance(spec, dict):
                continue
            override = panel_overrides.get(gas)
            if isinstance(override, dict):
                spec = {**spec, **override}
            model = panel_data[gas]["ml_model"]
            cfg_label = panel_data[gas]["ml_config"]
            text = f"{MODEL_DISPLAY.get(model, model)}-{cfg_label}"
            placement = spec.get("placement", "above_center_short")

            if placement == "right_top":
                gap_x = spec.get("gap_x_px", 6)
                x = ml_geom["x_center"] + bar_w / 2 + gap_x
                y = ml_geom["top"] + callout_font * 0.85
                parts.append(
                    f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="start" '
                    f'fill="{callout_color}" font-size="{callout_font}" '
                    f'font-style="italic" font-weight="600">{text}</text>'
                )
                continue

            offset = spec.get("offset_above_px", 28)
            cx = ml_geom["x_center"]
            y = ml_geom["top"] - offset
            leader_y_top = y + 4
            leader_y_bot = ml_geom["top"] - 2
            if leader_y_bot - leader_y_top > 4:
                parts.append(
                    f'<line x1="{cx:.2f}" y1="{leader_y_top:.2f}" '
                    f'x2="{cx:.2f}" y2="{leader_y_bot:.2f}" '
                    f'stroke="{leader_color}" stroke-width="1.0" '
                    f'opacity="{leader_opacity}" '
                    f'stroke-dasharray="{leader_dash}" />'
                )
            parts.append(
                f'<text x="{cx:.2f}" y="{y:.2f}" text-anchor="middle" '
                f'fill="{callout_color}" font-size="{callout_font}" '
                f'font-style="italic" font-weight="600">{text}</text>'
            )

    for gi, gas in enumerate(pollutants):
        cx = group_centers[gi]
        label_html, _ = POLLUTANT_LABELS[gas]
        # Pollutant, sensor and optional reference labels under the axis
        pollutant_y = scales.y_bottom + typ["pollutant_label_px"] + 18
        parts.append(
            f'<text x="{cx:.2f}" y="{pollutant_y:.2f}" '
            f'text-anchor="middle" fill="{pal["text_primary"]}" '
            f'font-size="{typ["pollutant_label_px"]}">{label_html}</text>'
        )
        sensor_y = pollutant_y + typ["sensor_label_px"] * 1.7
        parts.append(
            f'<text x="{cx:.2f}" y="{sensor_y:.2f}" '
            f'text-anchor="middle" fill="{pal["text_secondary"]}" '
            f'font-size="{typ["sensor_label_px"]}" font-style="italic">'
            f'{SENSOR_NAMES[gas]}</text>'
        )
        ref_label = cfg.get("ref_labels", {}).get(gas)
        if ref_label:
            parts.append(
                f'<text x="{cx:.2f}" y="{sensor_y + typ["sensor_label_px"] * 1.25:.2f}" '
                f'text-anchor="middle" fill="{pal["text_muted"]}" '
                f'font-size="{typ["sensor_label_px"] * 0.85:.1f}" font-style="italic">'
                f'{ref_label}</text>'
            )

    # Zero line
    y_zero = scales.y_zero()
    parts.append(
        f'<line x1="{scales.x_left:.2f}" y1="{y_zero:.2f}" '
        f'x2="{scales.x_right:.2f}" y2="{y_zero:.2f}" '
        f'stroke="{pal["zero_line"]}" stroke-width="{ov["zero_linewidth"]}" />'
    )

    # Y axis, drawn on the left panel only
    if draw_y_axis:
        parts.append(
            f'<line x1="{scales.x_left:.2f}" y1="{scales.y_top:.2f}" '
            f'x2="{scales.x_left:.2f}" y2="{scales.y_bottom:.2f}" '
            f'stroke="{pal["axis_line"]}" stroke-width="{ov["axis_linewidth"]}" />'
        )
        for yv in plt_cfg["y_tick_values"]:
            gy = scales.sy(yv)
            parts.append(
                f'<line x1="{scales.x_left - 5:.2f}" y1="{gy:.2f}" '
                f'x2="{scales.x_left:.2f}" y2="{gy:.2f}" '
                f'stroke="{pal["axis_line"]}" stroke-width="{ov["axis_linewidth"]}" />'
            )
            parts.append(
                f'<text x="{scales.x_left - 12:.2f}" y="{gy + typ["tick_px"] * 0.35:.2f}" '
                f'text-anchor="end" fill="{pal["text_secondary"]}" '
                f'font-size="{typ["tick_px"]}">{yv:g}</text>'
            )
        y_label_x = scales.x_left - 64
        y_label_y = (scales.y_top + scales.y_bottom) / 2
        parts.append(
            f'<text x="{y_label_x:.2f}" y="{y_label_y:.2f}" '
            f'text-anchor="middle" fill="{pal["text_primary"]}" '
            f'font-size="{typ["axis_label_px"]}" font-style="italic" '
            f'transform="rotate(-90 {y_label_x:.2f} {y_label_y:.2f})">'
            f'R<tspan baseline-shift="super" font-size="0.7em">2</tspan>'
            f'  (test set)</text>'
        )


# -------------------------------------------------------------------------
# SVG
# -------------------------------------------------------------------------
def build_svg(cfg, node3_data, node5_data):
    vp = cfg["viewport"]
    width = vp["width_px"]
    height = vp["height_px"]
    pal = cfg["palette"]
    typ = cfg["typography"]
    ov = cfg["overlay"]
    plt_cfg = cfg["plot"]

    margin = plt_cfg["margin"]
    panel_gap = plt_cfg["panel_gap_px"]
    plot_top = margin["top"]
    plot_bottom = height - margin["bottom"]
    plot_left = margin["left"]
    plot_right = width - margin["right"]
    panel_w = (plot_right - plot_left - panel_gap) / 2

    chip_y = 62

    panel1 = PanelScales(
        x_left=plot_left, x_right=plot_left + panel_w,
        y_top=plot_top, y_bottom=plot_bottom,
        y_min=plt_cfg["y_min"], y_max=plt_cfg["y_max"],
    )
    panel2 = PanelScales(
        x_left=plot_left + panel_w + panel_gap, x_right=plot_right,
        y_top=plot_top, y_bottom=plot_bottom,
        y_min=plt_cfg["y_min"], y_max=plt_cfg["y_max"],
    )

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="R^2 by calibration method and node">'
    )
    parts.append(
        f'<rect width="{width}" height="{height}" fill="{pal["background"]}" />'
    )

    # Legend chips, centred above the panels
    chip_w = ov["chip_swatch_w"]
    chip_h = ov["chip_swatch_h"]
    chip_gap = ov["chip_swatch_gap"]
    row_gap = ov["chip_row_gap"]
    char_w = typ["chip_label_px"] * 0.45    # Estimated glyph width (px), used to centre the row
    chip_segments = []
    total_w = 0
    for method in METHOD_ORDER:
        text = METHOD_LABEL[method]
        text_w = len(text) * char_w
        seg_w = chip_w + 10 + text_w
        chip_segments.append((method, text, seg_w))
        total_w += seg_w
    total_w += row_gap * (len(chip_segments) - 1)

    canvas_center = width / 2
    cursor_x = canvas_center - total_w / 2
    for method, text, seg_w in chip_segments:
        sw_color = pal[f"{method}_bar"]
        text_color = pal[f"{method}_text"]
        sw_x = cursor_x
        sw_y = chip_y - chip_h * 0.75
        parts.append(
            f'<rect x="{sw_x:.2f}" y="{sw_y:.2f}" '
            f'width="{chip_w}" height="{chip_h}" rx="3" ry="3" '
            f'fill="{sw_color}" />'
        )
        text_x = sw_x + chip_w + 10
        parts.append(
            f'<text x="{text_x:.2f}" y="{chip_y:.2f}" '
            f'text-anchor="start" fill="{text_color}" '
            f'font-size="{typ["chip_label_px"]}" font-style="italic" '
            f'font-weight="600">{text}</text>'
        )
        cursor_x += seg_w + row_gap

    _add_panel(parts, "node3", panel1, node3_data, cfg, draw_y_axis=True)
    _add_panel(parts, "node5", panel2, node5_data, cfg, draw_y_axis=False)

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
<title>R2 by method and node</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Crimson+Pro:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
  :root {{ --bg: {pal["background"]}; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); }}
  body {{
    font-family: {typ["font_stack"]};
    color: {pal["text_primary"]};
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
  }}
  .figure-wrap {{ width: {vp["width_px"]}px; background: var(--bg); padding: 0; }}
  svg text {{ font-family: {typ["font_stack"]}; letter-spacing: 0.01em; }}
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
    # Optional legend label for the ML bars
    METHOD_LABEL["ml"] = cfg.get("method_label_ml", METHOD_LABEL["ml"])

    def abspath(rel):
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p

    node3 = load_node_results(
        abspath(cfg["paths"]["node3_costs"]),
        abspath(cfg["paths"]["node3_ml"]),
    )
    node5 = load_node_results(
        abspath(cfg["paths"]["node5_costs"]),
        abspath(cfg["paths"]["node5_ml"]),
    )

    print("Node 3:")
    for gas, vals in node3.items():
        print(f"  {gas}: mfg={vals['manufacturer']:.3f}  ols={vals['ols']:.3f}  "
              f"ml={vals['ml']:.3f} ({vals['ml_model']}-{vals['ml_config']})")
    print("Node 5:")
    for gas, vals in node5.items():
        print(f"  {gas}: mfg={vals['manufacturer']:.3f}  ols={vals['ols']:.3f}  "
              f"ml={vals['ml']:.3f} ({vals['ml_model']}-{vals['ml_config']})")

    svg = build_svg(cfg, node3, node5)
    html = build_html(cfg, svg)

    html_out = REPO_ROOT / cfg["paths"]["output_html"]
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(html, encoding="utf-8")
    print(f"HTML: {html_out}")

    png_out = REPO_ROOT / cfg["paths"]["output_image"]
    png_out.parent.mkdir(parents=True, exist_ok=True)
    render_with_selenium(html_out, png_out, cfg)
    print(f"PNG : {png_out}")


if __name__ == "__main__":
    main()
