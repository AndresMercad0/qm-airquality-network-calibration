"""
Figure 11: cross-node transfer of the selected models

Reads results/cross_node/best_transfer_source_selected_v6.csv, written by
experiments/build_fig11_input.py. Draws, for each transfer direction and
pollutant, the R2 at the source node beside the R2 at the destination
node, with the model name above. The PNG goes to figures/output/.
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
               if "--config" in sys.argv else SCRIPT_DIR / "fig11_cross_node_transfer_config.json")


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


def load_best_transfer(csv_path: Path) -> tuple[dict, dict]:
    """
    Return two {gas: values} dicts: Node 3 to Node 5, then Node 5 to Node 3.
    Columns read: direction, gas, model, config, R2_local, R2_transferred, delta_R2.
    """
    df = pd.read_csv(csv_path)

    def _to_dict(direction: str) -> dict:
        sub = df[df["direction"] == direction]
        out = {}
        for _, row in sub.iterrows():
            out[row["gas"]] = {
                "local":       float(row["R2_local"]),
                "transferred": float(row["R2_transferred"]),
                "delta":       float(row["delta_R2"]),
                "model":       str(row["model"]),
                "config":      str(row["config"]),
            }
        return out

    return _to_dict("N3_to_N5"), _to_dict("N5_to_N3")


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


# -------------------------------------------------------------------------
# Panel rendering
# -------------------------------------------------------------------------
PANEL_TITLES = {
    "n3_to_n5": ("Node 3 → Node 5",
                 "trained at Node 3, evaluated at Node 5"),
    "n5_to_n3": ("Node 5 → Node 3",
                 "trained at Node 5, evaluated at Node 3"),
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

BAR_ORDER = ("local", "transferred")
BAR_LABEL = {
    "local":       "R² local",
    "transferred": "R² transferred",
}


def _format_truncated_value(r2):
    """Format the true value of a bar cut at the axis minimum."""
    av = abs(r2)
    if av >= 1000:
        return f"{r2:,.0f}"
    if av >= 10:
        return f"{r2:.1f}"
    return f"{r2:.2f}"


def _add_panel(parts, panel_key, scales, panel_data, cfg, draw_y_axis=True):
    pal = cfg["palette"]
    typ = cfg["typography"]
    ov = cfg["overlay"]
    plt_cfg = cfg["plot"]

    bar_w = plt_cfg["bar_width_px"]
    intra = plt_cfg["intra_gap_px"]
    group_w = 2 * bar_w + intra
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

    # Panel title and subtitle: training node and evaluation node
    layout = cfg.get("layout", {})
    title, sub = PANEL_TITLES[panel_key]
    title_x = (scales.x_left + scales.x_right) / 2
    title_y = layout.get("title_row_y_px", scales.y_top - 56)
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
        for bi, role in enumerate(BAR_ORDER):
            x = x_first_bar_left + bi * (bar_w + intra)
            r2 = panel_data[gas][role]
            color = pal[f"{role}_bar"]

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
                    f'fill="{pal["text_muted"]}" opacity="0.8" />'
                )

            bar_geoms.append({
                "x_center": x + bar_w / 2,
                "x_left": x,
                "top": top,
                "bottom": bottom,
                "r2": r2,
                "role": role,
                "gi": gi,
                "gas": gas,
                "truncated": truncated,
            })

    # Second pass: value labels
    role_label_specs = {
        "local":       ("local_text",       "value_label_px", "600"),
        "transferred": ("transferred_text", "value_label_px", "700"),
    }

    by_group = {}
    for g in bar_geoms:
        by_group.setdefault(g["gi"], []).append(g)

    in_bar_min_h = plt_cfg.get("in_bar_label_min_height_px", 30)
    for gi, geoms in by_group.items():
        for g in geoms:
            role = g["role"]
            r2 = g["r2"]
            color_key, font_key, weight = role_label_specs[role]
            font_size = typ[font_key]
            color = pal[color_key]
            bar_h = g["bottom"] - g["top"]

            # Labels: inside truncated and tall negative bars, above positive, below short negative
            if g["truncated"]:
                txt = _format_truncated_value(r2)
                label_y = scales.sy(y_min) - 6
                parts.append(
                    f'<text x="{g["x_center"]:.2f}" y="{label_y:.2f}" '
                    f'text-anchor="middle" fill="#FFFFFF" '
                    f'font-size="{font_size}" font-weight="700">{txt}</text>'
                )
            elif r2 >= 0:
                txt = f"{(0.0 if abs(r2) < 0.005 else r2):.2f}"
                label_y = g["top"] - 6
                parts.append(
                    f'<text x="{g["x_center"]:.2f}" y="{label_y:.2f}" '
                    f'text-anchor="middle" fill="{color}" '
                    f'font-size="{font_size}" font-weight="{weight}">{txt}</text>'
                )
            elif bar_h >= in_bar_min_h:
                txt = f"{(0.0 if abs(r2) < 0.005 else r2):.2f}"
                label_y = g["bottom"] - 6
                inner_color = "#FFFFFF" if role == "transferred" else color
                weight_in = "700" if role == "transferred" else weight
                parts.append(
                    f'<text x="{g["x_center"]:.2f}" y="{label_y:.2f}" '
                    f'text-anchor="middle" fill="{inner_color}" '
                    f'font-size="{font_size}" font-weight="{weight_in}">{txt}</text>'
                )
            else:
                txt = f"{(0.0 if abs(r2) < 0.005 else r2):.2f}"
                label_y = g["bottom"] + font_size + 2
                parts.append(
                    f'<text x="{g["x_center"]:.2f}" y="{label_y:.2f}" '
                    f'text-anchor="middle" fill="{color}" '
                    f'font-size="{font_size}" font-weight="{weight}">{txt}</text>'
                )

    # Name of the transferred model above each pair of bars
    model_band_offset = plt_cfg.get("model_callout_band_offset_px", 14)
    model_band_y = scales.y_top - model_band_offset

    callout_font = typ.get("model_callout_px", 20)
    callout_color = pal["transferred_text"]
    for gi, gas in enumerate(pollutants):
        geoms = by_group[gi]
        local_geom = next(g for g in geoms if g["role"] == "local")
        transferred_geom = next(g for g in geoms if g["role"] == "transferred")
        model = panel_data[gas]["model"]
        cfg_label = panel_data[gas]["config"]
        text = f"{MODEL_DISPLAY.get(model, model)}-{cfg_label}"
        cx = (local_geom["x_center"] + transferred_geom["x_center"]) / 2
        parts.append(
            f'<text x="{cx:.2f}" y="{model_band_y:.2f}" text-anchor="middle" '
            f'fill="{callout_color}" font-size="{callout_font}" '
            f'font-style="italic" font-weight="600">{text}</text>'
        )

    # Pollutant and sensor labels under the axis
    pollutant_band_offset = plt_cfg.get("pollutant_band_offset_px", 55)
    pollutant_band_y = scales.y_bottom + pollutant_band_offset

    for gi, gas in enumerate(pollutants):
        cx = group_centers[gi]
        label_html, _ = POLLUTANT_LABELS[gas]
        parts.append(
            f'<text x="{cx:.2f}" y="{pollutant_band_y:.2f}" '
            f'text-anchor="middle" fill="{pal["text_primary"]}" '
            f'font-size="{typ["pollutant_label_px"]}">{label_html}</text>'
        )
        sensor_y = pollutant_band_y + typ["sensor_label_px"] * 1.5
        parts.append(
            f'<text x="{cx:.2f}" y="{sensor_y:.2f}" '
            f'text-anchor="middle" fill="{pal["text_secondary"]}" '
            f'font-size="{typ["sensor_label_px"]}" font-style="italic">'
            f'{SENSOR_NAMES[gas]}</text>'
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
def build_svg(cfg, n3_to_n5, n5_to_n3):
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

    layout_cfg = cfg.get("layout", {})
    chip_y = layout_cfg.get("chip_row_y_px", 62)

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
        f'role="img" aria-label="Cross-node transfer R^2 by direction and pollutant">'
    )
    parts.append(
        f'<rect width="{width}" height="{height}" fill="{pal["background"]}" />'
    )

    # Legend chips, centred above the panels
    chip_w = ov["chip_swatch_w"]
    chip_h = ov["chip_swatch_h"]
    row_gap = ov["chip_row_gap"]
    char_w = typ["chip_label_px"] * 0.45
    chip_segments = []
    total_w = 0
    for role in BAR_ORDER:
        text = BAR_LABEL[role]
        text_w = len(text) * char_w
        seg_w = chip_w + 10 + text_w
        chip_segments.append((role, text, seg_w))
        total_w += seg_w
    total_w += row_gap * (len(chip_segments) - 1)

    canvas_center = width / 2
    cursor_x = canvas_center - total_w / 2
    for role, text, seg_w in chip_segments:
        sw_color = pal[f"{role}_bar"]
        text_color = pal[f"{role}_text"]
        sw_x = cursor_x
        sw_y = chip_y - chip_h * 0.75
        parts.append(
            f'<rect x="{sw_x:.2f}" y="{sw_y:.2f}" '
            f'width="{chip_w}" height="{chip_h}" rx="3" ry="3" '
            f'fill="{sw_color}" />'
        )
        text_x = sw_x + chip_w + 10
        # R2 is written with a superscript, as in the axis label
        if text.startswith("R²"):
            tail = text[2:]
            tspan = (
                f'<text x="{text_x:.2f}" y="{chip_y:.2f}" '
                f'text-anchor="start" fill="{text_color}" '
                f'font-size="{typ["chip_label_px"]}" font-style="italic" '
                f'font-weight="600">'
                f'R<tspan baseline-shift="super" font-size="0.7em">2</tspan>'
                f'{tail}</text>'
            )
        else:
            tspan = (
                f'<text x="{text_x:.2f}" y="{chip_y:.2f}" '
                f'text-anchor="start" fill="{text_color}" '
                f'font-size="{typ["chip_label_px"]}" font-style="italic" '
                f'font-weight="600">{text}</text>'
            )
        parts.append(tspan)
        cursor_x += seg_w + row_gap

    _add_panel(parts, "n3_to_n5", panel1, n3_to_n5, cfg, draw_y_axis=True)
    _add_panel(parts, "n5_to_n3", panel2, n5_to_n3, cfg, draw_y_axis=False)


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
<title>Cross-node transfer</title>
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

    def abspath(rel):
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p

    n3_to_n5, n5_to_n3 = load_best_transfer(
        abspath(cfg["paths"]["best_transfer_csv"])
    )

    print("Node 3 → Node 5 (trained at N3, evaluated at N5):")
    for gas, vals in n3_to_n5.items():
        print(f"  {gas:6} local={vals['local']:+.3f}  transferred={vals['transferred']:+.3f}  "
              f"ΔR²={vals['delta']:+.3f}  ({vals['model']}-{vals['config']})")
    print("Node 5 → Node 3 (trained at N5, evaluated at N3):")
    for gas, vals in n5_to_n3.items():
        print(f"  {gas:6} local={vals['local']:+.3f}  transferred={vals['transferred']:+.3f}  "
              f"ΔR²={vals['delta']:+.3f}  ({vals['model']}-{vals['config']})")

    svg = build_svg(cfg, n3_to_n5, n5_to_n3)
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
