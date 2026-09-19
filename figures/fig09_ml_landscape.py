"""
Figure 9: test R2 of the ML combinations at both nodes

Reads tabla_maestra_v6.csv of Node 3 and of Node 5. Draws one heat map
per node, with a row per model and a column per target and feature
configuration; the value is printed in each cell. The PNG goes to
figures/output/.
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
               if "--config" in sys.argv else SCRIPT_DIR / "fig09_ml_landscape_config.json")

# -------------------------------------------------------------------------
# Labels
# -------------------------------------------------------------------------
MODEL_DISPLAY = {
    "DNN":      "DNN",
    "LSTM":     "LSTM",
    "LightGBM": "LGBM",
    "RF":       "RF",
    "XGBoost":  "XGB",
}
GAS_LABEL_HTML = {
    "PM2.5":     'PM<tspan baseline-shift="sub" font-size="0.72em">2.5</tspan>',
    "CO":        'CO',
    "NO2":       'NO<tspan baseline-shift="sub" font-size="0.72em">2</tspan> <tspan font-size="0.78em" font-style="italic">low</tspan>',
    "NO2_alpha": 'NO<tspan baseline-shift="sub" font-size="0.72em">2</tspan> <tspan font-size="0.78em" font-style="italic">mid</tspan>',
    "O3":        'O<tspan baseline-shift="sub" font-size="0.72em">3</tspan>',
}
GAS_SENSOR_LABEL = {
    "PM2.5":     "SPS30",
    "CO":        "GM-702B",
    "NO2":       "GM-102B",
    "NO2_alpha": "Alphasense B43F",
    "O3":        "MQ131",
}
NODE_TITLES = {
    "node3": "Node 3  ·  QMUL  ·  ref. TH2",
    "node5": "Node 5  ·  KEMP  ·  ref. TH7",
}


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_node_grid(csv_path: Path, models, gases, configs,
                   mid_csv_path: Path = None, mid_lstm_csv_path: Path = None):
    """Return {model: {gas: {config: test R2}}} for one node; missing cells are NaN."""
    df = pd.read_csv(csv_path)
    extra = []
    for p in (mid_csv_path, mid_lstm_csv_path):
        if p is not None and Path(p).exists():
            extra.append(pd.read_csv(p))
    if extra:
        df = pd.concat([df] + extra, ignore_index=True)
    pivot = df.pivot_table(
        index="model", columns=["gas", "config"], values="R2_test"
    )
    grid = {}
    for model in models:
        row = {}
        for gas in gases:
            row[gas] = {}
            for config in configs:
                try:
                    val = pivot.loc[model, (gas, config)]
                except KeyError:
                    val = float("nan")
                row[gas][config] = float(val)
        grid[model] = row
    return grid


# -------------------------------------------------------------------------
# Colours
# -------------------------------------------------------------------------
def _hx(s):
    return int(s, 16)


def _interp_color(c0, c1, t):
    """Interpolate linearly between two hex colours, clamping t to [0, 1]."""
    t = max(0.0, min(1.0, t))
    r0, g0, b0 = _hx(c0[1:3]), _hx(c0[3:5]), _hx(c0[5:7])
    r1, g1, b1 = _hx(c1[1:3]), _hx(c1[3:5]), _hx(c1[5:7])
    r = round(r0 + (r1 - r0) * t)
    g = round(g0 + (g1 - g0) * t)
    b = round(b0 + (b1 - b0) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def cell_fill(r2, pal, scale_max, neg_clip):
    """Return the cell fill: violet for R2 >= 0, grey down to -neg_clip, flat grey below."""
    if r2 != r2:
        return pal["nan_bg"]
    if r2 >= 0:
        t = min(r2 / scale_max, 1.0)
        return _interp_color(pal["pos_bg_min"], pal["pos_bg_max"], t)
    t = min(abs(r2) / neg_clip, 1.0)
    return _interp_color(pal["zero_bg"], pal["neg_bg_max"], t)


def cell_text_color(r2, fill_color, pal, scale_max, white_threshold):
    if r2 != r2:
        return pal["text_muted"]
    if r2 >= 0:
        t = r2 / scale_max
        return pal["text_on_dark"] if t >= white_threshold else pal["text_primary"]
    return pal["text_negative"]


def fmt_r2(r2):
    """Format R2 with fewer decimals as it grows more negative; NaN becomes a dash."""
    if r2 != r2:
        return "—"
    if abs(r2) < 0.005:
        return "0.00"
    if r2 <= -10:
        return f"{r2:.0f}"
    if r2 <= -1:
        return f"{r2:.1f}"
    return f"{r2:.2f}"


# -------------------------------------------------------------------------
# SVG
# -------------------------------------------------------------------------
def render_node_strip(parts, node_key, grid, x_left, y_top, cfg):
    """Draw the heat map of one node: one row per model, one group of columns per pollutant."""
    pal = cfg["palette"]
    typ = cfg["typography"]
    plt_cfg = cfg["plot"]
    ov = cfg["overlay"]

    cw = plt_cfg["cell_w_px"]
    ch = plt_cfg["cell_h_px"]
    rr = plt_cfg["cell_radius_px"]
    intra = plt_cfg["intra_group_gap_px"]
    inter = plt_cfg["inter_group_gap_px"]
    row_gap = plt_cfg["row_gap_px"]
    models = plt_cfg["model_order"]
    gases = plt_cfg["gas_order"]
    configs = plt_cfg["config_order"]
    n_cfgs = len(configs)
    group_w = n_cfgs * cw + (n_cfgs - 1) * intra

    title = NODE_TITLES[node_key]
    parts.append(
        f'<text x="{x_left:.2f}" y="{y_top - 10:.2f}" '
        f'text-anchor="start" fill="{pal["node_title"]}" '
        f'font-size="{typ["node_title_px"]}" font-style="italic" '
        f'font-weight="700">{title}</text>'
    )

    for mi, model in enumerate(models):
        ry = y_top + mi * (ch + row_gap)
        parts.append(
            f'<text x="{x_left - 8:.2f}" y="{ry + ch * 0.65:.2f}" '
            f'text-anchor="end" fill="{pal["model_label"]}" '
            f'font-size="{typ["model_label_px"]}" font-style="italic">'
            f'{MODEL_DISPLAY.get(model, model)}</text>'
        )
        for gi, gas in enumerate(gases):
            for ci, cfg_label in enumerate(configs):
                cx = (
                    x_left
                    + gi * (group_w + inter)
                    + ci * (cw + intra)
                )
                r2 = grid[model][gas][cfg_label]
                fill = cell_fill(
                    r2, pal,
                    scale_max=plt_cfg["color_scale_max"],
                    neg_clip=plt_cfg["neg_clip"],
                )
                txt_color = cell_text_color(
                    r2, fill, pal,
                    scale_max=plt_cfg["color_scale_max"],
                    white_threshold=ov["white_text_threshold"],
                )
                parts.append(
                    f'<rect x="{cx:.2f}" y="{ry:.2f}" '
                    f'width="{cw}" height="{ch}" rx="{rr}" ry="{rr}" '
                    f'fill="{fill}" stroke="{pal["cell_border"]}" '
                    f'stroke-width="{ov["cell_border_w"]}" />'
                )
                parts.append(
                    f'<text x="{cx + cw / 2:.2f}" y="{ry + ch * 0.66:.2f}" '
                    f'text-anchor="middle" fill="{txt_color}" '
                    f'font-size="{typ["cell_value_px"]}" '
                    f'font-family="\'Crimson Text\', serif">'
                    f'{fmt_r2(r2)}</text>'
                )


def render_bottom_axis_headers(parts, x_left, y_baseline, cfg):
    """Draw the config letters, then the pollutant and sensor labels, under the lower strip."""
    pal = cfg["palette"]
    typ = cfg["typography"]
    plt_cfg = cfg["plot"]

    cw = plt_cfg["cell_w_px"]
    intra = plt_cfg["intra_group_gap_px"]
    inter = plt_cfg["inter_group_gap_px"]
    gases = plt_cfg["gas_order"]
    configs = plt_cfg["config_order"]
    group_w = len(configs) * cw + (len(configs) - 1) * intra

    cfg_y = y_baseline
    gas_y = cfg_y + plt_cfg.get("gas_label_gap_px", 22)
    sensor_px = typ.get("sensor_label_px", 11)
    sensor_y = gas_y + plt_cfg.get("sensor_label_gap_px", 14)

    for gi, gas in enumerate(gases):
        gx = x_left + gi * (group_w + inter) + group_w / 2
        for ci, cfg_label in enumerate(configs):
            ccx = (
                x_left
                + gi * (group_w + inter)
                + ci * (cw + intra)
                + cw / 2
            )
            parts.append(
                f'<text x="{ccx:.2f}" y="{cfg_y:.2f}" '
                f'text-anchor="middle" fill="{pal["header_config"]}" '
                f'font-size="{typ["header_config_px"]}" font-style="italic">'
                f'{cfg_label}</text>'
            )
        parts.append(
            f'<text x="{gx:.2f}" y="{gas_y:.2f}" '
            f'text-anchor="middle" fill="{pal["header_gas"]}" '
            f'font-size="{typ["header_gas_px"]}" font-style="italic" '
            f'font-weight="600">{GAS_LABEL_HTML[gas]}</text>'
        )
        sensor_lbl = GAS_SENSOR_LABEL.get(gas)
        if sensor_lbl:
            parts.append(
                f'<text x="{gx:.2f}" y="{sensor_y:.2f}" '
                f'text-anchor="middle" fill="{pal["text_muted"]}" '
                f'font-size="{sensor_px}" font-style="italic">'
                f'{sensor_lbl}</text>'
            )


def render_legend(parts, x, y, cfg):
    """Draw the colour bar of the positive scale and the swatch of negative values."""
    pal = cfg["palette"]
    typ = cfg["typography"]
    ov = cfg["overlay"]
    plt_cfg = cfg["plot"]
    bw = ov["legend_bar_w"]
    bh = ov["legend_bar_h"]

    steps = 16    # Segments of the gradient bar
    seg_w = bw / steps
    for i in range(steps):
        t = (i + 0.5) / steps
        col = _interp_color(pal["pos_bg_min"], pal["pos_bg_max"], t)
        parts.append(
            f'<rect x="{x + i * seg_w:.2f}" y="{y:.2f}" '
            f'width="{seg_w + 0.5:.2f}" height="{bh}" fill="{col}" />'
        )
    parts.append(
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{bw}" height="{bh}" '
        f'fill="none" stroke="{pal["cell_border"]}" stroke-width="0.6" />'
    )
    parts.append(
        f'<text x="{x - 6:.2f}" y="{y + bh * 0.85:.2f}" '
        f'text-anchor="end" fill="{pal["legend_text"]}" '
        f'font-size="{typ["legend_label_px"]}" font-style="italic">'
        f'R<tspan baseline-shift="super" font-size="0.7em">2</tspan>=0</text>'
    )
    parts.append(
        f'<text x="{x + bw + 6:.2f}" y="{y + bh * 0.85:.2f}" '
        f'text-anchor="start" fill="{pal["legend_text"]}" '
        f'font-size="{typ["legend_label_px"]}" font-style="italic">'
        f'≥ {plt_cfg["color_scale_max"]:.2f}</text>'
    )
    swx = x + bw + 70
    parts.append(
        f'<rect x="{swx:.2f}" y="{y - 1:.2f}" '
        f'width="{bh + 2}" height="{bh + 2}" rx="2" ry="2" '
        f'fill="{pal["neg_bg_max"]}" stroke="{pal["cell_border"]}" '
        f'stroke-width="0.6" />'
    )
    parts.append(
        f'<text x="{swx + bh + 6:.2f}" y="{y + bh * 0.85:.2f}" '
        f'text-anchor="start" fill="{pal["legend_text"]}" '
        f'font-size="{typ["legend_label_px"]}" font-style="italic">'
        f'R<tspan baseline-shift="super" font-size="0.7em">2</tspan>&lt;0</text>'
    )


def build_svg(cfg, node3_grid, node5_grid):
    vp = cfg["viewport"]
    width, height = vp["width_px"], vp["height_px"]
    pal = cfg["palette"]
    typ = cfg["typography"]
    plt_cfg = cfg["plot"]
    margin = plt_cfg["margin"]

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="ML R^2 landscape: 60 combinations per node">'
    )
    parts.append(
        f'<rect width="{width}" height="{height}" fill="{pal["background"]}" />'
    )

    # Node 3 strip above Node 5; shared headers under the lower strip
    n_models = len(plt_cfg["model_order"])
    strip_h = (
        n_models * plt_cfg["cell_h_px"]
        + (n_models - 1) * plt_cfg["row_gap_px"]
    )
    x_left = margin["left"]
    y_top_n3 = margin["top"]
    render_node_strip(parts, "node3", node3_grid, x_left, y_top_n3, cfg)

    y_top_n5 = y_top_n3 + strip_h + plt_cfg["node_gap_px"]
    render_node_strip(parts, "node5", node5_grid, x_left, y_top_n5, cfg)

    bottom_axis_y = y_top_n5 + strip_h + plt_cfg.get("bottom_axis_gap_px", 18)
    render_bottom_axis_headers(parts, x_left, bottom_axis_y, cfg)

    # Legend in the top-right corner
    legend_x = width - margin["right"] - cfg["overlay"]["legend_bar_w"] - 130
    legend_y = 14
    render_legend(parts, legend_x, legend_y, cfg)

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
<title>ML R2 landscape</title>
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

    plt_cfg = cfg["plot"]
    # The mid-cost tables are optional and are merged when the config names them
    n3 = load_node_grid(
        abspath(cfg["paths"]["node3_ml"]),
        plt_cfg["model_order"], plt_cfg["gas_order"], plt_cfg["config_order"],
        mid_csv_path=abspath(cfg["paths"]["node3_ml_mid"])
            if "node3_ml_mid" in cfg["paths"] else None,
        mid_lstm_csv_path=abspath(cfg["paths"]["node3_ml_mid_lstm"])
            if "node3_ml_mid_lstm" in cfg["paths"] else None,
    )
    n5 = load_node_grid(
        abspath(cfg["paths"]["node5_ml"]),
        plt_cfg["model_order"], plt_cfg["gas_order"], plt_cfg["config_order"],
        mid_csv_path=abspath(cfg["paths"]["node5_ml_mid"])
            if "node5_ml_mid" in cfg["paths"] else None,
        mid_lstm_csv_path=abspath(cfg["paths"]["node5_ml_mid_lstm"])
            if "node5_ml_mid_lstm" in cfg["paths"] else None,
    )

    print("Node 3 (R^2 test):")
    for m, row in n3.items():
        flat = " | ".join(
            f"{g}({c})={fmt_r2(row[g][c])}"
            for g in plt_cfg["gas_order"] for c in plt_cfg["config_order"]
        )
        print(f"  {m:8s} {flat}")
    print("Node 5 (R^2 test):")
    for m, row in n5.items():
        flat = " | ".join(
            f"{g}({c})={fmt_r2(row[g][c])}"
            for g in plt_cfg["gas_order"] for c in plt_cfg["config_order"]
        )
        print(f"  {m:8s} {flat}")

    svg = build_svg(cfg, n3, n5)
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
