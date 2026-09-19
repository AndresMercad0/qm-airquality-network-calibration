"""
Figure 10: monthly R2 at Node 5

Reads metricas_ols_ml_mensual_v6.csv (PM2.5, CO and NO2), the O3 series
of metricas_drift_mensual.csv and the mid-cost NO2 series of
metricas_alphasense_no2_ols_mensual_v6.csv. Draws one panel per pollutant
with the test period shaded. The PNG goes to figures/output/.
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
               if "--config" in sys.argv else SCRIPT_DIR / "fig10_monthly_r2_config.json")
PLOT_FLAGS: dict = {}    # Filled in main() with the plot section of the config

# -------------------------------------------------------------------------
# Months of the study period
# -------------------------------------------------------------------------
MONTH_LABEL = {
    "2025-04": "Apr", "2025-05": "May", "2025-06": "Jun", "2025-07": "Jul",
    "2025-08": "Aug", "2025-09": "Sep", "2025-10": "Oct", "2025-11": "Nov",
    "2025-12": "Dec", "2026-01": "Jan", "2026-02": "Feb",
}
MONTH_ORDER = list(MONTH_LABEL.keys())


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _select_rows(csv_path: Path, gas: str, model: str, config: str, basis=None):
    """
    Return the rows of one model indexed by month, in study order.
    basis filters the files that carry that column; a config of '-' is not filtered.
    """
    df = pd.read_csv(csv_path)
    sub = df[(df["gas"] == gas) & (df["model"] == model)]
    if config != "-":
        sub = sub[sub["config"] == config]
    if basis is not None and "basis" in sub.columns:
        sub = sub[sub["basis"] == basis]
    return sub.set_index("month").reindex(MONTH_ORDER)


def load_panel_series(csv_path: Path, gas: str, model: str, config: str, basis=None):
    """Return [(month, R2)] for one model; months without data are NaN."""
    sub = _select_rows(csv_path, gas, model, config, basis)
    return [(m, sub.at[m, "R2"] if m in sub.index else float("nan"))
            for m in MONTH_ORDER]


def load_panel_n_obs(csv_path: Path, gas: str, model: str, config: str, basis=None):
    """Return {month: number of valid rows} when the file has an n_obs column."""
    sub = _select_rows(csv_path, gas, model, config, basis)
    if "n_obs" not in sub.columns:
        return {}
    return {m: (int(sub.at[m, "n_obs"]) if m in sub.index and sub.at[m, "n_obs"] == sub.at[m, "n_obs"] else None)
            for m in MONTH_ORDER}


def _fmt_n(n):
    """Format a row count, in thousands from 1000 up."""
    if n is None:
        return "--"
    return f"{n / 1000:.1f}k" if n >= 1000 else f"{n}"


# -------------------------------------------------------------------------
# SVG
# -------------------------------------------------------------------------
def _draw_out_of_range(parts, series, sx, y_floor, color, font_px):
    """Mark points below the axis minimum with a triangle on the axis floor and their value."""
    for idx, (_m, v) in enumerate(series):
        if v != v:
            continue
        if v < _OUT_OF_RANGE_MIN[0]:
            x = sx(idx)
            h = 7.0
            parts.append(
                f'<polygon points="{x - h:.2f},{y_floor - h - 1:.2f} {x + h:.2f},{y_floor - h - 1:.2f} '
                f'{x:.2f},{y_floor - 1:.2f}" fill="{color}" opacity="0.9" />'
            )
            parts.append(
                f'<text x="{x:.2f}" y="{y_floor - h - 5:.2f}" text-anchor="middle" fill="{color}" '
                f'font-size="{font_px * 0.86:.1f}" font-weight="600">{v:.2f}</text>'
            )


_OUT_OF_RANGE_MIN = [-0.5]    # Axis minimum, updated by render_panel


def _draw_series(parts, series, sx, sy, line_color, marker_color,
                 marker_stroke, line_w, marker_r, marker_stroke_w):
    """Draw a polyline with circle markers, broken at missing values."""
    pts = []
    segments = []
    for idx, (_, val) in enumerate(series):
        if val == val:
            pts.append((sx(idx), sy(val)))
        else:
            if len(pts) >= 2:
                segments.append(pts)
            pts = []
    if len(pts) >= 2:
        segments.append(pts)
    for seg in segments:
        d = " ".join(f"{x:.2f},{y:.2f}" for (x, y) in seg)
        parts.append(
            f'<polyline points="{d}" fill="none" stroke="{line_color}" '
            f'stroke-width="{line_w}" stroke-linejoin="round" />'
        )
    for idx, (_, val) in enumerate(series):
        if val != val:
            continue
        cx, cy = sx(idx), sy(val)
        parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{marker_r}" '
            f'fill="{marker_color}" stroke="{marker_stroke}" '
            f'stroke-width="{marker_stroke_w}" />'
        )


def render_panel(parts, panel, series, x_left, x_right, y_top, y_bottom, cfg,
                 panel_letter=None, extra_series=None):
    pal = cfg["palette"]
    typ = cfg["typography"]
    plt_cfg = cfg["plot"]
    ov = cfg["overlay"]

    y_min = plt_cfg["y_min"]
    y_max = plt_cfg["y_max"]
    _OUT_OF_RANGE_MIN[0] = y_min
    plot_w = x_right - x_left
    plot_h = y_bottom - y_top

    n_months = len(series)
    # Months evenly spaced along x
    def sx(idx):
        if n_months == 1:
            return (x_left + x_right) / 2
        return x_left + idx * (plot_w / (n_months - 1))

    def sy(value):
        f = (value - y_min) / (y_max - y_min)
        return y_bottom - f * plot_h

    # Shaded test period, bounded halfway between the last training and the first test month
    boundary_month = plt_cfg["test_period_start_month"]
    if boundary_month in MONTH_ORDER:
        b_idx = MONTH_ORDER.index(boundary_month)
        if b_idx > 0:
            boundary_x = (sx(b_idx - 1) + sx(b_idx)) / 2
        else:
            boundary_x = sx(0)
        parts.append(
            f'<rect x="{boundary_x:.2f}" y="{y_top:.2f}" '
            f'width="{x_right - boundary_x:.2f}" height="{plot_h:.2f}" '
            f'fill="{pal["test_bg"]}" />'
        )

    for tv in plt_cfg["y_tick_values"]:
        gy = sy(tv)
        parts.append(
            f'<line x1="{x_left:.2f}" y1="{gy:.2f}" '
            f'x2="{x_right:.2f}" y2="{gy:.2f}" '
            f'stroke="{pal["grid_line"]}" stroke-width="{ov["grid_linewidth"]}" '
            f'stroke-dasharray="1 4" />'
        )
    # Zero line
    if y_min < 0 < y_max:
        gy = sy(0.0)
        parts.append(
            f'<line x1="{x_left:.2f}" y1="{gy:.2f}" '
            f'x2="{x_right:.2f}" y2="{gy:.2f}" '
            f'stroke="{pal["zero_line"]}" stroke-width="0.9" />'
        )

    if boundary_month in MONTH_ORDER:
        parts.append(
            f'<line x1="{boundary_x:.2f}" y1="{y_top:.2f}" '
            f'x2="{boundary_x:.2f}" y2="{y_bottom:.2f}" '
            f'stroke="{pal["boundary_line"]}" stroke-width="1.0" '
            f'stroke-dasharray="{ov["boundary_dash"]}" />'
        )

    # Points outside the axis are left out of the line and drawn as out-of-range marks
    def _clip(s):
        out = []
        for m, v in s:
            if v != v or v < y_min or v > y_max:
                out.append((m, float("nan")))
            else:
                out.append((m, v))
        return out

    line_color = pal[panel.get("line_color_key", "line")]
    marker_fill_color = pal[panel.get("marker_fill_key", "marker_fill")]
    line_w = plt_cfg["line_width"]
    r = plt_cfg["marker_radius"]
    _draw_series(
        parts, _clip(series), sx, sy,
        line_color, marker_fill_color, pal["marker_stroke"],
        line_w, r, ov["marker_stroke_w"],
    )

    _draw_out_of_range(parts, series, sx, y_bottom, line_color, typ["tick_px"])

    # Optional second series on the same axes
    es_spec = panel.get("extra_series")
    if extra_series is not None and es_spec:
        es_line = pal[es_spec.get("line_color_key", "line")]
        es_marker = pal[es_spec.get("marker_fill_key", "marker_fill")]
        _draw_series(
            parts, _clip(extra_series), sx, sy,
            es_line, es_marker, pal["marker_stroke"],
            line_w, r, ov["marker_stroke_w"],
        )

    if extra_series is not None and es_spec:
        _draw_out_of_range(parts, extra_series, sx, y_bottom, pal[es_spec.get("line_color_key", "line")], typ["tick_px"])

    parts.append(
        f'<line x1="{x_left:.2f}" y1="{y_bottom:.2f}" '
        f'x2="{x_right:.2f}" y2="{y_bottom:.2f}" '
        f'stroke="{pal["axis_line"]}" stroke-width="{ov["axis_linewidth"]}" />'
    )
    parts.append(
        f'<line x1="{x_left:.2f}" y1="{y_top:.2f}" '
        f'x2="{x_left:.2f}" y2="{y_bottom:.2f}" '
        f'stroke="{pal["axis_line"]}" stroke-width="{ov["axis_linewidth"]}" />'
    )

    for tv in plt_cfg["y_tick_values"]:
        gy = sy(tv)
        parts.append(
            f'<text x="{x_left - 6:.2f}" y="{gy + typ["tick_px"] * 0.35:.2f}" '
            f'text-anchor="end" fill="{pal["text_secondary"]}" '
            f'font-size="{typ["tick_px"]}">{tv:g}</text>'
        )

    # Month labels, rotated
    skip = max(1, plt_cfg.get("x_tick_skip", 1))
    for idx, (mkey, _) in enumerate(series):
        if idx % skip != 0 and idx != n_months - 1:
            continue
        tx = sx(idx)
        ty = y_bottom + 12
        parts.append(
            f'<text x="{tx:.2f}" y="{ty:.2f}" '
            f'text-anchor="end" fill="{pal["text_secondary"]}" '
            f'font-size="{typ["tick_px"]}" '
            f'transform="rotate(-30 {tx:.2f} {ty:.2f})">'
            f'{MONTH_LABEL[mkey]}</text>'
        )

    # Valid rows per month, under the month labels
    n_obs = panel.get("_n_obs")
    if plt_cfg.get("annotate_n_obs") and n_obs:
        n_px = typ.get("n_obs_px", 11)
        n_dy = plt_cfg.get("n_obs_dy", 34)
        for idx, (mkey, _) in enumerate(series):
            parts.append(
                f'<text x="{sx(idx):.2f}" y="{y_bottom + n_dy:.2f}" '
                f'text-anchor="middle" fill="{pal["text_muted"]}" '
                f'font-size="{n_px}">{_fmt_n(n_obs.get(mkey))}</text>'
            )
        if plt_cfg.get("n_obs_label") and panel_letter == "a":
            parts.append(
                f'<text x="{x_left - 6:.2f}" y="{y_bottom + n_dy:.2f}" '
                f'text-anchor="end" fill="{pal["text_muted"]}" '
                f'font-size="{n_px}" font-style="italic">{plt_cfg["n_obs_label"]}</text>'
            )

    # Panel title: pollutant, sensor and model
    title_x = (x_left + x_right) / 2
    title_y = y_top - 12
    model_label_color = pal[panel.get("model_label_color_key", "model_label")]
    parts.append(
        f'<text x="{title_x:.2f}" y="{title_y:.2f}" '
        f'text-anchor="middle" fill="{pal["panel_title"]}" '
        f'font-size="{typ["panel_title_px"]}" font-weight="600">'
        f'{panel["title_html"]}'
        f'<tspan fill="{pal["text_secondary"]}" font-weight="400" '
        f'font-style="italic">  ·  {panel["sensor"]}  ·  </tspan>'
        f'<tspan fill="{model_label_color}" font-weight="700" '
        f'font-style="italic">{panel["model_label"]}</tspan>'
        f'</text>'
    )

    # Panel letter in the lower-left corner
    if panel_letter is not None:
        pl_size = typ.get("panel_letter_px", 20)
        parts.append(
            f'<text x="{x_left + 8:.2f}" y="{y_bottom - 8:.2f}" '
            f'text-anchor="start" fill="{pal["text_primary"]}" '
            f'font-size="{pl_size}" font-style="italic" font-weight="700">'
            f'({panel_letter})</text>'
        )

    # Two-entry legend when a second series is drawn
    if extra_series is not None and es_spec and PLOT_FLAGS.get("show_mini_legend", True):
        leg_size = typ.get("mini_legend_px", 13)
        sample_w = 18
        gap = 6
        line_h = leg_size * 1.35
        leg_y_1 = y_top + leg_size + 4
        leg_y_2 = leg_y_1 + line_h
        labels = [
            (es_spec.get("primary_label", panel["model_label"]),
             line_color, leg_y_1),
            (es_spec.get("extra_label", "OLS"),
             es_line, leg_y_2),
        ]
        for label, col, ly in labels:
            sx_end = x_right - 8
            sx_start = sx_end - len(label) * leg_size * 0.46
            line_x_end = sx_start - gap
            line_x_start = line_x_end - sample_w
            parts.append(
                f'<line x1="{line_x_start:.2f}" y1="{ly - leg_size * 0.32:.2f}" '
                f'x2="{line_x_end:.2f}" y2="{ly - leg_size * 0.32:.2f}" '
                f'stroke="{col}" stroke-width="{plt_cfg["line_width"]}" />'
            )
            parts.append(
                f'<circle cx="{(line_x_start + line_x_end) / 2:.2f}" '
                f'cy="{ly - leg_size * 0.32:.2f}" r="{plt_cfg["marker_radius"]}" '
                f'fill="{col}" stroke="{pal["marker_stroke"]}" '
                f'stroke-width="{ov["marker_stroke_w"]}" />'
            )
            parts.append(
                f'<text x="{sx_end:.2f}" y="{ly:.2f}" '
                f'text-anchor="end" fill="{pal["text_secondary"]}" '
                f'font-size="{leg_size}" font-style="italic">{label}</text>'
            )


def build_svg(cfg, panel_data):
    vp = cfg["viewport"]
    width, height = vp["width_px"], vp["height_px"]
    pal = cfg["palette"]
    typ = cfg["typography"]
    plt_cfg = cfg["plot"]
    margin = plt_cfg["margin"]

    plot_left = margin["left"]
    plot_right = width - margin["right"]
    plot_top = margin["top"]
    plot_bottom = height - margin["bottom"]

    h_gap = plt_cfg["panel_h_gap_px"]
    n_panels = len(panel_data)
    panel_w = (plot_right - plot_left - (n_panels - 1) * h_gap) / n_panels
    panel_h = plot_bottom - plot_top

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Monthly R2 drift, Node 5">'
    )
    parts.append(
        f'<rect width="{width}" height="{height}" fill="{pal["background"]}" />'
    )

    # One row of panels
    panel_letters = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for i, item in enumerate(panel_data):
        if len(item) == 3:
            panel, series, extra = item
        else:
            panel, series = item
            extra = None
        x_left = plot_left + i * (panel_w + h_gap)
        x_right = x_left + panel_w
        y_top = plot_top
        y_bottom = y_top + panel_h
        render_panel(
            parts, panel, series, x_left, x_right, y_top, y_bottom, cfg,
            panel_letter=panel_letters[i] if i < len(panel_letters) else None,
            extra_series=extra,
        )

    # Shared y label
    y_lbl_x = 22
    y_lbl_y = (plot_top + plot_bottom) / 2
    parts.append(
        f'<text x="{y_lbl_x:.2f}" y="{y_lbl_y:.2f}" '
        f'text-anchor="middle" fill="{pal["text_primary"]}" '
        f'font-size="{typ["axis_label_px"]}" font-style="italic" '
        f'font-weight="600" '
        f'transform="rotate(-90 {y_lbl_x:.2f} {y_lbl_y:.2f})">'
        f'Monthly R<tspan baseline-shift="super" font-size="0.7em">2</tspan></text>'
    )

    # Optional legend of the train/test boundary
    if not PLOT_FLAGS.get("show_bottom_legend", True):
        parts.append("</svg>")
        return "".join(parts)
    legend_text = "train / test boundary (Oct → Nov 2025); shaded area = test period"
    leg_y = height - 10
    text_w_est = len(legend_text) * typ["boundary_label_px"] * 0.46
    line_sample_w = 26
    gap_line_text = 10
    block_w = line_sample_w + gap_line_text + text_w_est
    block_x = (width - block_w) / 2
    line_y = leg_y - typ["boundary_label_px"] * 0.35
    parts.append(
        f'<line x1="{block_x:.2f}" y1="{line_y:.2f}" '
        f'x2="{block_x + line_sample_w:.2f}" y2="{line_y:.2f}" '
        f'stroke="{pal["boundary_line"]}" stroke-width="1.4" '
        f'stroke-dasharray="3 3" />'
    )
    parts.append(
        f'<text x="{block_x + line_sample_w + gap_line_text:.2f}" y="{leg_y:.2f}" '
        f'text-anchor="start" fill="{pal["text_secondary"]}" '
        f'font-size="{typ["boundary_label_px"]}" font-style="italic">'
        f'{legend_text}</text>'
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
<title>Monthly R2, Node 5</title>
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

    PLOT_FLAGS.update(cfg.get("plot", {}))
    csv_path = Path(cfg["paths"]["node5_drift"])
    if not csv_path.is_absolute():
        csv_path = REPO_ROOT / csv_path

    panel_data = []
    for spec in cfg["panels"]:
        # A panel may name its own CSV; otherwise the monthly drift file of the node is used
        panel_csv = Path(spec["csv"]) if spec.get("csv") else csv_path
        if not panel_csv.is_absolute():
            panel_csv = REPO_ROOT / panel_csv
        series = load_panel_series(
            panel_csv, spec["gas"], spec["model"], spec["config"]
        )
        extra = None
        es_spec = spec.get("extra_series")
        if es_spec:
            es_csv = Path(es_spec["csv"]) if es_spec.get("csv") else csv_path
            if not es_csv.is_absolute():
                es_csv = REPO_ROOT / es_csv
            extra = load_panel_series(
                es_csv, es_spec["gas"], es_spec["model"], es_spec["config"],
                basis=es_spec.get("basis"),
            )
        if cfg["plot"].get("annotate_n_obs"):
            spec["_n_obs"] = load_panel_n_obs(panel_csv, spec["gas"], spec["model"], spec["config"])
        panel_data.append((spec, series, extra))
        flat = " ".join(
            f"{MONTH_LABEL[m]}={v:+.2f}" if v == v else f"{MONTH_LABEL[m]}=--"
            for (m, v) in series
        )
        print(f"  {spec['gas']:6s} {spec['model']}-{spec['config']}  {flat}")
        if extra is not None:
            flat_e = " ".join(
                f"{MONTH_LABEL[m]}={v:+.2f}" if v == v else f"{MONTH_LABEL[m]}=--"
                for (m, v) in extra
            )
            print(f"    extra: {es_spec['gas']:6s} {es_spec['model']}-{es_spec['config']}  {flat_e}")

    svg = build_svg(cfg, panel_data)
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
