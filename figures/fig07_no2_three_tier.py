"""
Figure 7: NO2 per cost tier at Node 3 (R2, Pearson r and mean bias)

Reads results/cross_node/single_basis_metrics_node3.csv (period test,
basis intersection). Draws one panel per metric with the manufacturer,
OLS and ML bars of the low-cost and mid-cost sensors; the reference is
drawn hollow. The PNG goes to figures/output/.
"""

import sys
from pathlib import Path

import pandas as pd

from fig07_helpers import PanelScales, _format_truncated, build_html, load_config
from render import render_with_selenium

# -------------------------------------------------------------------------
# Paths and configuration
# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Default config next to the script; --config <path> overrides it
CONFIG_PATH = (Path(sys.argv[sys.argv.index("--config") + 1]).resolve()
               if "--config" in sys.argv else SCRIPT_DIR / "fig07_no2_three_tier_config.json")

# -------------------------------------------------------------------------
# Legend order and tier labels
# -------------------------------------------------------------------------
CHIP_ORDER = ("manufacturer", "ols", "ml", "reference")
SUB_LABELS = [("GM-102B", "low-cost"), ("Alphasense B43F", "mid-cost"), ("LAQN ref.", "high-cost")]


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------
def load_metrics(csv_path: Path, ml_low: tuple, ml_mid: tuple) -> dict:
    """Return {metric: {tier: {method: value}}} for NO2 on the test rows shared by all methods."""
    d = pd.read_csv(csv_path)
    t = d[(d["period"] == "test") & (d["basis"] == "intersection")]

    def row(gas, method, model=None, config=None, clip=None):
        g = t[(t["gas"] == gas) & (t["method"] == method)]
        if model is not None:
            g = g[(g["model"] == model) & (g["config"] == config)]
        if clip is not None and "clip_zero" in g.columns:
            g = g[g["clip_zero"] == clip]
        if g.empty:
            raise SystemExit(f"no row for {gas}/{method}/{model}-{config}")
        return g.iloc[0]

    # Low-cost GM-102B and mid-cost Alphasense cell, three methods each
    spec = {
        "low_cost": {"manufacturer": row("NO2", "manufacturer_curve"),
                     "ols": row("NO2", "ols_train_only"),
                     "ml": row("NO2", "ml", *ml_low)},
        "mid_cost": {"manufacturer": row("NO2_alpha", "alphasense_factory", clip=False),
                     "ols": row("NO2_alpha", "ols_train_only"),
                     "ml": row("NO2_alpha", "ml", *ml_mid)},
    }
    out = {"r2": {}, "r": {}, "mbe": {}}
    for sub, methods in spec.items():
        out["r2"][sub] = {m: float(r["R2"]) for m, r in methods.items()}
        out["r"][sub] = {m: float(r["pearson_r"]) for m, r in methods.items()}
        out["mbe"][sub] = {m: float(r["MBE"]) for m, r in methods.items()}
    # The reference scores 1, 1 and 0 against itself by definition
    for metric, ref in (("r2", 1.0), ("r", 1.0), ("mbe", 0.0)):
        out[metric]["high_cost"] = {"reference": ref}
    out["ml_labels"] = {"low_cost": f"{ml_low[0]}-{ml_low[1]}", "mid_cost": f"{ml_mid[0]}-{ml_mid[1]}"}
    return out


# -------------------------------------------------------------------------
# Panel
# -------------------------------------------------------------------------
def fmt_value(metric_cfg, v):
    """Format a bar value (signed for the bias) with a typographic minus."""
    if metric_cfg["key"] == "mbe":
        s = f"{v:+.1f}"
    else:
        s = f"{v:.2f}"
    return s.replace("-", "−")


def add_metric_panel(parts, scales, values, cfg, mcfg):
    """Draw one metric panel: three tiers, one bar per method."""
    pal, typ, ov, plt_cfg = cfg["palette"], cfg["typography"], cfg["overlay"], cfg["plot"]
    bar_w, intra = plt_cfg["bar_width_px"], plt_cfg["intra_gap_px"]
    corner_r = plt_cfg.get("bar_corner_radius_px", 0)
    inner_pad = plt_cfg.get("panel_inner_pad_px", 0)
    inner_left, inner_right = scales.x_left + inner_pad, scales.x_right - inner_pad
    col_w = (inner_right - inner_left) / 3
    sub_centers = [inner_left + col_w * (i + 0.5) for i in range(3)]
    sub_specs = [("low_cost", ("manufacturer", "ols", "ml")), ("mid_cost", ("manufacturer", "ols", "ml")),
                 ("high_cost", ("reference",))]
    y_min, y_max = mcfg["y_min"], mcfg["y_max"]

    # Panel title and grid
    tx = (scales.x_left + scales.x_right) / 2
    ty = scales.y_top - typ["panel_subtitle_px"] * 1.2
    parts.append(f'<text x="{tx:.2f}" y="{ty:.2f}" text-anchor="middle" fill="{pal["panel_subtitle"]}" '
                 f'font-size="{typ["panel_title_px"]}" font-weight="700" font-style="italic">{mcfg["title"]}</text>')
    for yv in mcfg["ticks"]:
        gy = scales.sy(yv)
        parts.append(f'<line x1="{scales.x_left:.2f}" y1="{gy:.2f}" x2="{scales.x_right:.2f}" y2="{gy:.2f}" '
                     f'stroke="{pal["grid_line"]}" stroke-width="{ov["grid_linewidth"]}" stroke-dasharray="1 4" />')
    # Bars
    y_zero = scales.sy(0.0)
    font = typ["value_label_px"]
    for (sub_key, methods), cx in zip(sub_specs, sub_centers):
        n = len(methods)
        group_w = n * bar_w + (n - 1) * intra
        x_first = cx - group_w / 2
        for mi, method in enumerate(methods):
            x = x_first + mi * (bar_w + intra)
            color, color_text = pal[f"{method}_bar"], pal[f"{method}_text"]
            v = values[sub_key][method]
            trunc_low, trunc_high = v < y_min, v > y_max
            display = y_min if trunc_low else (y_max if trunc_high else v)
            yv = scales.sy(display)
            top, bottom = min(y_zero, yv), max(y_zero, yv)
            height = bottom - top
            definitional = method == "reference"
            # The reference is drawn hollow; a zero-height bar becomes a short mark on the zero line
            if definitional:
                if height < 4:
                    top, height = y_zero - 3, 6
                r = min(corner_r, height / 2.0)
                parts.append(f'<rect x="{x:.2f}" y="{top:.2f}" width="{bar_w:.2f}" height="{height:.2f}" rx="{r:.2f}" ry="{r:.2f}" '
                             f'fill="{pal["background"]}" stroke="{color}" stroke-width="2.2" stroke-dasharray="7 5" />')
                label, ly, lcolor, weight = mcfg["ref_label"], top - 6, color_text, "700"
            else:
                r = min(corner_r, height / 2.0) if corner_r else 0
                parts.append(f'<rect x="{x:.2f}" y="{top:.2f}" width="{bar_w:.2f}" height="{height:.2f}" rx="{r:.2f}" ry="{r:.2f}" fill="{color}" />')
                # Out-of-range bars are cut at the axis limit and labelled with their true value
                s = ov["trunc_arrow_size"]
                tcx = x + bar_w / 2
                if trunc_low:
                    tcy = scales.sy(y_min) + 7
                    parts.append(f'<polygon points="{tcx - s:.2f},{tcy:.2f} {tcx + s:.2f},{tcy:.2f} {tcx:.2f},{tcy + s * 1.15:.2f}" fill="{pal["manufacturer_text"]}" opacity="0.8" />')
                    label, ly, lcolor, weight = _format_truncated(v).replace("-", "−"), y_zero + font + 4, pal["text_secondary"], "600"
                elif trunc_high:
                    tcy = scales.sy(y_max) - 7
                    parts.append(f'<polygon points="{tcx - s:.2f},{tcy:.2f} {tcx + s:.2f},{tcy:.2f} {tcx:.2f},{tcy - s * 1.15:.2f}" fill="{pal["manufacturer_text"]}" opacity="0.8" />')
                    label, ly, lcolor, weight = "+" + _format_truncated(v), scales.sy(y_max) - s * 1.6 - 6, pal["text_secondary"], "600"
                elif v >= 0:
                    label, ly, lcolor, weight = fmt_value(mcfg, v), top - 6, color_text, ("700" if method == "ml" else "600")
                else:
                    label, ly, lcolor, weight = fmt_value(mcfg, v), bottom + font + 2, color_text, "600"
            parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{ly:.2f}" text-anchor="middle" fill="{lcolor}" '
                         f'font-size="{font}" font-weight="{weight}">{label}</text>')
    # Sensor and tier labels under the axis
    label_y = scales.y_bottom + typ["tier_label_px"] + 18
    for cx, (sensor, tier) in zip(sub_centers, SUB_LABELS):
        parts.append(f'<text x="{cx:.2f}" y="{label_y:.2f}" text-anchor="middle" fill="{pal["text_primary"]}" font-size="{typ["tier_label_px"]}">{sensor}</text>')
        parts.append(f'<text x="{cx:.2f}" y="{label_y + typ["sensor_label_px"] * 1.5:.2f}" text-anchor="middle" fill="{pal["text_secondary"]}" '
                     f'font-size="{typ["sensor_label_px"]}" font-style="italic">{tier}</text>')
    # Zero line, y axis, ticks and axis label
    parts.append(f'<line x1="{scales.x_left:.2f}" y1="{y_zero:.2f}" x2="{scales.x_right:.2f}" y2="{y_zero:.2f}" stroke="{pal["zero_line"]}" stroke-width="{ov["zero_linewidth"]}" />')
    parts.append(f'<line x1="{scales.x_left:.2f}" y1="{scales.y_top:.2f}" x2="{scales.x_left:.2f}" y2="{scales.y_bottom:.2f}" stroke="{pal["axis_line"]}" stroke-width="{ov["axis_linewidth"]}" />')
    for yv in mcfg["ticks"]:
        gy = scales.sy(yv)
        parts.append(f'<line x1="{scales.x_left - 5:.2f}" y1="{gy:.2f}" x2="{scales.x_left:.2f}" y2="{gy:.2f}" stroke="{pal["axis_line"]}" stroke-width="{ov["axis_linewidth"]}" />')
        parts.append(f'<text x="{scales.x_left - 12:.2f}" y="{gy + typ["tick_px"] * 0.35:.2f}" text-anchor="end" fill="{pal["text_secondary"]}" font-size="{typ["tick_px"]}">{yv:g}</text>')
    lx, lyy = scales.x_left - mcfg.get("axis_label_offset_px", 70), (scales.y_top + scales.y_bottom) / 2
    parts.append(f'<text x="{lx:.2f}" y="{lyy:.2f}" text-anchor="middle" fill="{pal["text_primary"]}" font-size="{typ["axis_label_px"]}" '
                 f'font-style="italic" transform="rotate(-90 {lx:.2f} {lyy:.2f})">{mcfg["axis_label"]}</text>')


# -------------------------------------------------------------------------
# SVG
# -------------------------------------------------------------------------
def build_svg(cfg, data):
    vp, pal, typ, ov, plt_cfg = cfg["viewport"], cfg["palette"], cfg["typography"], cfg["overlay"], cfg["plot"]
    width, height = vp["width_px"], vp["height_px"]
    m = plt_cfg["margin"]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="NO2 three-tier comparison">',
             f'<rect width="{width}" height="{height}" fill="{pal["background"]}" />']
    # Header: node title, subtitle and legend chips
    parts.append(f'<text x="{width / 2:.2f}" y="{cfg["layout"]["node_title_y_px"]}" text-anchor="middle" fill="{pal["panel_subtitle"]}" '
                 f'font-size="{typ["panel_title_px"]}" font-weight="700" font-style="italic">{cfg["node_title"]}</text>')
    parts.append(f'<text x="{width / 2:.2f}" y="{cfg["layout"]["node_title_y_px"] + typ["panel_subtitle_px"] * 1.4:.2f}" text-anchor="middle" '
                 f'fill="{pal["text_secondary"]}" font-size="{typ["panel_subtitle_px"]}" font-style="italic">{cfg["node_subtitle"]}</text>')
    chip_w, chip_h, gap = ov["chip_swatch_w"], ov["chip_swatch_h"], ov["chip_row_gap"]
    char_w = typ["chip_label_px"] * 0.45
    labels = cfg["chip_labels"]
    segs = [(mth, labels[mth], chip_w + 10 + len(labels[mth]) * char_w) for mth in CHIP_ORDER]
    total = sum(s[2] for s in segs) + gap * (len(segs) - 1)
    cx = width / 2 - total / 2
    y = cfg["layout"]["chip_y_px"]
    for mth, text, seg_w in segs:
        sw_y = y - chip_h * 0.75
        if mth == "reference":
            parts.append(f'<rect x="{cx:.2f}" y="{sw_y:.2f}" width="{chip_w}" height="{chip_h}" rx="3" ry="3" fill="{pal["background"]}" stroke="{pal["reference_bar"]}" stroke-width="2" stroke-dasharray="5 3" />')
        else:
            parts.append(f'<rect x="{cx:.2f}" y="{sw_y:.2f}" width="{chip_w}" height="{chip_h}" rx="3" ry="3" fill="{pal[f"{mth}_bar"]}" />')
        parts.append(f'<text x="{cx + chip_w + 10:.2f}" y="{y:.2f}" text-anchor="start" fill="{pal[f"{mth}_text"]}" font-size="{typ["chip_label_px"]}" font-style="italic" font-weight="600">{text}</text>')
        cx += seg_w + gap
    # One panel per metric
    n_panels = len(cfg["metrics"])
    panel_gap = plt_cfg["panel_gap_px"]
    plot_w = width - m["left"] - m["right"]
    pw = (plot_w - panel_gap * (n_panels - 1)) / n_panels
    for i, mcfg in enumerate(cfg["metrics"]):
        x_left = m["left"] + i * (pw + panel_gap)
        scales = PanelScales(x_left=x_left, x_right=x_left + pw, y_top=m["top"], y_bottom=height - m["bottom"],
                             y_min=mcfg["y_min"], y_max=mcfg["y_max"])
        add_metric_panel(parts, scales, data[mcfg["key"]], cfg, mcfg)
    parts.append("</svg>")
    return "".join(parts)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def main():
    cfg = load_config(CONFIG_PATH)
    csv = Path(cfg["paths"]["metrics_csv"])
    csv = csv if csv.is_absolute() else REPO_ROOT / csv
    data = load_metrics(csv, tuple(cfg["ml_low"]), tuple(cfg["ml_mid"]))
    for metric in ("r2", "r", "mbe"):
        print(metric, {k: {mm: round(v, 3) for mm, v in d.items()} for k, d in data[metric].items()})
    svg = build_svg(cfg, data)
    html = build_html(cfg, svg)
    html_out = REPO_ROOT / cfg["paths"]["output_html"]
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(html, encoding="utf-8")
    png_out = REPO_ROOT / cfg["paths"]["output_image"]
    png_out.parent.mkdir(parents=True, exist_ok=True)
    render_with_selenium(html_out, png_out, cfg)
    print(f"HTML: {html_out}\nPNG : {png_out}")


if __name__ == "__main__":
    main()
