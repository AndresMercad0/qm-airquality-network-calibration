"""
Figure 12: test R2 per pollutant and cost tier at Node 3

Configuration only: the R2 values and the labels are in the JSON. Draws
one panel per pollutant with a point per cost tier, hollow where the
value is fixed by definition, and the cost of each tier under the panels.
The PNG goes to figures/output/.
"""

import json
import sys
from pathlib import Path

from render import render_with_selenium

# -------------------------------------------------------------------------
# Paths and configuration
# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Default config next to the script; --config <path> overrides it
CONFIG_PATH = (Path(sys.argv[sys.argv.index("--config") + 1]).resolve()
               if "--config" in sys.argv else SCRIPT_DIR / "fig12_cost_performance_config.json")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------------------
# Geometry
# -------------------------------------------------------------------------
class PanelScales:
    """Hold the pixel frame of one panel; the tiers sit at fixed fractions of its inner width."""

    TIER_KEYS = ("low", "mid", "high")
    TIER_X_NORM = (0.18, 0.50, 0.82)

    def __init__(self, x_left, x_right, y_top, y_bottom, y_min, y_max,
                 inner_pad):
        self.x_left = x_left
        self.x_right = x_right
        self.y_top = y_top
        self.y_bottom = y_bottom
        self.y_min = y_min
        self.y_max = y_max
        self.inner_pad = inner_pad
        self.inner_left = x_left + inner_pad
        self.inner_right = x_right - inner_pad
        self.inner_w = self.inner_right - self.inner_left
        self.height = y_bottom - y_top

    def sx(self, tier_key):
        return self.inner_left + self.TIER_X_NORM[
            self.TIER_KEYS.index(tier_key)] * self.inner_w

    def sy(self, value):
        f = (value - self.y_min) / (self.y_max - self.y_min)
        return self.y_bottom - f * self.height

    def y_zero(self):
        return self.sy(0.0)


def _bezier(start_x, start_y, end_x, end_y, curve):
    """Return a quadratic path with its control point offset sideways by curve times the chord."""
    mx = (start_x + end_x) / 2
    my = (start_y + end_y) / 2
    dx = end_x - start_x
    dy = end_y - start_y
    nx, ny = -dy, dx
    norm = (nx ** 2 + ny ** 2) ** 0.5 or 1.0
    nx /= norm
    ny /= norm
    cx = mx + nx * curve * norm
    cy = my + ny * curve * norm
    return f"M {start_x:.2f} {start_y:.2f} Q {cx:.2f} {cy:.2f} {end_x:.2f} {end_y:.2f}"


# -------------------------------------------------------------------------
# HTML inside SVG
# -------------------------------------------------------------------------
def _fo(x, y, w, h, html_inner):
    """
    Wrap an HTML fragment in an SVG foreignObject.
    The XHTML namespace on the inner div makes the browser parse it as HTML.
    """
    return (
        f'<foreignObject x="{x:.2f}" y="{y:.2f}" '
        f'width="{w:.2f}" height="{h:.2f}">'
        f'<div xmlns="http://www.w3.org/1999/xhtml">{html_inner}</div>'
        f'</foreignObject>'
    )


# -------------------------------------------------------------------------
# Panel elements
# -------------------------------------------------------------------------
def _draw_panel_graphics(panel_cfg, scales, cfg):
    """Return the SVG fragments of one panel and the position of each tier point."""
    pal = cfg["palette"]
    ov = cfg["overlay"]
    plt_cfg = cfg["plot"]

    parts = []

    # Tier background bands
    band_keys = ("low", "mid", "high")
    band_colors = (pal["low_band"], pal["mid_band"], pal["high_band"])
    half_w = scales.inner_w / 6
    band_top = scales.y_top + 4
    band_bottom = scales.y_bottom
    for tier, color in zip(band_keys, band_colors):
        cx = scales.sx(tier)
        x0 = max(cx - half_w, scales.x_left)
        x1 = min(cx + half_w, scales.x_right)
        parts.append(
            f'<rect x="{x0:.2f}" y="{band_top:.2f}" '
            f'width="{(x1 - x0):.2f}" height="{(band_bottom - band_top):.2f}" '
            f'fill="{color}" opacity="{ov["tier_band_alpha"]}" />'
        )

    for yv in plt_cfg["y_tick_values"]:
        gy = scales.sy(yv)
        parts.append(
            f'<line x1="{scales.x_left:.2f}" y1="{gy:.2f}" '
            f'x2="{scales.x_right:.2f}" y2="{gy:.2f}" '
            f'stroke="{pal["grid_line"]}" '
            f'stroke-width="{ov["grid_linewidth"]}" stroke-dasharray="1 4" />'
        )

    # Optional reference line
    if plt_cfg.get("reference_line_y") is not None:
        ref_y = scales.sy(plt_cfg["reference_line_y"])
        parts.append(
            f'<line x1="{scales.x_left:.2f}" y1="{ref_y:.2f}" '
            f'x2="{scales.x_right:.2f}" y2="{ref_y:.2f}" '
            f'stroke="{pal["reference_line"]}" '
            f'stroke-width="{ov["ref_linewidth"]}" '
            f'stroke-dasharray="5 5" opacity="0.95" />'
        )

    # Zero line
    yz = scales.y_zero()
    parts.append(
        f'<line x1="{scales.x_left:.2f}" y1="{yz:.2f}" '
        f'x2="{scales.x_right:.2f}" y2="{yz:.2f}" '
        f'stroke="{pal["zero_line"]}" '
        f'stroke-width="{ov["zero_linewidth"]}" opacity="0.45" />'
    )

    # Line through the available points
    available = [p for p in panel_cfg["points"] if p["available"]]
    if len(available) >= 2:
        d_parts = []
        for i, p in enumerate(available):
            x = scales.sx(p["tier"])
            y = scales.sy(p["r2"])
            d_parts.append(("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}")
        parts.append(
            f'<path d="{" ".join(d_parts)}" fill="none" '
            f'stroke="{pal["dominant"]}" '
            f'stroke-width="{ov["series_linewidth"]}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="0.85" />'
        )

    # Points: filled if measured, hollow and dashed if fixed by definition, small grey if absent
    point_geoms = {}
    for p in panel_cfg["points"]:
        x = scales.sx(p["tier"])
        if p["available"]:
            y = scales.sy(p["r2"])
            if p.get("definitional"):
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" '
                    f'r="{ov["point_radius"]}" fill="{pal["dominant_fill"]}" '
                    f'stroke="{pal["reference_line"]}" '
                    f'stroke-width="{ov["point_stroke_w"] * 1.4}" stroke-dasharray="3 2" />'
                )
            else:
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" '
                    f'r="{ov["point_radius"]}" fill="{pal["dominant"]}" '
                    f'stroke="{pal["dominant_fill"]}" '
                    f'stroke-width="{ov["point_stroke_w"]}" />'
                )
            point_geoms[p["tier"]] = (x, y)
        else:
            y = scales.y_zero()
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" '
                f'r="{ov["na_radius"]}" fill="{pal["na_marker"]}" '
                f'stroke="{pal["dominant_fill"]}" '
                f'stroke-width="{ov["point_stroke_w"]}" />'
            )
            point_geoms[p["tier"]] = (x, y)

    return parts, point_geoms


def _value_labels_html(panel_cfg, scales, cfg):
    """Return the value and method labels of every point as HTML blocks."""
    ov = cfg["overlay"]
    fragments = []
    label_w = 150
    label_h = 54
    pad = 6
    for p in panel_cfg["points"]:
        x = scales.sx(p["tier"])
        if p["available"]:
            y = scales.sy(p["r2"])
            inner_class = "value-label"
            pos = p.get("label_pos")
            if p.get("definitional"):
                # Points fixed by definition carry a one-line label above the marker
                single_h, single_w = 34, 260
                fo_y = y - ov["point_radius"] - 8 - single_h
                inner = (f'<div class="value-label single"><span class="value-num">1 by def.</span>'
                         f'<span class="value-method">{p["label"]}</span></div>')
                fragments.append(_fo(x - single_w / 2, fo_y, single_w, single_h, inner))
                continue
            # Labels go below high points (R2 >= 0.70), above the others; label_pos forces a side
            if pos == "below" or (pos is None and p["r2"] >= 0.70):
                fo_y = y + ov["point_radius"] + pad
                inner_class += " below"
            else:
                fo_y = y - ov["point_radius"] - pad - label_h
                inner_class += " above"
            num = ("1 by def." if p.get("definitional")
                   else f'{(0.0 if abs(p["r2"]) < 0.005 else p["r2"]):.2f}')
            inner = (
                f'<div class="{inner_class}">'
                f'<span class="value-num">{num}</span>'
                f'<span class="value-method">{p["label"]}</span>'
                f'</div>'
            )
            fragments.append(_fo(x - label_w / 2, fo_y, label_w, label_h, inner))
        else:
            y = scales.y_zero()
            # The n/a text sits above its marker, clear of the tier names
            fo_h_na = 30
            fo_y = y - ov["na_radius"] - 4 - fo_h_na
            inner = f'<div class="na-label">{p.get("label", "n/a")}</div>'
            fragments.append(_fo(x - label_w / 2, fo_y, label_w, fo_h_na, inner))
    return fragments


def _tier_axis_html(panel_cfg, scales, cfg):
    """Return the tier names under the x axis."""
    tiers_cfg = cfg["tiers"]
    plt_cfg = cfg["plot"]

    fo_x = scales.x_left
    fo_y = scales.y_bottom + plt_cfg.get("tier_axis_y_offset_px", 10)
    fo_w = scales.x_right - scales.x_left
    fo_h = plt_cfg.get("tier_axis_h_px", 44)
    inner_parts = ['<div class="tier-axis">']
    for key in PanelScales.TIER_KEYS:
        info = tiers_cfg[key]
        inner_parts.append(
            f'<div class="tier-cell">'
            f'<div class="tier-name">{info["label"]}</div>'
            f'</div>'
        )
    inner_parts.append('</div>')
    return _fo(fo_x, fo_y, fo_w, fo_h, "".join(inner_parts))


def _cost_legend_html(plot_left, plot_right, plot_bottom, cfg):
    """Return the shared row with the cost of each tier, at the foot of the figure."""
    tiers_cfg = cfg["tiers"]
    plt_cfg = cfg["plot"]
    tier_axis_h = plt_cfg.get("tier_axis_h_px", 44)
    tier_axis_offset = plt_cfg.get("tier_axis_y_offset_px", 10)
    legend_offset = plt_cfg.get("cost_legend_y_offset_px", 22)
    fo_y = plot_bottom + tier_axis_offset + tier_axis_h + legend_offset
    fo_x = plot_left
    fo_w = plot_right - plot_left
    fo_h = plt_cfg.get("cost_legend_h_px", 50)

    items = []
    for key in PanelScales.TIER_KEYS:
        info = tiers_cfg[key]
        items.append(
            f'<div class="legend-item">'
            f'<span class="legend-tier">{info["label"]}</span>'
            f'<span class="legend-sep">·</span>'
            f'<span class="legend-cost">{info["cost_text"]}</span>'
            f'<span class="legend-unit">{info["cost_unit"]}</span>'
            f'</div>'
        )
    inner = (
        f'<div class="cost-legend">{"".join(items)}</div>'
    )
    return _fo(fo_x, fo_y, fo_w, fo_h, inner)


def _panel_title_html(panel_cfg, scales, cfg):
    """Return the panel title, placed just above the plot area."""
    plt_cfg = cfg["plot"]
    title_band_top = plt_cfg.get("panel_title_band_top_px", 200)
    fo_h = plt_cfg["margin"]["top"] - title_band_top
    fo_x = scales.x_left
    fo_y = title_band_top
    fo_w = scales.x_right - scales.x_left
    inner = (
        f'<div class="panel-title-block">'
        f'<div class="panel-title">{panel_cfg["title"]}</div>'
        f'</div>'
    )
    return _fo(fo_x, fo_y, fo_w, fo_h, inner)


def _y_axis(scales, cfg):
    """Return the spine and ticks as SVG and the rotated axis label as HTML."""
    pal = cfg["palette"]
    typ = cfg["typography"]
    ov = cfg["overlay"]
    plt_cfg = cfg["plot"]
    parts = []
    parts.append(
        f'<line x1="{scales.x_left:.2f}" y1="{scales.y_top:.2f}" '
        f'x2="{scales.x_left:.2f}" y2="{scales.y_bottom:.2f}" '
        f'stroke="{pal["axis_line"]}" '
        f'stroke-width="{ov["axis_linewidth"]}" />'
    )
    tick_labels_html = []
    for yv in plt_cfg["y_tick_values"]:
        gy = scales.sy(yv)
        parts.append(
            f'<line x1="{(scales.x_left - 6):.2f}" y1="{gy:.2f}" '
            f'x2="{scales.x_left:.2f}" y2="{gy:.2f}" '
            f'stroke="{pal["axis_line"]}" '
            f'stroke-width="{ov["axis_linewidth"]}" />'
        )
        tick_w = 56
        tick_h = typ["tick_px"] * 1.6
        parts.append(_fo(
            scales.x_left - tick_w - 10,
            gy - tick_h / 2,
            tick_w, tick_h,
            f'<div class="tick-label">{yv:g}</div>',
        ))
    label_w = scales.height
    label_h = 40
    cx = scales.x_left - 76
    cy = (scales.y_top + scales.y_bottom) / 2
    parts.append(
        f'<g transform="rotate(-90 {cx:.2f} {cy:.2f})">'
        f'{_fo(cx - label_w / 2, cy - label_h / 2, label_w, label_h, "<div class=\"y-axis-label\">R<sup>2</sup>&#160;&#160;(test set)</div>")}'
        f'</g>'
    )
    return parts


# -------------------------------------------------------------------------
# SVG
# -------------------------------------------------------------------------
def build_svg(cfg):
    vp = cfg["viewport"]
    width = vp["width_px"]
    height = vp["height_px"]
    pal = cfg["palette"]
    plt_cfg = cfg["plot"]
    ov = cfg["overlay"]
    tb = cfg["top_band"]

    margin = plt_cfg["margin"]
    plot_top = margin["top"]
    plot_bottom = height - margin["bottom"]
    plot_left = margin["left"]
    plot_right = width - margin["right"]
    plot_w = plot_right - plot_left
    n_panels = len(cfg["panels"])
    gap = plt_cfg["panel_gap_px"]
    panel_w = (plot_w - gap * (n_panels - 1)) / n_panels

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xhtml="http://www.w3.org/1999/xhtml" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Cost-performance trade-off">',
        f'<rect width="{width}" height="{height}" fill="{pal["background"]}" />',
    ]

    # Figure title
    title_block_w = width - 200
    parts.append(_fo(
        100, tb["title_y_px"], title_block_w, 50,
        f'<div class="header">'
        f'<h1>{tb["title"]}</h1>'
        f'</div>',
    ))

    # Panel graphics
    inner_pad = plt_cfg["panel_inner_pad_px"]
    panel_meta = []
    for i, panel_cfg in enumerate(cfg["panels"]):
        x_left = plot_left + i * (panel_w + gap)
        x_right = x_left + panel_w
        scales = PanelScales(
            x_left=x_left, x_right=x_right,
            y_top=plot_top, y_bottom=plot_bottom,
            y_min=plt_cfg["y_min"], y_max=plt_cfg["y_max"],
            inner_pad=inner_pad,
        )
        graphics, point_geoms = _draw_panel_graphics(panel_cfg, scales, cfg)
        parts.extend(graphics)
        panel_meta.append({
            "cfg": panel_cfg,
            "scales": scales,
            "point_geoms": point_geoms,
        })

    # Y axis on the first panel only
    parts.extend(_y_axis(panel_meta[0]["scales"], cfg))

    # Value labels, tier names and panel titles
    for meta in panel_meta:
        parts.extend(_value_labels_html(meta["cfg"], meta["scales"], cfg))
        parts.append(_tier_axis_html(meta["cfg"], meta["scales"], cfg))
        parts.append(_panel_title_html(meta["cfg"], meta["scales"], cfg))

    # Shared cost legend
    parts.append(_cost_legend_html(plot_left, plot_right, plot_bottom, cfg))

    parts.append("</svg>")
    return "".join(parts)


# -------------------------------------------------------------------------
# HTML and CSS
# -------------------------------------------------------------------------
def build_html(cfg, svg_markup):
    typ = cfg["typography"]
    pal = cfg["palette"]
    vp = cfg["viewport"]
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cost-performance trade-off</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400;1,600&family=Crimson+Pro:ital,wght@0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:        {pal["background"]};
    --primary:   {pal["text_primary"]};
    --secondary: {pal["text_secondary"]};
    --muted:     {pal["text_muted"]};
    --dominant:  {pal["dominant"]};
    --accent:    {pal["annotation_text"]};
    --na:        {pal["na_text"]};
    --ref:       {pal["reference_text"]};
  }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); }}
  body {{
    font-family: {typ["font_stack"]};
    color: var(--primary);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
  }}
  .figure-wrap {{ width: {vp["width_px"]}px; background: var(--bg); padding: 0; }}
  svg text {{ font-family: inherit; letter-spacing: 0.01em; }}
  foreignObject {{ overflow: visible; }}

  /* Header */
  .header {{ text-align: center; }}
  .header h1 {{
    margin: 0;
    font-size: {typ["chip_title_px"]}px;
    color: var(--dominant);
    font-style: italic;
    font-weight: 700;
    line-height: 1.1;
  }}

  /* Panel titles right above the plot */
  .panel-title-block {{
    text-align: center;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    padding-bottom: 4px;
  }}
  .panel-title {{
    font-size: {typ["panel_title_px"]}px;
    color: {pal["panel_subtitle"]};
    font-style: italic;
    font-weight: 700;
    line-height: 1;
  }}

  /* Value labels above (or below) points. The above/below modifiers
     anchor the text to the bottom or top edge so the foreignObject can
     be placed exactly relative to the point. */
  .value-label {{
    height: 100%;
    text-align: center;
    line-height: 1.05;
    display: flex;
    flex-direction: column;
    align-items: center;
  }}
  .value-label.above {{ justify-content: flex-end; }}
  .value-label.below {{ justify-content: flex-start; }}
  /* Points fixed by definition carry a one-line label above the marker */
  .value-label.single {{ flex-direction: row; align-items: baseline; justify-content: center; gap: 8px; white-space: nowrap; }}
  .value-num {{
    font-size: {typ["value_label_px"]}px;
    color: var(--dominant);
    font-weight: 700;
  }}
  .value-method {{
    margin-top: 2px;
    font-size: {typ["method_label_px"]}px;
    color: var(--secondary);
    font-style: italic;
  }}
  .na-label {{
    height: 100%;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    text-align: center;
    font-size: {typ["method_label_px"]}px;
    color: var(--na);
    font-style: italic;
  }}

  /* Tier axis under each panel */
  .tier-axis {{
    display: flex;
    width: 100%;
    height: 100%;
    align-items: flex-start;
  }}
  .tier-cell {{
    flex: 1 1 0;
    text-align: center;
    line-height: 1.15;
  }}
  .tier-name {{
    font-size: {typ["tier_label_px"]}px;
    color: var(--primary);
    font-weight: 600;
  }}

  /* Shared cost legend at the foot of the figure (one row, three items). */
  .cost-legend {{
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: baseline;
    gap: 36px;
    width: 100%;
    height: 100%;
    color: var(--secondary);
  }}
  .legend-item {{
    display: inline-flex;
    align-items: baseline;
    gap: 8px;
  }}
  .legend-tier {{
    font-size: {typ["legend_tier_px"]}px;
    color: var(--primary);
    font-weight: 600;
  }}
  .legend-sep {{
    color: var(--muted);
    font-size: {typ["legend_tier_px"]}px;
  }}
  .legend-cost {{
    font-size: {typ["legend_cost_px"]}px;
    font-style: italic;
    color: var(--secondary);
  }}
  .legend-unit {{
    font-size: {typ["legend_unit_px"]}px;
    font-style: italic;
    color: var(--muted);
  }}

  /* Y axis labels */
  .tick-label {{
    text-align: right;
    font-size: {typ["tick_px"]}px;
    color: var(--secondary);
    line-height: 1;
  }}
  .y-axis-label {{
    text-align: center;
    font-size: {typ["axis_label_px"]}px;
    color: var(--primary);
    font-style: italic;
    line-height: 1;
  }}
  .y-axis-label sup {{
    font-size: 0.7em;
    vertical-align: super;
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
    print("Building the cost-performance figure...")
    for panel in cfg["panels"]:
        bits = []
        for pt in panel["points"]:
            if pt["available"]:
                bits.append(f"{pt['tier']}={pt['r2']:.3f} ({pt['label']})")
            else:
                bits.append(f"{pt['tier']}=n/a")
        print(f"  {panel['title']:8s}  " + "  |  ".join(bits))

    svg = build_svg(cfg)
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
