"""
Figure 5: cross-node transfer protocol

Configuration only. Draws the two node boxes joined by one arc per
transfer direction, the arc labels and, under the diagram, the definition
of the change in R2. The PNG goes to figures/output/.
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
               if "--config" in sys.argv else SCRIPT_DIR / "fig05_cross_node_protocol_config.json")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------------------
# SVG
# -------------------------------------------------------------------------
def build_svg(cfg) -> str:
    vp = cfg["viewport"]
    pal = cfg["palette"]
    typ = cfg["typography"]
    L = cfg["layout"]
    C = cfg["content"]

    W, H = vp["width_px"], vp["height_px"]

    # Node boxes, left and right
    b1_x, b1_y = L["box1_x"], L["box_y"]
    b2_x, b2_y = L["box2_x"], L["box_y"]
    bw, bh = L["box_w"], L["box_h"]
    br = L["box_radius"]

    b1_cx = b1_x + bw / 2
    b2_cx = b2_x + bw / 2
    b_cy_text_base = b1_y + 52

    # Arc endpoints on the inner edges of the boxes
    top_y = b1_y + 55
    bot_y = b1_y + bh - 55
    top_start_x = b1_x + bw
    top_end_x = b2_x
    bot_start_x = b2_x
    bot_end_x = b1_x + bw

    amp = L["arc_amplitude"]
    top_mid_y = top_y - amp
    bot_mid_y = bot_y + amp
    mid_x = (b1_x + bw + b2_x) / 2

    # Arrowhead marker
    defs = f"""
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="{pal['arrow']}"/>
        </marker>
      </defs>
    """

    def box(x, y, content):
        """Build one node box: label, site, location and reference station."""
        lbl = content["label"]
        site = content["site"]
        loc = content["location"]
        ref = content["reference"]
        cx = x + bw / 2
        y0 = y + 60
        y1 = y0 + 40
        y2 = y1 + 30
        sep_y = y2 + 32
        y3 = sep_y + 40
        return f"""
        <rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="{br}" ry="{br}"
              fill="{pal['box_bg']}" stroke="{pal['box_border']}" stroke-width="{L['border_lw']}"/>
        <text x="{cx}" y="{y0}" text-anchor="middle"
              fill="{pal['text_primary']}" font-size="{typ['node_title_px']}"
              font-weight="700" font-style="italic">{lbl}</text>
        <text x="{cx}" y="{y1}" text-anchor="middle"
              fill="{pal['text_accent']}" font-size="{typ['node_site_px']}"
              font-style="italic" font-weight="600">{site}</text>
        <text x="{cx}" y="{y2}" text-anchor="middle"
              fill="{pal['text_secondary']}" font-size="{typ['node_location_px']}"
              font-style="italic">{loc}</text>
        <line x1="{x + 30}" y1="{sep_y}" x2="{x + bw - 30}" y2="{sep_y}"
              stroke="{pal['separator']}" stroke-width="1"/>
        <text x="{cx}" y="{y3}" text-anchor="middle"
              fill="{pal['text_secondary']}" font-size="{typ['ref_label_px']}"
              font-style="italic">{ref}</text>
        """

    boxes_svg = box(b1_x, b1_y, C["node1"]) + box(b2_x, b2_y, C["node2"])

    # Quadratic arcs, inset from the box edges so the arrowheads stay visible
    inset = 6
    top_arc = (
        f"M {top_start_x + inset} {top_y} "
        f"Q {mid_x} {top_mid_y} {top_end_x - inset} {top_y}"
    )
    bot_arc = (
        f"M {bot_start_x - inset} {bot_y} "
        f"Q {mid_x} {bot_mid_y} {bot_end_x + inset} {bot_y}"
    )

    arrows_svg = f"""
      <path d="{top_arc}" fill="none" stroke="{pal['arrow']}"
            stroke-width="{L['arrow_lw']}" marker-end="url(#arrow)"/>
      <path d="{bot_arc}" fill="none" stroke="{pal['arrow']}"
            stroke-width="{L['arrow_lw']}" marker-end="url(#arrow)"/>
    """

    # Two-line label at the apex of each arc
    line_height = int(typ["arrow_label_px"] * 1.25)
    top_label_y1 = top_mid_y - line_height - 2
    top_label_y2 = top_mid_y - 2
    bot_label_y1 = bot_mid_y + line_height - 10
    bot_label_y2 = bot_label_y1 + line_height
    arc_labels_svg = f"""
      <text x="{mid_x}" y="{top_label_y1}" text-anchor="middle"
            fill="{pal['arrow_label']}" font-size="{typ['arrow_label_px']}"
            font-style="italic">{C['arrow_top_label_line1']}</text>
      <text x="{mid_x}" y="{top_label_y2}" text-anchor="middle"
            fill="{pal['arrow_label']}" font-size="{typ['arrow_label_px']}"
            font-style="italic">{C['arrow_top_label_line2']}</text>
      <text x="{mid_x}" y="{bot_label_y1}" text-anchor="middle"
            fill="{pal['arrow_label']}" font-size="{typ['arrow_label_px']}"
            font-style="italic">{C['arrow_bottom_label_line1']}</text>
      <text x="{mid_x}" y="{bot_label_y2}" text-anchor="middle"
            fill="{pal['arrow_label']}" font-size="{typ['arrow_label_px']}"
            font-style="italic">{C['arrow_bottom_label_line2']}</text>
    """

    # Formula and its note below the diagram
    formula_y = bot_label_y2 + int(typ["formula_px"] * 2.3)
    formula_sub_y = formula_y + int(typ["formula_sub_px"] * 1.4)
    formula_svg = f"""
      <text x="{W/2}" y="{formula_y}" text-anchor="middle"
            fill="{pal['text_accent']}" font-size="{typ['formula_px']}"
            font-style="italic" font-weight="600">{C['formula']}</text>
      <text x="{W/2}" y="{formula_sub_y}" text-anchor="middle"
            fill="{pal['text_secondary']}" font-size="{typ['formula_sub_px']}"
            font-style="italic">{C['formula_sub']}</text>
    """

    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
         width="{W}" height="{H}">
      {defs}
      <rect x="0" y="0" width="{W}" height="{H}" fill="{pal['background']}"/>
      {boxes_svg}
      {arrows_svg}
      {arc_labels_svg}
      {formula_svg}
    </svg>
    """


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
<title>Cross-node transfer protocol</title>
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
  .figure-wrap {{
    width: {vp["width_px"]}px;
    background: var(--bg);
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
    svg = build_svg(cfg)
    html = build_html(cfg, svg)

    html_out = REPO_ROOT / cfg["paths"]["output_html"]
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(html, encoding="utf-8")
    print(f"HTML: {html_out}")

    png_out = REPO_ROOT / cfg["paths"]["output_image"]
    png_out.parent.mkdir(parents=True, exist_ok=True)
    render_with_selenium(html_out, png_out, cfg, wait_s=2.0)
    print(f"PNG : {png_out}")


if __name__ == "__main__":
    main()
