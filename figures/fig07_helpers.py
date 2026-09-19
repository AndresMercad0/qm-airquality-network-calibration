"""
Helpers of Figure 7

Configuration loader, panel scale, compact number format and the HTML
page that wraps the SVG. Imported by fig07_no2_three_tier.py; nothing is
drawn or written here.
"""

import json
from pathlib import Path


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------------------
# Geometry
# -------------------------------------------------------------------------
class PanelScales:
    """Hold the pixel frame of one panel and map values to y."""

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
# Labels
# -------------------------------------------------------------------------
def _format_truncated(r2):
    """Format an out-of-range value compactly, for example -684K or -1.4M."""
    av = abs(r2)
    sign = "-" if r2 < 0 else ""
    if av >= 1_000_000:
        return f"{sign}{av / 1_000_000:.1f}M"
    if av >= 1_000:
        return f"{sign}{av / 1_000:.0f}K"
    return f"{r2:.0f}"


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
<title>NO2 three-tier comparison</title>
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
