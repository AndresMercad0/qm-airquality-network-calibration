"""
Figure 4: training, validation and test periods at both nodes

Configuration only: the dates are in the JSON. Draws one lane per node on
a common axis from April 2025 to February 2026, with the training band,
the validation block, the test period and the interruptions of the node
record. The PNG goes to figures/output/.
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

from render import render_with_selenium

# -------------------------------------------------------------------------
# Paths and configuration
# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Default config next to the script; --config <path> overrides it
CONFIG_PATH = (Path(sys.argv[sys.argv.index("--config") + 1]).resolve()
               if "--config" in sys.argv else SCRIPT_DIR / "fig04_temporal_design_config.json")


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def d(s):
    """Parse an ISO date."""
    return date.fromisoformat(s)


def esc(t):
    """Escape text for SVG."""
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# -------------------------------------------------------------------------
# SVG
# -------------------------------------------------------------------------
def build_svg(cfg):
    vp, pal, typ, plt_cfg = cfg["viewport"], cfg["palette"], cfg["typography"], cfg["plot"]
    W, H = vp["width_px"], vp["height_px"]
    m = plt_cfg["margin"]
    x_left, x_right = m["left"], W - m["right"]
    t0, t1 = d(cfg["axis"]["start"]), d(cfg["axis"]["end"])

    # Linear time scale over the axis window
    def sx(day):
        return x_left + (day - t0).days / (t1 - t0).days * (x_right - x_left)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
             f'aria-label="Temporal design of training, selection and evaluation">',
             f'<rect width="{W}" height="{H}" fill="{pal["background"]}"/>']
    # Title
    parts.append(f'<text x="{W/2:.1f}" y="46" text-anchor="middle" fill="{pal["title"]}" font-size="{typ["title_px"]}" '
                 f'font-weight="700" font-style="italic">{esc(cfg["title"])}</text>')
    # Legend entry of the record interruptions
    gap_item = next(i for i in cfg["legend"] if i["key"] == "gap")
    lx, ly = x_left, 96
    parts.append(f'<rect x="{lx}" y="{ly-16}" width="40" height="16" fill="{pal["gap"]}" stroke="{pal["gap_stroke"]}" stroke-width="1.6" stroke-dasharray="5 3"/>')
    parts.append(f'<text x="{lx+50}" y="{ly}" fill="{pal["text_secondary"]}" font-size="{typ["legend_px"]}" font-style="italic">{esc(gap_item["label"])}</text>')
    # Month grid and axis labels
    y_axis = H - m["bottom"] + 8
    months = []
    y0, mo = 2025, 4
    for _ in range(cfg["axis"]["n_months"] + 1):
        months.append(date(y0, mo, 1))
        mo += 1
        if mo == 13:
            mo, y0 = 1, y0 + 1
    top_y = m["top"]
    for k, mdate in enumerate(months):
        gx = sx(mdate)
        parts.append(f'<line x1="{gx:.1f}" y1="{top_y-10}" x2="{gx:.1f}" y2="{y_axis-18}" stroke="{pal["grid_line"]}" stroke-width="1" stroke-dasharray="1 4"/>')
        if k < len(months) - 1:
            label = mdate.strftime("%b") + (f" {mdate.year}" if mdate.month in (4, 1) else "")
            cx = (sx(mdate) + sx(months[k + 1])) / 2
            parts.append(f'<text x="{cx:.1f}" y="{y_axis}" text-anchor="middle" fill="{pal["text_secondary"]}" font-size="{typ["tick_px"]}">{label}</text>')
    parts.append(f'<line x1="{x_left}" y1="{y_axis-18}" x2="{x_right}" y2="{y_axis-18}" stroke="{pal["axis_line"]}" stroke-width="1.8"/>')

    # One lane per node
    band_h, rec_h = plt_cfg["band_h_px"], plt_cfg["record_h_px"]
    lane_h = 30 + band_h + plt_cfg["record_offset_px"] + rec_h + 34
    y = top_y
    for node in cfg["nodes"]:
        rs, re_, te_, ts_ = d(node["record_start"]), d(node["record_end"]), d(node["train_end"]), d(node["test_start"])
        vmin, vmax = d(node["val_start_min"]), d(node["val_start_max"])
        yb = y + 30                                         # Band top
        yr = yb + band_h + plt_cfg["record_offset_px"]      # Record bar top
        parts.append(f'<text x="{x_left-18}" y="{yb+band_h/2+2}" text-anchor="end" fill="{pal["text_primary"]}" font-size="{typ["node_px"]}" font-weight="700" font-style="italic">{esc(node["title"])}</text>')
        parts.append(f'<text x="{x_left-18}" y="{yb+band_h/2+26}" text-anchor="end" fill="{pal["text_secondary"]}" font-size="{typ["node_sub_px"]}" font-style="italic">{esc(node["subtitle"])}</text>')
        # Training band, validation band (range of start dates, then the common block) and test band
        parts.append(f'<rect x="{sx(rs):.1f}" y="{yb}" width="{sx(te_+timedelta(days=1))-sx(rs):.1f}" height="{band_h}" fill="{pal["train"]}"/>')
        parts.append(f'<rect x="{sx(vmin):.1f}" y="{yb}" width="{sx(te_+timedelta(days=1))-sx(vmin):.1f}" height="{band_h}" fill="{pal["val"]}" opacity="0.55"/>')
        parts.append(f'<rect x="{sx(vmax):.1f}" y="{yb}" width="{sx(te_+timedelta(days=1))-sx(vmax):.1f}" height="{band_h}" fill="{pal["val"]}"/>')
        parts.append(f'<rect x="{sx(ts_):.1f}" y="{yb}" width="{sx(re_+timedelta(days=1))-sx(ts_):.1f}" height="{band_h}" fill="{pal["test"]}"/>')
        parts.append(f'<text x="{(sx(rs)+sx(vmin))/2:.1f}" y="{yb+band_h/2+7}" text-anchor="middle" fill="{pal["band_text"]}" font-size="{typ["band_px"]}" font-weight="600">Training data</text>')
        parts.append(f'<text x="{(sx(vmax)+sx(te_))/2:.1f}" y="{yb+band_h/2+7}" text-anchor="middle" fill="{pal["band_text"]}" font-size="{typ["band_px"]}" font-weight="600">Validation</text>')
        parts.append(f'<text x="{(sx(ts_)+sx(re_))/2:.1f}" y="{yb+band_h/2+7}" text-anchor="middle" fill="{pal["test_text"]}" font-size="{typ["band_px"]}" font-weight="600">Test period</text>')
        # Bracket over the range of validation start dates
        bx0, bx1, by = sx(vmin), sx(vmax), yb - 10
        parts.append(f'<path d="M{bx0:.1f},{by-8} L{bx0:.1f},{by} L{bx1:.1f},{by} L{bx1:.1f},{by-8}" fill="none" stroke="{pal["bracket"]}" stroke-width="1.6"/>')
        parts.append(f'<text x="{(bx0+bx1)/2:.1f}" y="{by-14}" text-anchor="middle" fill="{pal["bracket"]}" font-size="{typ["small_px"]}" font-style="italic">{esc(node["val_start_label"])}</text>')
        parts.append(f'<text x="{sx(te_+timedelta(days=1))-4:.1f}" y="{yb-8}" text-anchor="end" fill="{pal["text_secondary"]}" font-size="{typ["small_px"]}">{esc(node["train_end_label"])}</text>')
        parts.append(f'<text x="{sx(ts_)+4:.1f}" y="{yb+band_h+22}" text-anchor="start" fill="{pal["text_secondary"]}" font-size="{typ["small_px"]}">{esc(node["test_start_label"])}</text>')
        # Node record with its interruptions
        yr2 = yr + 10
        parts.append(f'<rect x="{sx(rs):.1f}" y="{yr2}" width="{sx(re_+timedelta(days=1))-sx(rs):.1f}" height="{rec_h}" rx="2" fill="{pal["record"]}"/>')
        for g0, g1, lab in node["gaps"]:
            gx0, gx1 = sx(d(g0)), sx(d(g1) + timedelta(days=1))
            parts.append(f'<rect x="{gx0:.1f}" y="{yr2-1}" width="{gx1-gx0:.1f}" height="{rec_h+2}" fill="{pal["gap"]}" stroke="{pal["gap_stroke"]}" stroke-width="1.2" stroke-dasharray="4 3"/>')
            if lab:
                parts.append(f'<text x="{(gx0+gx1)/2:.1f}" y="{yr2+rec_h+24}" text-anchor="middle" fill="{pal["text_muted"]}" font-size="{typ["small_px"]}" font-style="italic">{lab}</text>')
        parts.append(f'<text x="{x_left-18}" y="{yr2+rec_h/2+5}" text-anchor="end" fill="{pal["text_muted"]}" font-size="{typ["small_px"]}" font-style="italic">node record</text>')
        y += lane_h + plt_cfg["lane_gap_px"]
    parts.append("</svg>")
    return "".join(parts)


# -------------------------------------------------------------------------
# HTML wrapper
# -------------------------------------------------------------------------
def build_html(cfg, svg):
    typ, pal, vp = cfg["typography"], cfg["palette"], cfg["viewport"]
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Temporal design</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400;1,600;1,700&display=swap" rel="stylesheet">
<style>html,body{{margin:0;padding:0;background:{pal["background"]}}} body{{font-family:{typ["font_stack"]};-webkit-font-smoothing:antialiased}}
.figure-wrap{{width:{vp["width_px"]}px}} svg text{{font-family:{typ["font_stack"]};letter-spacing:0.01em}}</style></head>
<body><div class="figure-wrap" id="figure-root">{svg}</div></body></html>"""


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def main():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    svg = build_svg(cfg)
    html_out = REPO_ROOT / cfg["paths"]["output_html"]
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(build_html(cfg, svg), encoding="utf-8")
    png_out = REPO_ROOT / cfg["paths"]["output_image"]
    png_out.parent.mkdir(parents=True, exist_ok=True)
    render_with_selenium(html_out, png_out, cfg)
    print(f"PNG : {png_out}")


if __name__ == "__main__":
    main()
