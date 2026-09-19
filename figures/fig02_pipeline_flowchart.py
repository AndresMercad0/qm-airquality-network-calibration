"""
Figure 2: data processing pipeline

Configuration only. Draws the flowchart of the eleven stages, s0 to s10,
in five rows, with the three calibration stages highlighted and the phase
labels at the right. The PNG goes to figures/output/.
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
               if "--config" in sys.argv else SCRIPT_DIR / "fig02_pipeline_flowchart_config.json")


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------------------------------------------------------
# Layout helpers
# -------------------------------------------------------------------------
def stage_position(cfg, stage):
    """Return the centre (x, y) of a stage box from its grid column and row."""
    return (cfg["layout"]["col_x"][stage["col"]],
            cfg["layout"]["row_y"][stage["row"] - 1])


def row_y(cfg, row_idx):
    return cfg["layout"]["row_y"][row_idx - 1]


def render_label(line: str, font_px: float) -> str:
    """Replace the $R_0$ token of a label with SVG subscript markup."""
    if "$R_0$" in line:
        sub_size = f"{font_px * 0.72:.1f}"
        return line.replace(
            "$R_0$",
            f'R<tspan baseline-shift="sub" font-size="{sub_size}">0</tspan>',
        )
    return line


# -------------------------------------------------------------------------
# SVG
# -------------------------------------------------------------------------
def build_svg(cfg):
    vp = cfg["viewport"]
    W, H = vp["width_px"], vp["height_px"]
    L = cfg["layout"]
    pal = cfg["palette"]
    typ = cfg["typography"]
    calib = set(cfg["calib_stages"])

    positions = {st["id"]: stage_position(cfg, st) for st in cfg["stages"]}

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="Data processing pipeline flowchart">'
    )

    # Arrowhead marker
    head_w = typ.get("arrowhead_w", 9)
    head_h = typ.get("arrowhead_h", 7)
    parts.append(
        '<defs>'
        '<marker id="arrowhead" viewBox="0 0 10 8" refX="9" refY="4" '
        f'markerWidth="{head_w}" markerHeight="{head_h}" orient="auto" markerUnits="userSpaceOnUse">'
        f'<path d="M 0 0 L 10 4 L 0 8 z" fill="{pal["arrow"]}" />'
        '</marker>'
        '</defs>'
    )

    parts.append(
        f'<rect width="{W}" height="{H}" fill="{pal["background"]}" />'
    )

    # Separators between pipeline phases
    for (r_top, r_bot) in L["separators_between_rows"]:
        y = (row_y(cfg, r_top) + row_y(cfg, r_bot)) / 2
        x1 = L["margin"]["left"]
        x2 = W - L["margin"]["right"]
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" '
            f'stroke="{pal["separator"]}" stroke-width="{typ["separator_linewidth"]}" />'
        )

    # Phase labels on the right-hand side
    px = L["phase_label_x"]
    phase_fs = typ["phase_label_px"]
    for phase in cfg["phase_labels"]:
        rows = phase["rows"]
        if len(rows) == 1:
            py = row_y(cfg, rows[0])
        else:
            py = sum(row_y(cfg, r) for r in rows) / len(rows)
        lines = phase["text"].split("\n")
        n = len(lines)
        for i, line in enumerate(lines):
            y_line = py + (i - (n - 1) / 2) * phase_fs * 1.25
            parts.append(
                f'<text x="{px:.1f}" y="{y_line:.1f}" text-anchor="start" '
                f'fill="{pal["phase_label"]}" font-size="{phase_fs}" '
                f'font-style="italic" dominant-baseline="middle">{line}</text>'
            )

    # Arrows between stages
    aw = typ["arrow_linewidth"]
    bw2 = L["box_w"] / 2
    bh2 = L["box_h"] / 2
    pad = L["box_arrow_pad"]
    for arr in cfg["arrows"]:
        x1, y1 = positions[arr["from"]]
        x2, y2 = positions[arr["to"]]
        kind = arr["kind"]
        if kind == "right":
            sx, sy = x1 + bw2 + pad, y1
            ex, ey = x2 - bw2 - pad, y2
            d = f"M {sx:.1f} {sy:.1f} L {ex:.1f} {ey:.1f}"
        elif kind == "left":
            sx, sy = x1 - bw2 - pad, y1
            ex, ey = x2 + bw2 + pad, y2
            d = f"M {sx:.1f} {sy:.1f} L {ex:.1f} {ey:.1f}"
        elif kind == "down":
            sx, sy = x1, y1 + bh2 + pad
            ex, ey = x2, y2 - bh2 - pad
            d = f"M {sx:.1f} {sy:.1f} L {ex:.1f} {ey:.1f}"
        elif kind == "diag_dl":
            sx, sy = x1 - bw2 - pad, y1
            ex, ey = x2, y2 - bh2 - pad
            # Cubic curve from the left edge of a box down to the top of the next row
            dx_seg = sx - ex
            dy_seg = ey - sy
            c1x, c1y = sx - dx_seg * 0.5, sy
            c2x, c2y = ex, ey - dy_seg * 0.5
            d = (f"M {sx:.1f} {sy:.1f} "
                 f"C {c1x:.1f} {c1y:.1f} {c2x:.1f} {c2y:.1f} "
                 f"{ex:.1f} {ey:.1f}")
        else:
            raise ValueError(f"Unknown arrow kind: {kind}")
        parts.append(
            f'<path d="{d}" fill="none" stroke="{pal["arrow"]}" '
            f'stroke-width="{aw}" stroke-linecap="round" '
            f'stroke-linejoin="round" marker-end="url(#arrowhead)" />'
        )

    # Stage boxes; calibration stages are highlighted
    bw, bh, rx = L["box_w"], L["box_h"], L["box_rx"]
    for st in cfg["stages"]:
        sid = st["id"]
        label = st["label"]
        x, y = positions[sid]
        is_calib = sid in calib
        bg = pal["box_bg_calib"] if is_calib else pal["box_bg_regular"]
        border = pal["box_border_calib"] if is_calib else pal["box_border_regular"]
        border_w = (typ["box_border_calib_w"] if is_calib
                    else typ["box_border_regular_w"])

        parts.append(
            f'<rect x="{x - bw/2:.1f}" y="{y - bh/2:.1f}" '
            f'width="{bw}" height="{bh}" rx="{rx}" ry="{rx}" '
            f'fill="{bg}" stroke="{border}" stroke-width="{border_w}" />'
        )

        # Stage identifier in the top-left corner
        id_color = pal["stage_id_calib"] if is_calib else pal["stage_id_regular"]
        id_fs = typ["stage_id_px"]
        parts.append(
            f'<text x="{x - bw/2 + 14:.1f}" '
            f'y="{y - bh/2 + id_fs * 0.95 + 4:.1f}" '
            f'text-anchor="start" fill="{id_color}" '
            f'font-size="{id_fs}" font-style="italic" '
            f'font-weight="700">{sid}</text>'
        )

        # Label lines, squeezed with textLength when wider than the box
        label_lines = label.split("\n")
        txt_fs = (typ["stage_text_px"] if len(label) < 24
                  else typ["stage_text_compact_px"])
        txt_weight = "600" if is_calib else "400"
        n_lines = len(label_lines)
        lineh = txt_fs * 1.22
        block_h = (n_lines - 1) * lineh
        label_center_y = y + 8
        first_y = label_center_y - block_h / 2 + txt_fs * 0.34
        pad_x = typ.get("text_padding_x", 10)
        char_ratio = typ.get("char_width_ratio", 0.48)
        available_w = bw - 2 * pad_x
        for i, raw_line in enumerate(label_lines):
            line_y = first_y + i * lineh
            tspan_markup = render_label(raw_line, txt_fs)
            plain_len = len(raw_line.replace("$R_0$", "R0"))
            natural_w = plain_len * txt_fs * char_ratio
            if natural_w > available_w:
                length_attr = (f' textLength="{available_w:.1f}" '
                               'lengthAdjust="spacingAndGlyphs"')
            else:
                length_attr = ""
            parts.append(
                f'<text x="{x:.1f}" y="{line_y:.1f}" text-anchor="middle" '
                f'fill="{pal["stage_text"]}" font-size="{txt_fs}" '
                f'font-weight="{txt_weight}" '
                f'dominant-baseline="alphabetic"{length_attr}>{tspan_markup}</text>'
            )

    # Legend
    lg = cfg["legend"]
    lx, ly = lg["x"], lg["y"]
    sw, sh = lg["swatch_w"], lg["swatch_h"]
    parts.append(
        f'<rect x="{lx:.1f}" y="{ly - sh/2:.1f}" '
        f'width="{sw}" height="{sh}" rx="6" ry="6" '
        f'fill="{pal["box_bg_calib"]}" stroke="{pal["box_border_calib"]}" '
        f'stroke-width="{typ["box_border_calib_w"]}" />'
    )
    parts.append(
        f'<text x="{lx + sw + 12:.1f}" y="{ly:.1f}" text-anchor="start" '
        f'fill="{pal["phase_label"]}" font-size="{typ["legend_px"]}" '
        f'font-style="italic" dominant-baseline="middle">{lg["text"]}</text>'
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
<title>Pipeline flowchart</title>
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
    letter-spacing: 0;
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
    render_with_selenium(html_out, png_out, cfg, wait_s=2.2)
    print(f"PNG : {png_out}")


if __name__ == "__main__":
    main()
