"""
Figure 1: study sites and three-tier co-location at KEMP

Reads the configuration and the photograph figures/source/KEMP_Node5.JPG,
and downloads the basemap tiles at run time. Draws (a) the map of the two
sites and (b) the photograph with the three cost tiers marked. The PNG
goes to figures/output/.
"""

import json
import sys
from pathlib import Path
from typing import Tuple

import contextily as cx
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, FancyBboxPatch
from PIL import Image
from pyproj import Transformer

# -------------------------------------------------------------------------
# Paths and configuration
# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Default config next to the script; --config <path> overrides it
CONFIG_PATH = (Path(sys.argv[sys.argv.index("--config") + 1]).resolve()
               if "--config" in sys.argv else SCRIPT_DIR / "fig01_sites_overview_config.json")
# Optional TTF files, registered with matplotlib when the folder exists
FONT_DIR = SCRIPT_DIR / "source" / "fonts"


# -------------------------------------------------------------------------
# Fonts, configuration and coordinates
# -------------------------------------------------------------------------
def register_local_fonts():
    if FONT_DIR.exists():
        for ttf in sorted(FONT_DIR.glob("*.ttf")):
            fm.fontManager.addfont(str(ttf))


_LL_TO_MERC = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def apply_typography(fonts):
    plt.rcParams["text.parse_math"] = False
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = fonts
    plt.rcParams["font.weight"] = "regular"


def ll_to_merc(lon: float, lat: float) -> Tuple[float, float]:
    """Convert longitude and latitude (degrees) to Web Mercator (m)."""
    return _LL_TO_MERC.transform(lon, lat)


def _resolve_basemap_provider(path: str):
    """Resolve a dotted contextily provider name such as CartoDB.Positron."""
    provider = cx.providers
    for part in path.split("."):
        provider = getattr(provider, part)
    return provider


# -------------------------------------------------------------------------
# Panel (a): map of the two sites
# -------------------------------------------------------------------------
def draw_map(ax, cfg):
    """Draw the basemap, site markers, scale bar and panel label; return the KEMP position."""
    typo = cfg["typography"]
    palette = cfg["palette"]
    sites = cfg["sites"]
    map_cfg = cfg["map"]
    frame_cfg = cfg.get("map_frame", {})

    lon_min, lon_max = map_cfg["lon_extent"]
    lat_min, lat_max = map_cfg["lat_extent"]
    x_min, y_min = ll_to_merc(lon_min, lat_min)
    x_max, y_max = ll_to_merc(lon_max, lat_max)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    frame_color = palette.get("frame", palette.get("connector"))
    frame_lw = frame_cfg.get("linewidth", 1.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(frame_color)
        spine.set_linewidth(frame_lw)

    # Basemap tiles are downloaded from the provider at run time
    provider = _resolve_basemap_provider(map_cfg["basemap_source"])
    add_basemap_kwargs = dict(
        source=provider,
        crs="EPSG:3857",
        zoom=map_cfg["basemap_zoom"],
    )
    if map_cfg.get("show_attribution", True):
        add_basemap_kwargs["attribution_size"] = 7
    else:
        add_basemap_kwargs["attribution"] = False
    cx.add_basemap(ax, **add_basemap_kwargs)

    marker_color = palette["marker"]
    edge_color = palette["marker_edge"]
    # Site markers and labels
    for site in sites:
        x, y = ll_to_merc(site["lon"], site["lat"])
        ax.plot(
            x, y,
            marker="o",
            markersize=16,
            markerfacecolor=marker_color,
            markeredgecolor=edge_color,
            markeredgewidth=2.2,
            zorder=5,
        )
        ax.annotate(
            site["label"],
            xy=(x, y),
            xytext=tuple(site["label_offset_px"]),
            textcoords="offset points",
            fontsize=typo["marker_label_fontsize"],
            fontstyle="italic",
            color=palette.get("site_label_fg", "white"),
            ha=site.get("ha", "center"),
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.45,rounding_size=0.6",
                facecolor=palette.get("site_label_bg", marker_color),
                edgecolor=palette.get("site_label_edge", "none"),
                linewidth=1.2,
                alpha=0.95,
            ),
            zorder=6,
        )

    # The KEMP marker anchors the connectors to the photo panel
    kemp = next(s for s in sites if s["id"] == "KEMP")
    kemp_x, kemp_y = ll_to_merc(kemp["lon"], kemp["lat"])

    _draw_scalebar(ax, map_cfg["scalebar_km"], palette["scalebar"], typo["scalebar_fontsize"])

    banner_text = map_cfg.get("banner_text")
    if banner_text:
        ax.text(
            0.02, 0.98,
            banner_text,
            transform=ax.transAxes,
            fontsize=typo.get("banner_fontsize", 20),
            fontstyle="italic",
            color=palette.get("banner_fg", palette["frame"]),
            ha="left",
            va="top",
            zorder=10,
        )

    ax.text(
        0.97, 0.03,
        "(a)",
        transform=ax.transAxes,
        fontsize=typo["panel_label_fontsize"],
        fontweight="bold",
        fontstyle="italic",
        color=palette["panel_label_fg"],
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.40,rounding_size=0.4",
            facecolor=palette["panel_label_bg"],
            edgecolor="none",
            alpha=0.92,
        ),
        zorder=10,
    )

    return (kemp_x, kemp_y)


def _draw_scalebar(ax, length_km: float, color: str, fontsize: float):
    """Draw a scale bar of the given ground length in the lower-left corner."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bar_len_m = length_km * 1000.0
    # Mercator scale factor at the map latitude (51.516 deg N)
    scale_factor = np.cosh(np.arcsinh(np.tan(np.radians(51.516))))
    bar_len_merc = bar_len_m * scale_factor
    bar_x0 = x0 + (x1 - x0) * 0.03
    bar_x1 = bar_x0 + bar_len_merc
    bar_y0 = y0 + (y1 - y0) * 0.05
    ax.plot(
        [bar_x0, bar_x1],
        [bar_y0, bar_y0],
        color=color, linewidth=3.2, solid_capstyle="butt", zorder=8,
    )
    ax.text(
        (bar_x0 + bar_x1) / 2.0,
        bar_y0 + (y1 - y0) * 0.014,
        f"{length_km:g} km",
        fontsize=fontsize,
        fontstyle="italic",
        color=color,
        ha="center",
        va="bottom",
        zorder=9,
    )


# -------------------------------------------------------------------------
# Panel (b): site photo with the three cost tiers
# -------------------------------------------------------------------------
def draw_photo(ax, cfg):
    typo = cfg["typography"]
    palette = cfg["palette"]
    frame_cfg = cfg.get("photo_frame", {})
    photo_path = REPO_ROOT / cfg["paths"]["photo_png"]

    img = np.array(Image.open(photo_path))
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    frame_color = palette.get("frame", palette.get("connector"))
    frame_lw = frame_cfg.get("linewidth", 1.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(frame_color)
        spine.set_linewidth(frame_lw)
    ax.set_aspect("equal", adjustable="box")

    ax.text(
        0.03, 0.03,
        "(b)",
        transform=ax.transAxes,
        fontsize=typo["panel_label_fontsize"],
        fontweight="bold",
        fontstyle="italic",
        color=palette["panel_label_fg"],
        ha="left",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.40,rounding_size=0.4",
            facecolor=palette["panel_label_bg"],
            edgecolor="none",
            alpha=0.92,
        ),
        zorder=10,
    )

    if "photo_annotations" in cfg:
        _draw_photo_annotations(ax, cfg)


# -------------------------------------------------------------------------
# Photo annotations (axes-fraction coordinates from the config)
# -------------------------------------------------------------------------
def _title_chip(ax, x, y, text, bg, fontsize, ha="center", va="center"):
    """Draw a tier title as bold white text on a solid chip."""
    return ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold",
        color="#FFFFFF", ha=ha, va=va, zorder=12,
        bbox=dict(boxstyle="round,pad=0.50,rounding_size=0.7",
                  facecolor=bg, edgecolor="none", alpha=0.97),
    )


def _sub_chip(ax, x, y, text, fontsize, ha="center", va="center"):
    """Draw a sub-label as dark text on a white chip."""
    return ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=fontsize, fontweight="regular",
        color="#1F1F1F", ha=ha, va=va, zorder=12,
        bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.6",
                  facecolor="#FFFFFF", edgecolor="none", alpha=0.93),
    )


def _tier_box(ax, rect, color, lw, fill_alpha):
    x, y, w, h = rect
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        transform=ax.transAxes,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        facecolor=to_rgba(color, fill_alpha), edgecolor=color,
        linewidth=lw, zorder=11,
    ))


def _leader(ax, p1, p2, color, lw):
    ax.add_line(Line2D(
        [p1[0], p2[0]], [p1[1], p2[1]],
        transform=ax.transAxes,
        color=color, linewidth=lw, zorder=10.5,
        solid_capstyle="round",
    ))


def _bbox_axes(ax, txt):
    """Return the box of a text chip, padding included, in axes coordinates."""
    renderer = ax.figure.canvas.get_renderer()
    patch = txt.get_bbox_patch()
    bb = (patch.get_window_extent(renderer) if patch is not None
          else txt.get_window_extent(renderer))
    return bb.transformed(ax.transAxes.inverted())


def _draw_photo_annotations(ax, cfg):
    ann = cfg["photo_annotations"]
    fs_title = ann["title_fontsize"]
    fs_sub = ann["sub_fontsize"]
    pad = ann["box_pad"]
    inset = ann["chip_inset"]

    for chip in ann.get("site_chips", []):
        fs = fs_title if chip.get("size") == "title" else fs_sub
        _sub_chip(ax, chip["x"], chip["y"], chip["text"], fs,
                  ha=chip.get("ha", "center"), va=chip.get("va", "center"))

    titles, subs = {}, {}
    for tier in ann["tiers"]:
        _tier_box(ax, tier["box"], tier["color"],
                  ann["box_linewidth"], ann["fill_alpha"])
        t = tier["title"]
        titles[tier["name"]] = _title_chip(
            ax, t["x"], t["y"], t["text"], tier["color"], fs_title,
            ha=t.get("ha", "center"), va=t.get("va", "center"))
        s = tier["sub"]
        subs[tier["name"]] = _sub_chip(
            ax, s["x"], s["y"], s["text"], fs_sub,
            ha=s.get("ha", "center"), va=s.get("va", "center"))

    # Chip extents are only known after a draw
    ax.figure.canvas.draw()

    # Give the target tier the same title-to-sub-label gap as the reference tier
    eq = ann.get("equalize_sub_gap")
    if eq:
        ref_gap = (_bbox_axes(ax, titles[eq["reference"]]).y0
                   - _bbox_axes(ax, subs[eq["reference"]]).y1)
        tgt_title = _bbox_axes(ax, titles[eq["target"]])
        tgt_sub_txt = subs[eq["target"]]
        tgt_sub = _bbox_axes(ax, tgt_sub_txt)
        shift = (tgt_title.y0 - ref_gap) - tgt_sub.y1
        tgt_sub_txt.set_y(tgt_sub_txt.get_position()[1] + shift)

    for tier in ann["tiers"]:
        bb = _bbox_axes(ax, titles[tier["name"]])
        x, y, w, h = tier["box"]
        # Leader from the title chip to the tier box
        if tier["leader_from"] == "bottom_right":
            start = (bb.x1 - inset, bb.y0 + inset)
        else:
            start = ((bb.x0 + bb.x1) / 2, bb.y0 + inset)
        if tier["leader_to"] == "top":
            end = (x + w / 2, y + h + pad)
        else:
            end = (x - pad, y + h / 2)
        _leader(ax, start, end, tier["color"], ann["box_linewidth"])


# -------------------------------------------------------------------------
# Connectors from the KEMP marker to the photo panel
# -------------------------------------------------------------------------
def add_connectors(fig, ax_map, ax_photo, kemp_xy, cfg):
    palette = cfg["palette"]
    conn = cfg["connector"]
    kx, ky = kemp_xy
    shrink_a = conn.get("shrink_map_pts", 0)
    shrink_b = conn.get("shrink_photo_pts", 0)

    for xy_photo in [(0.0, 1.0), (0.0, 0.0)]:
        con = ConnectionPatch(
            xyA=(kx, ky), coordsA=ax_map.transData,
            xyB=xy_photo, coordsB=ax_photo.transAxes,
            color=palette["connector"],
            linewidth=conn["linewidth"],
            linestyle=conn["linestyle"],
            alpha=conn["alpha"],
            shrinkA=shrink_a,
            shrinkB=shrink_b,
            zorder=3,
        )
        fig.add_artist(con)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def main():
    cfg = load_config(CONFIG_PATH)
    register_local_fonts()
    apply_typography(cfg["typography"]["fonts"])

    fig = plt.figure(figsize=tuple(cfg["layout"]["figsize_inches"]), dpi=200)
    gs = fig.add_gridspec(
        1, 2,
        width_ratios=cfg["layout"]["width_ratios"],
        wspace=cfg["layout"]["wspace"],
    )
    ax_map = fig.add_subplot(gs[0, 0])
    ax_photo = fig.add_subplot(gs[0, 1])

    kemp_xy = draw_map(ax_map, cfg)
    draw_photo(ax_photo, cfg)
    add_connectors(fig, ax_map, ax_photo, kemp_xy, cfg)

    out = REPO_ROOT / cfg["paths"]["output_image"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
