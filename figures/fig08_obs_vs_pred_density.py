"""
Figure 8: observed against predicted concentrations at Node 3

Rebuilds the Node 3 dataset with the pipeline modules (outputs of steps
s1, s3 and s4, and the reference file), loads the saved model of each
panel and predicts the test rows; R2 is read from tabla_maestra_v6.csv.
Draws one density panel per target. The PNG goes to figures/output/.
"""

import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from render import render_with_selenium

# -------------------------------------------------------------------------
# Paths and configuration
# -------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
# Default config next to the script; --config <path> overrides it
CONFIG_PATH = (Path(sys.argv[sys.argv.index("--config") + 1]).resolve()
               if "--config" in sys.argv else SCRIPT_DIR / "fig08_obs_vs_pred_density_config.json")

# The pipeline is imported for Node 3; AQMS_NODE must be set before its first import
PIPELINE = REPO_ROOT / "pipeline"
os.environ["AQMS_NODE"] = "node_3"
sys.path.insert(0, str(PIPELINE))


# -------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------
def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_predictions_for_panel(df, train_mask, test_mask, gas, model_name,
                              config, models_dir, master_table):
    """Return the observed and predicted test values of one model and its tabulated test R2."""
    from ml.data_loader import get_feature_columns
    from config import COL_TS, ML_REF_COLS

    ref_col = ML_REF_COLS[gas]
    feature_cols = get_feature_columns(gas, config, df)
    n_features = len(feature_cols)
    required = feature_cols + [ref_col]
    valid = df[required].notna().all(axis=1)
    test_valid = valid & test_mask

    X_test = df.loc[test_valid, feature_cols].values
    y_test = df.loc[test_valid, ref_col].values
    ts_test = df.loc[test_valid, COL_TS].reset_index(drop=True)

    # Saved models are named <gas>_<config>_<model>, in lower case
    pt_path = models_dir / f"{gas.lower()}_{config.lower()}_{model_name.lower()}.pt"
    row = master_table[
        (master_table["gas"] == gas)
        & (master_table["config"] == config)
        & (master_table["model"] == model_name)
    ].iloc[0]
    best_params = ast.literal_eval(row["best_params"])

    # Tree models are stored with joblib, neural models as PyTorch weights plus hyperparameters
    if model_name in ("RF", "XGBoost", "LightGBM"):
        import joblib
        model = joblib.load(models_dir / f"{gas.lower()}_{config.lower()}_{model_name.lower()}.joblib")
        y_pred = model.predict(X_test)
    elif model_name == "LSTM":
        from ml.models import LSTMRegressor
        kwargs = {k: v for k, v in best_params.items()
                  if k in ("lstm_units", "fc_dim", "dropout", "lr",
                           "batch_size", "epochs", "patience", "window")}
        model = LSTMRegressor(**kwargs)
        model.load_torch(pt_path, n_features=n_features)
        y_pred = model.predict(X_test, timestamps=ts_test)
    else:
        from ml.models import DNNRegressor
        kwargs = {k: v for k, v in best_params.items()
                  if k in ("hidden_dim_1", "hidden_dim_2", "dropout", "lr",
                           "batch_size", "epochs", "patience")}
        model = DNNRegressor(**kwargs)
        model.load_torch(pt_path, n_features=n_features)
        y_pred = model.predict(X_test)

    valid_pred = np.isfinite(y_pred) & np.isfinite(y_test)
    return y_test[valid_pred], y_pred[valid_pred], float(row["R2_test"])


def load_all_panel_data(cfg):
    """Build the Node 3 dataset, refit the OLS on the training rows and predict every panel."""
    from ml.data_loader import prepare_dataset, get_train_test_masks
    from ml.ols_refit import refit_ols_on_train
    from utils import detect_csv
    from config import (
        NODE_ID, DATA_S1, DATA_S3, DATA_S4,
        REF_DATA_DIR, REF_CSV_NAME,
        RESULTS_S9, RESULTS_S9_MODELS,
    )

    # Pipeline intermediates of Node 3 and its reference file
    s1 = detect_csv(DATA_S1, f"{NODE_ID}_QC_RAW_wide_data_*.csv")
    s3 = detect_csv(DATA_S3, f"{NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
    s4 = detect_csv(DATA_S4, f"{NODE_ID}_PREPROCESS_wide_data_*.csv")
    ref = REF_DATA_DIR / REF_CSV_NAME

    df = prepare_dataset(s1, s3, s4, ref)
    train_mask, test_mask = get_train_test_masks(df)
    # Configs B and C take the OLS estimate as a feature
    refit_ols_on_train(df, train_mask)

    # Master table with the hyperparameters and the test R2 of every model
    if cfg.get("master_table"):
        master = pd.read_csv(RESULTS_S9 / cfg["master_table"])
    else:
        master = pd.read_csv(RESULTS_S9 / "tabla_maestra_resultados.csv")
        master_alpha_path = RESULTS_S9 / "tabla_maestra_no2alpha.csv"
        if master_alpha_path.exists():
            master_alpha = pd.read_csv(master_alpha_path)
            master = pd.concat([master, master_alpha], ignore_index=True)

    out = []
    for spec in cfg["panels"]:
        y_true, y_pred, r2 = get_predictions_for_panel(
            df, train_mask, test_mask,
            gas=spec["gas"], model_name=spec["model"], config=spec["config"],
            models_dir=RESULTS_S9_MODELS, master_table=master,
        )
        out.append({**spec, "y_true": y_true, "y_pred": y_pred, "r2": r2})
        print(f"  {spec['gas']:10s} {spec['model']}-{spec['config']}  "
              f"n={len(y_true):>5d}  R2={r2:+.3f}  {spec['ref_label']}")
    return out


# -------------------------------------------------------------------------
# SVG
# -------------------------------------------------------------------------
def _interp_color(c0, c1, t):
    """Interpolate linearly between two hex colours."""
    def hx(s):
        return int(s, 16)
    r0, g0, b0 = hx(c0[1:3]), hx(c0[3:5]), hx(c0[5:7])
    r1, g1, b1 = hx(c1[1:3]), hx(c1[3:5]), hx(c1[5:7])
    r = round(r0 + (r1 - r0) * t)
    g = round(g0 + (g1 - g0) * t)
    b = round(b0 + (b1 - b0) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


def _density_panel(parts, panel, x_left, x_right, y_top, y_bottom, cfg):
    """Draw one observed-versus-predicted panel as a 2-D histogram with a log colour scale."""
    pal = cfg["palette"]
    typ = cfg["typography"]
    ov = cfg["overlay"]
    plt_cfg = cfg["plot"]

    y_true = panel["y_true"]
    y_pred = panel["y_pred"]
    r2 = panel["r2"]

    x_max = panel["x_max"]
    x_min_frac = plt_cfg["x_padding_frac"]
    x_min = -x_max * x_min_frac
    y_min = x_min

    # Counts on a square grid that starts slightly below zero
    n_bins = plt_cfg["n_bins"]
    counts, x_edges, y_edges = np.histogram2d(
        y_true, y_pred, bins=n_bins,
        range=[[x_min, x_max], [y_min, x_max]],
    )

    plot_w = x_right - x_left
    plot_h = y_bottom - y_top

    def sx(v):
        return x_left + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v):
        return y_bottom - (v - y_min) / (x_max - y_min) * plot_h

    # Title block: pollutant, sensor and model
    title_x = (x_left + x_right) / 2
    sensor_px = typ.get("sensor_px", 15)
    title_y = y_top - 60
    parts.append(
        f'<text x="{title_x:.2f}" y="{title_y:.2f}" '
        f'text-anchor="middle" fill="{pal["text_primary"]}" '
        f'font-size="{typ["title_px"]}" font-weight="600">'
        f'{panel["title_html"]}</text>'
    )
    sensor_y = title_y + sensor_px * 1.9
    if panel.get("sensor_label"):
        parts.append(
            f'<text x="{title_x:.2f}" y="{sensor_y:.2f}" '
            f'text-anchor="middle" fill="{pal["text_muted"]}" '
            f'font-size="{sensor_px}" font-style="italic">'
            f'{panel["sensor_label"]}</text>'
        )
    parts.append(
        f'<text x="{title_x:.2f}" y="{sensor_y + typ["model_px"] * 1.2:.2f}" '
        f'text-anchor="middle" fill="{pal["text_secondary"]}" '
        f'font-size="{typ["model_px"]}" font-style="italic">'
        f'{panel["model_label"]}</text>'
    )

    # Density cells
    cell_w = plot_w / n_bins
    cell_h = plot_h / n_bins
    log_max = np.log(counts.max() + 1) if counts.max() > 0 else 1.0
    for i in range(n_bins):
        for j in range(n_bins):
            c = counts[i, j]
            if c == 0:
                continue
            t = np.log(c + 1) / log_max
            color = _interp_color(pal["density_low"], pal["density_high"], t)
            cx = x_left + i * cell_w
            cy = sy(y_edges[j + 1])
            parts.append(
                f'<rect x="{cx:.2f}" y="{cy:.2f}" '
                f'width="{cell_w:.2f}" height="{cell_h:.2f}" '
                f'fill="{color}" />'
            )

    # 1:1 line
    diag_xy0 = max(x_min, y_min)
    diag_xy1 = min(x_max, x_max)
    parts.append(
        f'<line x1="{sx(diag_xy0):.2f}" y1="{sy(diag_xy0):.2f}" '
        f'x2="{sx(diag_xy1):.2f}" y2="{sy(diag_xy1):.2f}" '
        f'stroke="{pal["diag_line"]}" stroke-width="{ov["diag_linewidth"]}" '
        f'stroke-dasharray="6 4" />'
    )

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

    # Ticks shared by both axes
    def _nice_ticks(vmax):
        if vmax <= 1:
            step = 0.25
        elif vmax <= 10:
            step = 2.5
        elif vmax <= 100:
            step = 25
        elif vmax <= 1000:
            step = 200
        else:
            step = 500
        ticks = [t for t in np.arange(0, vmax + step / 2, step) if t <= vmax]
        return ticks

    for tv in _nice_ticks(x_max):
        gx = sx(tv)
        parts.append(
            f'<line x1="{gx:.2f}" y1="{y_bottom:.2f}" '
            f'x2="{gx:.2f}" y2="{y_bottom + 4:.2f}" '
            f'stroke="{pal["axis_line"]}" stroke-width="{ov["axis_linewidth"]}" />'
        )
        parts.append(
            f'<text x="{gx:.2f}" y="{y_bottom + typ["tick_px"] + 6:.2f}" '
            f'text-anchor="middle" fill="{pal["text_secondary"]}" '
            f'font-size="{typ["tick_px"]}">{tv:g}</text>'
        )
        gy = sy(tv)
        parts.append(
            f'<line x1="{x_left - 4:.2f}" y1="{gy:.2f}" '
            f'x2="{x_left:.2f}" y2="{gy:.2f}" '
            f'stroke="{pal["axis_line"]}" stroke-width="{ov["axis_linewidth"]}" />'
        )
        parts.append(
            f'<text x="{x_left - 8:.2f}" y="{gy + typ["tick_px"] * 0.35:.2f}" '
            f'text-anchor="end" fill="{pal["text_secondary"]}" '
            f'font-size="{typ["tick_px"]}">{tv:g}</text>'
        )

    parts.append(
        f'<text x="{x_left + 8:.2f}" y="{y_top + typ["annot_px"] + 4:.2f}" '
        f'text-anchor="start" fill="{pal["r2_text"]}" '
        f'font-size="{typ["annot_px"]}" font-weight="700">'
        f'R<tspan baseline-shift="super" font-size="0.7em">2</tspan> = '
        f'{(0.0 if abs(r2) < 0.005 else r2):.2f}</text>'
    )
    parts.append(
        f'<text x="{x_left + 8:.2f}" y="{y_top + typ["annot_px"] * 2.0 + 4:.2f}" '
        f'text-anchor="start" fill="{pal["text_muted"]}" '
        f'font-size="{typ["annot_px"]}" font-style="italic">'
        f'n = {len(y_true):,}</text>'
    )

    # Reference label in the lower-right corner
    pad_x = ov["ref_label_pad_x"]
    pad_y = ov["ref_label_pad_y"]
    parts.append(
        f'<text x="{x_right - pad_x:.2f}" y="{y_bottom - pad_y:.2f}" '
        f'text-anchor="end" fill="{pal["ref_text"]}" '
        f'font-size="{typ["ref_px"]}" font-style="italic">'
        f'{panel["ref_label"]}</text>'
    )


def build_svg(cfg, panel_data):
    vp = cfg["viewport"]
    width = vp["width_px"]
    height = vp["height_px"]
    pal = cfg["palette"]
    typ = cfg["typography"]
    plt_cfg = cfg["plot"]

    margin = plt_cfg["margin"]
    panel_gap = plt_cfg["panel_gap_px"]
    plot_top = margin["top"]
    plot_bottom = height - margin["bottom"]
    plot_left = margin["left"]
    plot_right = width - margin["right"]

    n = len(panel_data)
    panel_w = (plot_right - plot_left - panel_gap * (n - 1)) / n

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="Observed vs predicted density (Node 3, with reference-type chips)">'
    )
    parts.append(
        f'<rect width="{width}" height="{height}" fill="{pal["background"]}" />'
    )

    # One panel per pollutant
    for k, panel in enumerate(panel_data):
        x_left = plot_left + k * (panel_w + panel_gap)
        x_right = x_left + panel_w
        _density_panel(parts, panel, x_left, x_right, plot_top, plot_bottom, cfg)

    # Shared axis labels
    x_lbl_x = (plot_left + plot_right) / 2
    x_lbl_y = height - margin["bottom"] / 3
    parts.append(
        f'<text x="{x_lbl_x:.2f}" y="{x_lbl_y:.2f}" '
        f'text-anchor="middle" fill="{pal["text_primary"]}" '
        f'font-size="{typ["axis_label_px"]}" font-style="italic">'
        f'Observed  (µg/m³)</text>'
    )

    y_lbl_x = 30
    y_lbl_y = (plot_top + plot_bottom) / 2
    parts.append(
        f'<text x="{y_lbl_x:.2f}" y="{y_lbl_y:.2f}" '
        f'text-anchor="middle" fill="{pal["text_primary"]}" '
        f'font-size="{typ["axis_label_px"]}" font-style="italic" '
        f'transform="rotate(-90 {y_lbl_x:.2f} {y_lbl_y:.2f})">'
        f'Predicted  (µg/m³)</text>'
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
<title>Observed against predicted density</title>
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

    print("Loading models and predicting on test set ...")
    panel_data = load_all_panel_data(cfg)

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
