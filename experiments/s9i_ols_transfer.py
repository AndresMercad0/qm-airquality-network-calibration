"""
Transfer of the train-only OLS between nodes with source coefficients

Fits the OLS (signal + T + RH) on the training rows of the source node,
applies its coefficients unchanged to the destination node and scores
them on the test period of the destination against its own reference.
Writes ols_transfer_v6.csv (one row per direction and target).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REF_CSV_NAME, load_node_pipeline

# -------------------------------------------------------------------------
# Configuration constants
# -------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "cross_node"
TARGETS = ["PM2.5", "CO", "NO2", "O3", "NO2_alpha"]


# -------------------------------------------------------------------------
# Frame of one node and scoring
# -------------------------------------------------------------------------
def load_frame(node: int):
    """
    Load the ML frame of one node with its training and test masks.
    The pipeline functions are returned too, because they are imported per node.
    """
    cfg, utils = load_node_pipeline(node)
    from ml.data_loader import prepare_dataset, get_train_test_masks
    from ml.ols_refit import refit_ols_on_train
    from ml.evaluator import evaluate_model
    s1 = utils.detect_csv(cfg.DATA_S1, f"{cfg.NODE_ID}_QC_RAW_wide_data_*.csv")
    s3 = utils.detect_csv(cfg.DATA_S3, f"{cfg.NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
    s4 = utils.detect_csv(cfg.DATA_S4, f"{cfg.NODE_ID}_PREPROCESS_wide_data_*.csv")
    ref = cfg.REF_DATA_DIR / REF_CSV_NAME[node]
    for n, p in (("s1", s1), ("s3", s3), ("s4", s4), ("ref", ref)):
        if p is None or not Path(p).exists():
            sys.exit(f"ERROR: {n} not found: {p}")
    df = prepare_dataset(s1, s3, s4, ref)
    train_mask, test_mask = get_train_test_masks(df)
    return cfg, df, train_mask, test_mask, refit_ols_on_train, evaluate_model


def score(evaluate_model, y, pred):
    """Return R2, RMSE and n on the finite pairs; metrics are NaN below 50 pairs."""
    ok = np.isfinite(y) & np.isfinite(pred)
    if ok.sum() < 50:
        return {"R2": np.nan, "RMSE": np.nan, "n": int(ok.sum())}
    m = evaluate_model(y[ok], pred[ok])
    return {"R2": m["R2"], "RMSE": m["RMSE"], "n": int(ok.sum())}


# -------------------------------------------------------------------------
# Transfer in both directions
# -------------------------------------------------------------------------
def main():
    print("OLS transfer with source coefficients")
    frames = {}
    coefs = {}
    src_test = {}
    for node in (3, 5):
        cfg, df, tr, te, refit, ev = load_frame(node)
        info = refit(df, tr)  # Train-only OLS of the node
        coefs[node] = {g: dict(info[g]["coefs"]) for g in TARGETS if g in info}
        src_test[node] = {}
        for g in TARGETS:
            y = df.loc[te, cfg.ML_REF_COLS[g]].to_numpy(float)
            p = df.loc[te, f"c_hat_ols_{g}"].to_numpy(float)
            src_test[node][g] = score(ev, y, p)
        frames[node] = (cfg, df, tr, te, ev)
        print(f"  Node {node}: OLS train-only fitted; test R2 " +
              ", ".join(f"{g}={src_test[node][g]['R2']:.3f}" for g in TARGETS))

    rows = []
    for src, dst in ((3, 5), (5, 3)):
        cfg, df, tr, te, ev = frames[dst]
        for g in TARGETS:
            c = coefs[src].get(g)
            if not c:
                continue
            # Source coefficients applied to the columns of the destination
            feats = [k for k in c if k != "intercept"]
            missing = [f for f in feats if f not in df.columns]
            if missing:
                sys.exit(f"ERROR: destination {dst} lacks columns {missing} for {g}")
            X = df.loc[te, feats].to_numpy(float)
            pred = c["intercept"] + X @ np.asarray([c[f] for f in feats], float)
            y = df.loc[te, cfg.ML_REF_COLS[g]].to_numpy(float)
            transferred = score(ev, y, pred)
            local = src_test[dst][g]  # OLS of the destination on its own test period
            rows.append({
                "gas": g, "direction": f"{src}->{dst}", "source_node": src, "dest_node": dst,
                "ols_features": "+".join(feats),
                "R2_source": src_test[src][g]["R2"], "n_source_test": src_test[src][g]["n"],
                "R2_dest_transferred": transferred["R2"], "RMSE_dest_transferred": transferred["RMSE"],
                "n_dest_test": transferred["n"],
                "R2_dest_local": local["R2"], "n_dest_local": local["n"],
                "delta_R2": transferred["R2"] - src_test[src][g]["R2"],
                "gap_to_local": transferred["R2"] - local["R2"],
                "period": "test", "basis": "own valid rows of the OLS features and the reference",
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "ols_transfer_v6.csv", index=False)
    pd.set_option("display.width", 200)
    print(out[["gas", "direction", "R2_source", "R2_dest_transferred", "R2_dest_local", "delta_R2", "gap_to_local", "n_dest_test"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
