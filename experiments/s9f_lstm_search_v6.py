"""
LSTM hyperparameter search with a masked scorer and gap-free windows

Runs the LSTM search of s9 for the 15 cells of a node (5 targets x 3
configurations). The scorer ignores non-finite predictions and the
timestamps travel in X, so windows are contiguous in every fold and in
the refit. Writes tabla_maestra_lstm_v6.csv, rewritten after each cell.
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    REF_CSV_NAME, load_node_pipeline, versions_banner, masked_neg_rmse,
    make_ts_subclass, epoch_seconds,
)

# -------------------------------------------------------------------------
# Configuration constants
# -------------------------------------------------------------------------
TARGETS = ["PM2.5", "CO", "NO2", "O3", "NO2_alpha"]
PILOT_MIN_PER_CELL = 15.4  # Minutes per cell assumed until the first cell finishes

# -------------------------------------------------------------------------
# Progress file (lstm_v6_status.json)
# -------------------------------------------------------------------------
import json

PROGRESS = {"node": None, "started_utc": None, "updated_utc": None,
            "cells_total": 0, "cells_done": 0, "cells_skipped": 0,
            "current_cell": None, "fits_done_in_cell": 0, "fits_total_in_cell": 0,
            "percent": 0.0, "elapsed_min": 0.0, "eta_min": None, "finished": False}
_STATUS_PATH = None
_T0 = None
_CELL_MINUTES: list[float] = []


def write_status(**kw):
    """
    Update the progress file.
    The percentage counts the finished cells plus the fraction of fits done in the
    current cell; the remaining time is the pending cells times the mean cell time.
    """
    PROGRESS.update(kw)
    now = time.time()
    PROGRESS["elapsed_min"] = round((now - _T0) / 60, 1)
    PROGRESS["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    done, total = PROGRESS["cells_done"], PROGRESS["cells_total"]
    frac = (PROGRESS["fits_done_in_cell"] / PROGRESS["fits_total_in_cell"]
            if PROGRESS["fits_total_in_cell"] else 0.0)
    searchable = max(total - PROGRESS["cells_skipped"], 1)
    PROGRESS["percent"] = round(100.0 * min(done + frac, searchable) / searchable, 1)
    per_cell = (sum(_CELL_MINUTES) / len(_CELL_MINUTES)) if _CELL_MINUTES else PILOT_MIN_PER_CELL
    remaining = (searchable - done - frac) * per_cell
    PROGRESS["eta_min"] = round(max(remaining, 0.0), 1)
    PROGRESS["eta_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + remaining * 60))
    if _STATUS_PATH is not None:
        tmp = _STATUS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(PROGRESS, indent=2), encoding="utf-8")
        tmp.replace(_STATUS_PATH)


def make_counting_subclass(LSTMRegressorTS):
    """Count every fit of a search (iterations x folds + refit) for the progress file."""
    class LSTMRegressorTSCounted(LSTMRegressorTS):
        def fit(self, X, y, timestamps=None):
            out = super().fit(X, y, timestamps=timestamps)
            write_status(fits_done_in_cell=PROGRESS["fits_done_in_cell"] + 1)
            return out
    LSTMRegressorTSCounted.__name__ = "LSTMRegressorTSCounted"
    return LSTMRegressorTSCounted


# -------------------------------------------------------------------------
# Search cells and columns of the output table
# -------------------------------------------------------------------------
CONFIGS = ["A", "B", "C"]
SUMMARY_COLS = [
    "gas", "config", "model", "n_train", "n_test", "n_features",
    "R2_train", "R2_test", "RMSE", "MAE", "MBE", "nRMSE",
    "pearson_r", "CEN_level", "overfit_ratio", "best_params", "model_path",
    "n_scored_train", "n_scored_test", "search_minutes", "status", "protocol",
    "original_R2_test", "original_best_params",
]


def original_lstm_rows(cfg) -> pd.DataFrame:
    """Return the LSTM rows of the s9 master tables, indexed by (gas, config)."""
    frames = []
    for name in ("tabla_maestra_resultados.csv", "tabla_maestra_no2alpha.csv",
                 "tabla_maestra_no2alpha_lstm.csv"):
        f = cfg.RESULTS_S9 / name
        if f.exists():
            frames.append(pd.read_csv(f))
    d = pd.concat(frames, ignore_index=True)
    return d[d.model == "LSTM"].set_index(["gas", "config"])


# -------------------------------------------------------------------------
# Search per cell
# -------------------------------------------------------------------------
def main():
    """
    Run the search for every pending cell of the node and save the refitted model.
    Cells whose original LSTM row has no test metric are recorded as not_scorable and skipped.
    """
    ap = argparse.ArgumentParser(description="LSTM hyperparameter search with a masked RMSE scorer and gap-free windows.")
    ap.add_argument("--node", type=int, choices=(3, 5), required=True)
    ap.add_argument("--gases", nargs="*", default=None, help="subset of targets")
    ap.add_argument("--configs", nargs="*", default=None, help="subset of configs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    global _T0, _STATUS_PATH
    t0 = _T0 = time.time()
    print("=" * 70)
    print(f"  LSTM hyperparameter search | Node {args.node}")
    print("=" * 70)
    versions_banner()

    cfg, utils = load_node_pipeline(args.node)
    from ml.data_loader import prepare_dataset, get_train_test_masks, get_feature_columns
    from ml.ols_refit import refit_ols_on_train
    from ml.models import LSTMRegressor, create_model
    from ml.evaluator import evaluate_model, compute_overfit_ratio
    from ml.trainer import _max_combinations
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    import torch

    np.random.seed(cfg.ML_RANDOM_STATE)
    torch.manual_seed(cfg.ML_RANDOM_STATE)

    s1_csv = utils.detect_csv(cfg.DATA_S1, f"{cfg.NODE_ID}_QC_RAW_wide_data_*.csv")
    s3_csv = utils.detect_csv(cfg.DATA_S3, f"{cfg.NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
    s4_csv = utils.detect_csv(cfg.DATA_S4, f"{cfg.NODE_ID}_PREPROCESS_wide_data_*.csv")
    ref_csv = cfg.REF_DATA_DIR / REF_CSV_NAME[args.node]
    for name, p in (("s1", s1_csv), ("s3", s3_csv), ("s4", s4_csv), ("REF", ref_csv)):
        if p is None or not Path(p).exists():
            sys.exit(f"ERROR: {name} CSV not found: {p}")
        print(f"  input {name:3s} {Path(p).name}")

    df = prepare_dataset(s1_csv, s3_csv, s4_csv, ref_csv)
    train_mask, test_mask = get_train_test_masks(df)
    ols_info = refit_ols_on_train(df, train_mask)
    orig = original_lstm_rows(cfg)

    out_csv = cfg.RESULTS_S9 / "tabla_maestra_lstm_v6.csv"
    _STATUS_PATH = None if args.dry_run else cfg.RESULTS_S9 / "lstm_v6_status.json"
    cv_dir = cfg.RESULTS_S9 / "lstm_v6_cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    # Resume from the cells already written
    rows = []
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        rows = prev.to_dict("records")
        print(f"  resuming: {len(rows)} cells already in {out_csv.name}")
    done = {(r["gas"], r["config"]) for r in rows}

    gases = args.gases or TARGETS
    configs = args.configs or CONFIGS
    LSTMRegressorTS = make_counting_subclass(make_ts_subclass(LSTMRegressor))
    cells = [(g, c) for g in gases for c in configs if (g, c) not in done]
    print(f"  pending cells: {len(cells)}")
    # Cells without a finite test R2 in the original run are not searched
    n_skip_expected = sum(1 for g, c in cells if (g, c) in orig.index
                          and not np.isfinite(float(orig.loc[(g, c)]["R2_test"])))
    write_status(node=args.node, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
                 cells_total=len(cells), cells_skipped=n_skip_expected)

    for k, (gas, config) in enumerate(cells, 1):
        t_cell = time.time()
        print("\n" + "-" * 70)
        print(f"  [{k}/{len(cells)}] {gas} / Config {config}   ({(time.time() - t0) / 60:.0f} min elapsed)")
        print("-" * 70)
        ref_col = cfg.ML_REF_COLS[gas]
        feature_cols = get_feature_columns(gas, config, df)
        if config in ("B", "C") and gas not in ols_info:
            print("  train-only OLS not available: skipped")
            continue
        valid = df[feature_cols + [ref_col]].notna().all(axis=1)
        train_valid, test_valid = valid & train_mask, valid & test_mask
        n_train, n_test = int(train_valid.sum()), int(test_valid.sum())
        o = orig.loc[(gas, config)] if (gas, config) in orig.index else None
        o_r2 = float(o["R2_test"]) if o is not None else np.nan
        o_bp = o["best_params"] if o is not None else None
        # The rows must reproduce those of the original s9 run
        if o is not None and not (int(o["n_train"]) == n_train and int(o["n_test"]) == n_test):
            sys.exit(f"ERROR: rows not reproduced for {gas}/{config}: "
                     f"{n_train}/{n_test} vs {o['n_train']}/{o['n_test']}")
        print(f"  rows train={n_train} test={n_test} features={len(feature_cols)} | "
              f"original R2_test={o_r2:.4f}")
        base = {"gas": gas, "config": config, "model": "LSTM", "n_train": n_train,
                "n_test": n_test, "n_features": len(feature_cols),
                "protocol": "v6: masked RMSE scorer + timestamps carried in X",
                "original_R2_test": o_r2, "original_best_params": o_bp}
        if o is not None and not np.isfinite(o_r2):
            print("  no metric in the original run (no contiguous 24 h window): skipped")
            rows.append({**base, "status": "not_scorable", "search_minutes": 0.0})
            if not args.dry_run:
                pd.DataFrame(rows).reindex(columns=SUMMARY_COLS).to_csv(out_csv, index=False)
            continue
        if args.dry_run:
            rows.append({**base, "status": "dry_run"})
            continue
        # Fits of the cell: iterations x folds + refit
        n_iter_cell = min(create_model("LSTM", 1)[2], _max_combinations(create_model("LSTM", 1)[1]))
        write_status(current_cell=f"{gas}/{config}", fits_done_in_cell=0,
                     fits_total_in_cell=n_iter_cell * cfg.ML_CV_SPLITS + 1)
        print(f"  progress: {PROGRESS['percent']}% | ETA {PROGRESS['eta_min']} min "
              f"({PROGRESS['eta_utc']})")

        X_train = df.loc[train_valid, feature_cols].to_numpy(dtype=np.float64)
        y_train = df.loc[train_valid, ref_col].to_numpy(dtype=np.float64)
        X_test = df.loc[test_valid, feature_cols].to_numpy(dtype=np.float64)
        y_test = df.loc[test_valid, ref_col].to_numpy(dtype=np.float64)
        # Timestamps (epoch seconds) travel as the last column of X
        X_train_in = np.column_stack([X_train, epoch_seconds(df.loc[train_valid, cfg.COL_TS].values)])
        X_test_in = np.column_stack([X_test, epoch_seconds(df.loc[test_valid, cfg.COL_TS].values)])

        estimator, param_dist, n_iter = create_model("LSTM", len(feature_cols))
        estimator = LSTMRegressorTS(**estimator.get_params())
        n_iter = min(n_iter, _max_combinations(param_dist))
        # Randomised search on a time-series split, scored with the masked RMSE
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search = RandomizedSearchCV(
                estimator, param_dist, n_iter=n_iter,
                cv=TimeSeriesSplit(n_splits=cfg.ML_CV_SPLITS),
                scoring=masked_neg_rmse, random_state=cfg.ML_RANDOM_STATE,
                n_jobs=1, verbose=0, refit=True, error_score=np.nan,
            )
            search.fit(X_train_in, y_train)
        cv = pd.DataFrame(search.cv_results_)
        cv.to_csv(cv_dir / f"{gas.replace('.', '_')}_{config}_cv_results.csv", index=False)
        n_finite = int(cv["mean_test_score"].notna().sum())

        # Metrics of the refitted model on the rows with a finite prediction
        best = search.best_estimator_
        y_pred_train = best.predict(X_train_in)
        y_pred_test = best.predict(X_test_in)
        vt, vv = np.isfinite(y_pred_train), np.isfinite(y_pred_test)
        m_train = evaluate_model(y_train[vt], y_pred_train[vt])
        m_test = evaluate_model(y_test[vv], y_pred_test[vv])
        overfit = compute_overfit_ratio(m_train["R2"], m_test["R2"])
        model_path = cfg.RESULTS_S9_MODELS / f"{gas}_{config}_lstm_v6.pt".lower()
        best.save_torch(model_path)
        mins = (time.time() - t_cell) / 60
        print(f"  search {mins:.1f} min | finite {n_finite}/{len(cv)} | best {search.best_params_}")
        print(f"  R2_train={m_train['R2']:.4f} R2_test={m_test['R2']:.4f} (original {o_r2:.4f}) "
              f"RMSE={m_test['RMSE']:.3f} scored {int(vv.sum())}/{n_test}")
        rows.append({**base,
                     "R2_train": m_train["R2"], "R2_test": m_test["R2"], "RMSE": m_test["RMSE"],
                     "MAE": m_test["MAE"], "MBE": m_test["MBE"], "nRMSE": m_test["nRMSE"],
                     "pearson_r": m_test["pearson_r"], "CEN_level": m_test["CEN_level"],
                     "overfit_ratio": overfit, "best_params": str(search.best_params_),
                     "model_path": model_path.name, "n_scored_train": int(vt.sum()),
                     "n_scored_test": int(vv.sum()), "search_minutes": round(mins, 1),
                     "status": f"ok ({n_finite}/{len(cv)} finite)"})
        pd.DataFrame(rows).reindex(columns=SUMMARY_COLS).to_csv(out_csv, index=False)
        _CELL_MINUTES.append(mins)
        write_status(cells_done=PROGRESS["cells_done"] + 1, current_cell=None,
                     fits_done_in_cell=0, fits_total_in_cell=0)
        print(f"  progress: {PROGRESS['percent']}% | ETA {PROGRESS['eta_min']} min "
              f"({PROGRESS['eta_utc']})")

    if not args.dry_run:
        write_status(finished=True, current_cell=None)
    if args.dry_run:
        print(f"\n  dry-run OK: {len(rows)} cells checked ({(time.time() - t0) / 60:.1f} min)")
        return
    print(f"\n  done: {len(rows)} rows in {out_csv} ({(time.time() - t0) / 60:.0f} min)")


if __name__ == "__main__":
    main()
