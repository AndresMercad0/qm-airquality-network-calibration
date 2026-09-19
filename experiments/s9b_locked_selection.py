"""
Model selection on a validation block of the training period

Refits the 15 models of each target with their stored hyperparameters and
scores them on a validation block of the training period, not on the test
period. Writes the selection table locked_selection_per_gas_node{N}.csv
(rules: lowest validation RMSE; simplest family within 2% of it).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    REF_CSV_NAME, load_node_pipeline, versions_banner, make_ts_subclass, epoch_seconds,
)

# -------------------------------------------------------------------------
# Configuration constants
# -------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "cross_node"
TARGETS = ["PM2.5", "CO", "NO2", "O3", "NO2_alpha"]
CONFIGS = ["A", "B", "C"]
FAMILIES = ["RF", "LightGBM", "XGBoost", "DNN", "LSTM"]
# Complexity rank for the parsimony rule (lower is simpler; same order as ml/drift.py)
COMPLEXITY = {"RF": 0, "XGBoost": 1, "LightGBM": 1, "DNN": 2, "LSTM": 3}
VAL_COLS = ["node", "gas", "config", "model", "seed", "n_train_inner", "n_val", "n_val_scored",
            "val_start", "val_end", "val_RMSE", "val_R2", "val_MAE", "val_nRMSE",
            "best_params", "params_source", "ols_variant", "R2_test_persisted",
            "RMSE_test_persisted", "n_test", "model_path", "fit_seconds", "notes"]


# -------------------------------------------------------------------------
# Master tables of s9 (stored hyperparameters and test metrics)
# -------------------------------------------------------------------------
def load_master(cfg) -> pd.DataFrame:
    """
    Collect the s9 master tables, one row per combination.
    LSTM rows are replaced by those of tabla_maestra_lstm_v6.csv when that file exists.
    """
    frames = []
    for name in ("tabla_maestra_resultados.csv", "tabla_maestra_no2alpha.csv",
                 "tabla_maestra_no2alpha_lstm.csv"):
        f = cfg.RESULTS_S9 / name
        if f.exists():
            d = pd.read_csv(f)
            d["params_source"] = name
            frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    v6 = cfg.RESULTS_S9 / "tabla_maestra_lstm_v6.csv"
    if v6.exists():
        d6 = pd.read_csv(v6)
        d6 = d6[d6["status"].astype(str).str.startswith("ok")].copy()
        d6["params_source"] = "tabla_maestra_lstm_v6.csv"
        keys = set(zip(d6.gas, d6.config))
        replaced = (m.model == "LSTM") & pd.Series(list(zip(m.gas, m.config)), index=m.index).isin(keys)
        m = m[~replaced]
        m = pd.concat([m, d6[m.columns.intersection(d6.columns)]], ignore_index=True)
    return m


# -------------------------------------------------------------------------
# Validation scores and selection per target
# -------------------------------------------------------------------------
def main():
    """
    Write locked_selection_validation_node{N}.csv (one row per combination and seed),
    its mean over seeds and locked_selection_per_gas_node{N}.csv (one row per rule).
    """
    ap = argparse.ArgumentParser(description="Select the ML model of each target on a validation block of the training period.")
    ap.add_argument("--node", type=int, choices=(3, 5), required=True)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--val-start", default=None)
    ap.add_argument("--val-end", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--gases", nargs="*", default=None)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--tag", default="", help="suffix of the output files (e.g. _fixedblock)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print("=" * 70)
    print(f"  Model selection on validation | Node {args.node} | val_frac={args.val_frac} "
          f"fixed={args.val_start}..{args.val_end} seeds={args.seeds}")
    print("=" * 70)
    versions_banner()

    cfg, utils = load_node_pipeline(args.node)
    from ml.data_loader import prepare_dataset, get_train_test_masks, get_feature_columns
    from ml.ols_refit import refit_ols_on_train
    from ml.models import LSTMRegressor, create_model
    from ml.trainer import _refit_lgbm_early_stopping
    from ml.evaluator import evaluate_model
    import torch

    s1 = utils.detect_csv(cfg.DATA_S1, f"{cfg.NODE_ID}_QC_RAW_wide_data_*.csv")
    s3 = utils.detect_csv(cfg.DATA_S3, f"{cfg.NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
    s4 = utils.detect_csv(cfg.DATA_S4, f"{cfg.NODE_ID}_PREPROCESS_wide_data_*.csv")
    ref = cfg.REF_DATA_DIR / REF_CSV_NAME[args.node]
    for n, p in (("s1", s1), ("s3", s3), ("s4", s4), ("ref", ref)):
        if p is None or not Path(p).exists():
            sys.exit(f"ERROR: {n} not found: {p}")
        print(f"  input {n:3s} {Path(p).name}")
    df = prepare_dataset(s1, s3, s4, ref)
    train_mask, test_mask = get_train_test_masks(df)
    ts_all = pd.to_datetime(df[cfg.COL_TS], utc=True)
    master = load_master(cfg)
    LSTMRegressorTS = make_ts_subclass(LSTMRegressor)

    gases = args.gases or TARGETS
    families = args.families or FAMILIES
    cells = [(g, c, f) for g in gases for c in CONFIGS for f in families]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_path = OUT_DIR / f"locked_selection_validation_node{args.node}{args.tag}.csv"
    status_path = OUT_DIR / f"e1_status_node{args.node}{args.tag}.json"
    rows: list[dict] = []
    if val_path.exists() and not args.dry_run:
        rows = pd.read_csv(val_path).to_dict("records")
        print(f"  resuming: {len(rows)} rows in {val_path.name}")
    done = {(r["gas"], r["config"], r["model"]) for r in rows}
    cell_minutes: list[float] = []

    # Progress file: cells done, percentage and remaining time from the mean cell time
    def status(**kw):
        d = {"node": args.node, "cells_total": len(cells), "cells_done": len({(r['gas'], r['config'], r['model']) for r in rows}),
             "elapsed_min": round((time.time() - t0) / 60, 1),
             "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}
        per = (sum(cell_minutes) / len(cell_minutes)) if cell_minutes else 1.5
        d["percent"] = round(100 * d["cells_done"] / max(len(cells), 1), 1)
        d["eta_min"] = round(per * (len(cells) - d["cells_done"]), 1)
        if not args.dry_run:
            status_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        return d

    for k, (gas, config, family) in enumerate(cells, 1):
        if (gas, config, family) in done:
            continue
        t_cell = time.time()
        ref_col = cfg.ML_REF_COLS[gas]
        mrow = master[(master.gas == gas) & (master.config == config) & (master.model == family)]
        if mrow.empty or pd.isna(mrow.iloc[0].get("R2_test", np.nan)):
            print(f"  [{k}/{len(cells)}] {gas}/{config}/{family}: no scorable row in s9, skipped")
            rows.append({"node": args.node, "gas": gas, "config": config, "model": family,
                         "seed": np.nan, "notes": "no scorable row in master table"})
            continue
        mrow = mrow.iloc[0]
        # Valid rows as in s9 (the OLS fitted on all training rows only defines the row mask)
        refit_ols_on_train(df, train_mask)
        feature_cols = get_feature_columns(gas, config, df)
        valid = df[feature_cols + [ref_col]].notna().all(axis=1)
        train_valid = valid & train_mask
        idx = np.where(train_valid.values)[0]
        n_train = len(idx)
        if int(mrow["n_train"]) != n_train:
            sys.exit(f"ERROR: training rows not reproduced {gas}/{config}/{family}: "
                     f"{n_train} vs {mrow['n_train']}")
        # Validation block: fixed dates, or the last val_frac of the valid training rows
        if args.val_start and args.val_end:
            in_block = (ts_all >= pd.Timestamp(args.val_start, tz="UTC")) & \
                       (ts_all <= pd.Timestamp(args.val_end, tz="UTC") + pd.Timedelta(days=1))
            val_idx = idx[in_block.values[idx]]
            inner_idx = idx[~in_block.values[idx]]
        else:
            n_val = int(round(args.val_frac * n_train))
            inner_idx, val_idx = idx[:-n_val], idx[-n_val:]
        inner_mask = pd.Series(False, index=df.index); inner_mask.iloc[inner_idx] = True
        val_start, val_end = ts_all.iloc[val_idx[0]], ts_all.iloc[val_idx[-1]]
        # Input OLS of configurations B and C refitted on the inner training rows only
        ols_variant = "none"
        if config in ("B", "C"):
            refit_ols_on_train(df, inner_mask)
            ols_variant = "inner_train"
            still_valid = df[feature_cols + [ref_col]].notna().all(axis=1).values
            if not still_valid[idx].all():
                # The inner OLS leaves NaN in rows that were valid before: drop them
                inner_idx = inner_idx[still_valid[inner_idx]]
                val_idx = val_idx[still_valid[val_idx]]
        X_in = df.iloc[inner_idx][feature_cols].to_numpy(dtype=float)
        y_in = df.iloc[inner_idx][ref_col].to_numpy(dtype=float)
        X_val = df.iloc[val_idx][feature_cols].to_numpy(dtype=float)
        y_val = df.iloc[val_idx][ref_col].to_numpy(dtype=float)
        st = status(current_cell=f"{gas}/{config}/{family}")
        print(f"  [{k}/{len(cells)}] {gas}/{config}/{family}: inner={len(inner_idx)} val={len(val_idx)} "
              f"({val_start:%Y-%m-%d} .. {val_end:%Y-%m-%d}) | {st['percent']}% ETA {st['eta_min']} min")
        if args.dry_run:
            rows.append({"node": args.node, "gas": gas, "config": config, "model": family,
                         "seed": np.nan, "n_train_inner": len(inner_idx), "n_val": len(val_idx),
                         "val_start": val_start, "val_end": val_end, "notes": "dry_run"})
            continue
        params = ast.literal_eval(mrow["best_params"]) if isinstance(mrow["best_params"], str) else {}
        # DNN and LSTM are fitted once per seed; the other families use the pipeline seed
        seeds = args.seeds if family in ("DNN", "LSTM") else [cfg.ML_RANDOM_STATE]
        for seed in seeds:
            t_fit = time.time()
            est, _, _ = create_model(family, len(feature_cols))
            if family == "LSTM":
                est = LSTMRegressorTS(**est.get_params())
            est.set_params(**params)
            if "random_state" in est.get_params():
                est.set_params(random_state=seed)
            np.random.seed(seed); torch.manual_seed(seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if family == "LSTM":
                    # Timestamps travel as the last column of X (gap-free windows)
                    Xi = np.column_stack([X_in, epoch_seconds(df.iloc[inner_idx][cfg.COL_TS].values)])
                    Xv = np.column_stack([X_val, epoch_seconds(df.iloc[val_idx][cfg.COL_TS].values)])
                    est.fit(Xi, y_in)
                    pred = est.predict(Xv)
                else:
                    est.fit(X_in, y_in)
                    if family == "LightGBM":
                        est = _refit_lgbm_early_stopping(est, params, X_in, y_in)
                    pred = np.asarray(est.predict(X_val), dtype=float)
            # Non-finite LSTM predictions are masked; at least 50 scored rows are required
            ok = np.isfinite(pred)
            m = evaluate_model(y_val[ok], pred[ok]) if ok.sum() >= 50 else \
                {"RMSE": np.nan, "R2": np.nan, "MAE": np.nan, "nRMSE": np.nan}
            rows.append({"node": args.node, "gas": gas, "config": config, "model": family, "seed": seed,
                         "n_train_inner": len(inner_idx), "n_val": len(val_idx), "n_val_scored": int(ok.sum()),
                         "val_start": val_start, "val_end": val_end, "val_RMSE": m["RMSE"], "val_R2": m["R2"],
                         "val_MAE": m["MAE"], "val_nRMSE": m["nRMSE"], "best_params": str(params),
                         "params_source": mrow["params_source"], "ols_variant": ols_variant,
                         "R2_test_persisted": float(mrow["R2_test"]), "RMSE_test_persisted": float(mrow["RMSE"]),
                         "n_test": int(mrow["n_test"]), "model_path": mrow["model_path"],
                         "fit_seconds": round(time.time() - t_fit, 1), "notes": ""})
            print(f"      seed {seed}: val_RMSE={m['RMSE']:.3f} val_R2={m['R2']:.3f} "
                  f"scored {int(ok.sum())}/{len(val_idx)} ({time.time() - t_fit:.0f} s)")
        pd.DataFrame(rows).reindex(columns=VAL_COLS).to_csv(val_path, index=False)
        cell_minutes.append((time.time() - t_cell) / 60)
        status(current_cell=None)

    if args.dry_run:
        print(f"\n  dry-run OK: {len(rows)} cells ({(time.time() - t0) / 60:.1f} min)")
        return

    # Selection per target on the validation RMSE averaged over the seeds
    val = pd.DataFrame(rows).reindex(columns=VAL_COLS)
    val = val[val["val_RMSE"].notna()]
    agg = (val.groupby(["gas", "config", "model"], as_index=False)
              .agg(val_RMSE=("val_RMSE", "mean"), val_R2=("val_R2", "mean"), n_seeds=("seed", "count"),
                   n_val=("n_val", "first"), n_val_scored=("n_val_scored", "min"),
                   val_start=("val_start", "first"), val_end=("val_end", "first"),
                   R2_test=("R2_test_persisted", "first"), RMSE_test=("RMSE_test_persisted", "first")))
    out = []
    for gas in gases:
        g = agg[agg.gas == gas].sort_values("val_RMSE")
        if g.empty:
            continue
        best = g.iloc[0]
        thr = best["val_RMSE"] * 1.02  # Parsimony tolerance (2% above the lowest RMSE)
        cands = g[g["val_RMSE"] <= thr].copy()
        cands["cx"] = cands["model"].map(COMPLEXITY)
        pars = cands.sort_values(["cx", "val_RMSE"]).iloc[0]
        # Stored test R2: maximum over the combinations and rank of the selected one
        max_test = g["R2_test"].max()
        g_by_test = g.sort_values("R2_test", ascending=False).reset_index(drop=True)
        for rule, pick in (("min_val_RMSE", best), ("parsimony_within_0.02", pars)):
            rank = int(g_by_test.index[(g_by_test.model == pick["model"]) & (g_by_test.config == pick["config"])][0]) + 1
            out.append({"node": args.node, "gas": gas, "rule": rule, "locked_model": pick["model"],
                        "locked_config": pick["config"], "val_RMSE": pick["val_RMSE"], "val_R2": pick["val_R2"],
                        "n_val": int(pick["n_val"]), "n_val_scored": int(pick["n_val_scored"]),
                        "val_start": pick["val_start"], "val_end": pick["val_end"],
                        "R2_test_locked": pick["R2_test"], "RMSE_test_locked": pick["RMSE_test"],
                        "R2_test_max_over_15": max_test, "delta_vs_max": pick["R2_test"] - max_test,
                        "rank_of_locked_by_test": rank, "n_combinations": int(len(g))})
    per_gas = pd.DataFrame(out)
    per_gas_path = OUT_DIR / f"locked_selection_per_gas_node{args.node}{args.tag}.csv"
    per_gas.to_csv(per_gas_path, index=False)
    agg.to_csv(OUT_DIR / f"locked_selection_validation_mean_node{args.node}{args.tag}.csv", index=False)
    status(current_cell=None, finished=True)
    print(f"\n  written {per_gas_path.name} ({len(per_gas)} rows) in {(time.time() - t0) / 60:.0f} min")
    print(per_gas[["gas", "rule", "locked_model", "locked_config", "val_RMSE", "R2_test_locked",
                   "R2_test_max_over_15", "delta_vs_max", "rank_of_locked_by_test"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
