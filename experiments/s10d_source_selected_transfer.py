"""
Transfer of the calibration between the two nodes

Applies the models of one node to the other in three scenarios: direct
transfer, OLS input refitted at the destination, and oracle candidates.
Scores the full period and the test period; writes
results/cross_node/source_selected_transfer.csv.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    REF_CSV_NAME, PREDICT_CHUNK, load_node_pipeline, versions_banner, make_ts_subclass, epoch_seconds,
)

# -------------------------------------------------------------------------
# Outputs, pollutants and expected OLS coefficients of each source node
# -------------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "cross_node"
GASES = ["PM2.5", "CO", "NO2", "O3"]
# OLS fitted on the training period (slopes, intercept); the run stops if they are not reproduced
EXPECTED_OLS = {
    3: {"NO2": ([0.7931, -0.2103, -0.1903], 27.4899), "CO": ([-412.9037, 4.5092, -0.0026], 415.8909),
        "O3": ([0.9787, 2.3411, -1.1382], 67.1736), "PM2.5": ([0.7010, -0.0370, -0.0199], 4.6786)},
    5: {"NO2": ([0.8888, -0.2034, -0.1076], 20.7304), "CO": ([-41.8313, 6.2262, 0.0362], 128.8074),
        "O3": ([-15.6624, 2.9029, -1.5341], 108.0308), "PM2.5": ([0.7221, 0.0088, -0.0455], 7.3747)},
}
# Order of model families for the parsimony rule (lower is simpler)
COMPLEXITY = {"RF": 0, "XGBoost": 1, "LightGBM": 1, "DNN": 2, "LSTM": 3}


# -------------------------------------------------------------------------
# Candidate models of the source node
# -------------------------------------------------------------------------
def load_master(cfg):
    """
    Return the master table of the node.
    LSTM rows of the corrected search replace the original LSTM rows when they exist.
    """
    frames = []
    for name in ("tabla_maestra_resultados.csv", "tabla_maestra_lstm_v6.csv"):
        f = cfg.RESULTS_S9 / name
        if f.exists():
            d = pd.read_csv(f)
            d["params_source"] = name
            frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    if "status" in m.columns:
        v6 = m[m["params_source"] == "tabla_maestra_lstm_v6.csv"]
        v6 = v6[v6["status"].astype(str).str.startswith("ok")]
        keys = set(zip(v6.gas, v6.config))
        base = m[m["params_source"] == "tabla_maestra_resultados.csv"]
        replaced = (base.model == "LSTM") & pd.Series(list(zip(base.gas, base.config)), index=base.index).isin(keys)
        m = pd.concat([base[~replaced], v6], ignore_index=True)
    return m[m["R2_test"].notna()]


def candidates(master, gas, locked_csv):
    """
    Return (selection_rule, model, config) for one pollutant: the validation-selected model,
    the simplest family within 0.05 of the highest test R2, and the highest test R2.
    """
    g = master[master.gas == gas].sort_values("R2_test", ascending=False)
    out = []
    if locked_csv is not None and locked_csv.exists():
        lk = pd.read_csv(locked_csv)
        lk = lk[(lk.gas == gas) & (lk.rule == "min_val_RMSE")]
        if not lk.empty:
            out.append(("locked_min_val_RMSE", lk.iloc[0]["locked_model"], lk.iloc[0]["locked_config"]))
    top = g.iloc[0]
    pick = top
    for _, r in g.iterrows():
        if (top["R2_test"] - r["R2_test"]) < 0.05 and COMPLEXITY.get(r["model"], 99) < COMPLEXITY.get(top["model"], 99):
            pick = r
            break
    out.append(("v5_parsimony", pick["model"], pick["config"]))
    out.append(("max_test_R2", top["model"], top["config"]))
    return out


def load_model(cfg, row, LSTMRegressorTS, DNNRegressor):
    """Rebuild the stored model of one master-table row, or return None if the file is missing."""
    import joblib
    path = cfg.RESULTS_S9_MODELS / Path(row["model_path"]).name
    if not path.exists():
        return None
    params = ast.literal_eval(row["best_params"]) if isinstance(row["best_params"], str) else {}
    n_features = int(row["n_features"])
    if row["model"] == "LSTM":
        m = LSTMRegressorTS(random_state=cfg.ML_RANDOM_STATE, window=cfg.LSTM_WINDOW,
                            **{k: params[k] for k in ("lstm_units", "fc_dim", "dropout") if k in params})
        m.load_torch(path, n_features)
    elif row["model"] == "DNN":
        m = DNNRegressor(random_state=cfg.ML_RANDOM_STATE,
                         **{k: params[k] for k in ("hidden_dim_1", "hidden_dim_2", "dropout") if k in params})
        m.load_torch(path, n_features)
    else:
        m = joblib.load(path)
    return m


# -------------------------------------------------------------------------
# Transfer in both directions
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Transfer the selected models between the two nodes: direct, OLS-refitted and oracle")
    ap.add_argument("--directions", nargs="*", default=["3-5", "5-3"])
    ap.add_argument("--gases", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print("=" * 70)
    print("  Cross-node transfer: direct, OLS-refitted and oracle")
    print("=" * 70)
    versions_banner()
    gases = args.gases or GASES
    rows = []

    for d in args.directions:
        src, dst = (int(x) for x in d.split("-"))
        print(f"\n{'-' * 70}\n  Direction: Node {src} -> Node {dst}\n{'-' * 70}")
        cfg, utils = load_node_pipeline(src)
        from ml.data_loader import prepare_dataset, get_train_test_masks, get_feature_columns
        from ml.ols_refit import refit_ols_on_train
        from ml.models import LSTMRegressor, DNNRegressor
        from utils import compute_ml_metrics, nivel_calidad_ml
        import cross_node_features as partner
        LSTMRegressorTS = make_ts_subclass(LSTMRegressor)

        # Source node: OLS fitted on its training period
        s1 = utils.detect_csv(cfg.DATA_S1, f"{cfg.NODE_ID}_QC_RAW_wide_data_*.csv")
        s3 = utils.detect_csv(cfg.DATA_S3, f"{cfg.NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
        s4 = utils.detect_csv(cfg.DATA_S4, f"{cfg.NODE_ID}_PREPROCESS_wide_data_*.csv")
        ref = cfg.REF_DATA_DIR / REF_CSV_NAME[src]
        df_src = prepare_dataset(s1, s3, s4, ref)
        train_src, _ = get_train_test_masks(df_src)
        ols_info = refit_ols_on_train(df_src, train_src)
        src_coefs = {}
        for gas in gases:
            info = ols_info.get(gas)
            if info is None:
                sys.exit(f"ERROR: no source OLS for {gas}")
            cdict = info["coefs"]
            feats = [k for k in cdict if k != "intercept"]
            coef = np.asarray([cdict[k] for k in feats], dtype=float)
            b0 = float(cdict["intercept"])
            exp_c, exp_b0 = EXPECTED_OLS[src][gas]
            if not (np.allclose(coef, exp_c, atol=5e-4) and abs(b0 - exp_b0) < 5e-4):
                sys.exit(f"ERROR: source OLS coefficients do not reproduce the expected values for Node {src} {gas}: "
                         f"{coef.round(4)} {b0:.4f} vs {exp_c} {exp_b0}")
            src_coefs[gas] = {"features": list(feats), "coef": coef.tolist(), "intercept": b0}
            print(f"  source OLS {gas}: {dict(zip(feats, coef.round(4)))} b0={b0:.4f} (as expected)")
        master = load_master(cfg)
        locked_csv = OUT_DIR / f"locked_selection_per_gas_node{src}.csv"
        if not locked_csv.exists():
            print(f"  WARNING: {locked_csv.name} not found: only v5_parsimony and max_test_R2 are scored")

        # Destination node: three versions of the OLS input of configurations B and C
        # (source coefficients, refit on the destination training period, fit on the full destination period)
        loaders = partner._load_partner_data()
        prep = partner._prepare_partner_features
        dest_csv = loaders[0]
        df_dst = prep(dest_csv, loaders[1], loaders[2], loaders[-1])
        DEST_REF = partner.PARTNER_REF_COLS
        ts_dst = pd.to_datetime(df_dst[cfg.COL_TS], utc=True)
        test_dst = ts_dst >= pd.Timestamp(cfg.ML_TEST_START, tz="UTC")
        train_dst = ts_dst <= pd.Timestamp(cfg.ML_TRAIN_END, tz="UTC")
        print(f"  dest frame: {len(df_dst)} rows (test rows {int(test_dst.sum())})")
        from sklearn.linear_model import LinearRegression
        dest_fit_r2 = {}
        for gas in gases:
            feats = src_coefs[gas]["features"]
            missing = [f for f in feats if f not in df_dst.columns]
            if missing:
                sys.exit(f"ERROR: destination lacks columns {missing} for {gas}")
            df_dst[f"c_hat_ols_destfull_{gas}"] = df_dst[f"c_hat_ols_{gas}"]
            X = df_dst[feats].to_numpy(dtype=float)
            df_dst[f"c_hat_ols_src_{gas}"] = X @ np.asarray(src_coefs[gas]["coef"]) + src_coefs[gas]["intercept"]
            ref_col = DEST_REF[gas]
            v = df_dst[feats + [ref_col]].notna().all(axis=1) & train_dst
            lr = LinearRegression().fit(df_dst.loc[v, feats].to_numpy(dtype=float), df_dst.loc[v, ref_col].to_numpy(dtype=float))
            ok = df_dst[feats].notna().all(axis=1)
            df_dst[f"c_hat_ols_desttrain_{gas}"] = np.nan
            df_dst.loc[ok, f"c_hat_ols_desttrain_{gas}"] = lr.predict(df_dst.loc[ok, feats].to_numpy(dtype=float))
            dest_fit_r2[gas] = {"destination_train": float(lr.score(df_dst.loc[v, feats].to_numpy(dtype=float), df_dst.loc[v, ref_col].to_numpy(dtype=float))), "n_fit": int(v.sum())}
        if args.dry_run:
            continue

        # Score every candidate on the full destination period and on its test period
        for gas in gases:
            ref_col = DEST_REF[gas]
            for rule, model_name, config in candidates(master, gas, locked_csv):
                mrow = master[(master.gas == gas) & (master.model == model_name) & (master.config == config)]
                if mrow.empty:
                    print(f"  {gas} {rule}: {model_name}-{config} has no row in the master table, skipped")
                    continue
                mrow = mrow.iloc[0]
                model = load_model(cfg, mrow, LSTMRegressorTS, DNNRegressor)
                if model is None:
                    print(f"  {gas} {rule}: model {Path(mrow['model_path']).name} missing, skipped")
                    continue
                variants = [("none", None)] if config == "A" else [
                    ("source", f"c_hat_ols_src_{gas}"), ("destination_train", f"c_hat_ols_desttrain_{gas}"),
                    ("destination_full", f"c_hat_ols_destfull_{gas}")]
                for variant, col in variants:
                    if col is not None:
                        df_dst[f"c_hat_ols_{gas}"] = df_dst[col]
                    feature_cols = get_feature_columns(gas, config, df_dst)
                    valid = df_dst[feature_cols + [ref_col]].notna().all(axis=1)
                    for period, pmask in (("full", pd.Series(True, index=df_dst.index)), ("test", test_dst)):
                        sel = valid & pmask
                        if sel.sum() < 100:
                            continue
                        X = df_dst.loc[sel, feature_cols].to_numpy(dtype=float)
                        y = df_dst.loc[sel, ref_col].to_numpy(dtype=float)
                        if model_name == "LSTM":
                            pred = model.predict(np.column_stack([X, epoch_seconds(df_dst.loc[sel, cfg.COL_TS].values)]))
                        else:
                            pred = np.asarray(model.predict(X), dtype=float)
                        ok = np.isfinite(pred)
                        if ok.sum() < 100:
                            continue
                        met = compute_ml_metrics(y[ok], pred[ok])
                        r2_local = float(mrow["R2_test"])
                        scenario = "source-selected" if variant in ("source", "none") else "adapted"
                        rows.append({"direction": f"N{src}_to_N{dst}", "gas": gas, "scenario": scenario,
                                     "selection_rule": rule, "model": model_name, "config": config,
                                     "ols_variant": variant, "period": period, "R2_local_test": r2_local,
                                     "R2_transferred": met["R2"], "delta_R2": met["R2"] - r2_local,
                                     "RMSE_transferred": met["RMSE"], "MAE_transferred": met["MAE"],
                                     "nRMSE_transferred": met["nRMSE"],
                                     "CEN_transferred": nivel_calidad_ml(met["R2"], met["nRMSE"]),
                                     "n_obs": int(sel.sum()), "n_pred_valid": int(ok.sum()),
                                     "src_ols_coefs": json.dumps(src_coefs[gas]) if variant == "source" else "",
                                     "dest_ols_r2_fit": dest_fit_r2[gas]["destination_train"] if variant == "destination_train" else np.nan,
                                     "model_path": Path(mrow["model_path"]).name, "notes": ""})
                        print(f"  {gas:5s} {rule:20s} {model_name}-{config} {variant:17s} {period:4s}: "
                              f"R2 {met['R2']:+.3f} (local {r2_local:+.3f}, d {met['R2'] - r2_local:+.3f}) n={int(ok.sum())}")
                del model

    if args.dry_run:
        print(f"\n  dry-run OK ({(time.time() - t0) / 60:.1f} min)")
        return
    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "source_selected_transfer.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  written {out_path} ({len(out)} rows) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
