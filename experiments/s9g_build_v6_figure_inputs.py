"""
Input tables of the figures

Builds, per node, the tables read by the figure scripts from the outputs
of the other experiments: metricas_por_nivel_de_costo_v6.csv (schema of
s7, intersection basis, test period), tabla_maestra_v6.csv (LSTM rows of
s9f) and tabla_maestra_locked_v6.csv (models selected on validation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Configuration constants
# -------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RES = REPO_ROOT / "results" / "cross_node"
TARGETS = ["PM2.5", "CO", "NO2", "O3", "NO2_alpha"]


# -------------------------------------------------------------------------
# Master table of s9
# -------------------------------------------------------------------------
def master_with_v6(s9_dir: Path) -> pd.DataFrame:
    """
    Concatenate the s9 master tables.
    LSTM rows are replaced by those of tabla_maestra_lstm_v6.csv when that file exists.
    """
    frames = []
    for name in ("tabla_maestra_resultados.csv", "tabla_maestra_no2alpha.csv",
                 "tabla_maestra_no2alpha_lstm.csv"):
        f = s9_dir / name
        if f.exists():
            d = pd.read_csv(f)
            d["source_table"] = name
            frames.append(d)
    m = pd.concat(frames, ignore_index=True)
    v6 = s9_dir / "tabla_maestra_lstm_v6.csv"
    if v6.exists():
        d6 = pd.read_csv(v6)
        d6 = d6[d6["status"].astype(str).str.startswith("ok")].copy()
        d6["source_table"] = "tabla_maestra_lstm_v6.csv"
        keys = set(zip(d6.gas, d6.config))
        replaced = (m.model == "LSTM") & pd.Series(list(zip(m.gas, m.config)), index=m.index).isin(keys)
        m = pd.concat([m[~replaced], d6[[c for c in m.columns if c in d6.columns]]], ignore_index=True)
    return m


# -------------------------------------------------------------------------
# Tables per node
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build the input tables of the figures from the experiment outputs.")
    ap.add_argument("--node", type=int, choices=(3, 5), required=True)
    ap.add_argument("--fallback", action="store_true", help="without the validation selection, use the highest R2_test")
    args = ap.parse_args()
    s9 = REPO_ROOT / f"results/node_{args.node}/s9_ml"
    e4 = RES / f"single_basis_metrics_node{args.node}.csv"
    if not e4.exists():
        sys.exit(f"ERROR: missing {e4}")
    d = pd.read_csv(e4)
    it = d[(d.basis == "intersection") & (d.period == "test")]

    # Schema of s7 with the values of the intersection basis on the test period
    spec = [  # Tuples (gas, method, clip_zero, nivel_costo, metodo)
        ("NO2_alpha", "alphasense_factory", False, "Mid-cost", "Cal. parámetros de fábrica (AAN-803-05)"),
        ("NO2_alpha", "ols_train_only", False, "Mid-cost OLS", "OLS colocalización (Alphasense NO₂-B43F), solo entrenamiento"),
        ("NO2", "manufacturer_curve", False, "Low-cost curvas fab.", "Curva fabricante GM-102B"),
        ("NO2", "ols_train_only", False, "Low-cost OLS", "OLS colocalización, solo entrenamiento"),
        ("CO", "manufacturer_curve", False, "Low-cost curvas fab.", "Curva fabricante GM-702B"),
        ("CO", "ols_train_only", False, "Low-cost OLS", "OLS colocalización, solo entrenamiento"),
        ("O3", "manufacturer_curve", False, "Low-cost curvas fab.", "Curva fabricante MQ131"),
        ("O3", "ols_train_only", False, "Low-cost OLS", "OLS colocalización, solo entrenamiento"),
        ("PM2.5", "sps30_raw", False, "Low-cost (cal. fabricante)", "Calibración del fabricante"),
        ("PM2.5", "ols_train_only", False, "Low-cost OLS", "OLS colocalización, solo entrenamiento"),
    ]
    rows = []
    for gas, method, clip, nivel, metodo in spec:
        r = it[(it.gas == gas) & (it.method == method) & (it.clip_zero == clip)]
        if r.empty:
            print(f"  WARNING: no single-basis row for {gas}/{method}")
            continue
        r = r.iloc[0]
        rows.append({"r2": round(float(r.R2), 4), "rmse": round(float(r.RMSE), 2), "nrmse": round(float(r.nRMSE), 1),
                     "mae": round(float(r.MAE), 2), "mbe": round(float(r.MBE), 2), "pearson_r": round(float(r.pearson_r), 4),
                     "nivel": r.CEN_level, "n": int(r.n_scored), "mean_ref": np.nan, "mean_est": np.nan,
                     "gas": "NO2" if gas == "NO2_alpha" else gas, "nivel_costo": nivel, "metodo": metodo,
                     "referencia": r.ref_col, "col_estimado": r.pred_col, "col_referencia": r.ref_col,
                     "basis": "intersection_test_v6"})
    out1 = pd.DataFrame(rows)
    out1.to_csv(s9 / "metricas_por_nivel_de_costo_v6.csv", index=False)
    print(f"  written metricas_por_nivel_de_costo_v6.csv ({len(out1)} rows)")

    # Master table with the LSTM cells of tabla_maestra_lstm_v6.csv
    m = master_with_v6(s9)
    m.to_csv(s9 / "tabla_maestra_v6.csv", index=False)
    print(f"  written tabla_maestra_v6.csv ({len(m)} rows; LSTM cells replaced: "
          f"{int((m.source_table == 'tabla_maestra_lstm_v6.csv').sum())})")

    # Models selected on validation (rule min_val_RMSE), or the highest test R2 with --fallback
    e1 = RES / f"locked_selection_per_gas_node{args.node}.csv"
    locked_rows = []
    if e1.exists():
        lk = pd.read_csv(e1)
        lk = lk[lk.rule == "min_val_RMSE"]
        for _, r in lk.iterrows():
            mr = m[(m.gas == r.gas) & (m.model == r.locked_model) & (m.config == r.locked_config)]
            if mr.empty:
                print(f"  WARNING: selected {r.gas} {r.locked_model}-{r.locked_config} has no row in the master table")
                continue
            row = mr.iloc[0].to_dict()
            row.update({"selection": "locked_min_val_RMSE", "val_RMSE": r.val_RMSE,
                        "R2_test_max_over_15": r.R2_test_max_over_15, "delta_vs_max": r.delta_vs_max,
                        "rank_of_locked_by_test": r.rank_of_locked_by_test})
            locked_rows.append(row)
    elif args.fallback:
        for gas in TARGETS:
            g = m[(m.gas == gas) & m.R2_test.notna()].sort_values("R2_test", ascending=False)
            if g.empty:
                continue
            row = g.iloc[0].to_dict()
            row.update({"selection": "max_test (fallback, E1 pendiente)"})
            locked_rows.append(row)
    else:
        sys.exit(f"ERROR: missing {e1} (use --fallback for the provisional highest test R2)")
    out3 = pd.DataFrame(locked_rows)
    # Test R2 on the intersection basis, so the figures compare every method on the same
    # rows; the R2 on the own rows of the model is kept in R2_test_own
    ml_it = it[it.method == "ml"]
    out3["R2_test_own"] = out3["R2_test"]
    for i, r in out3.iterrows():
        tag = "LSTM_v6" if str(r.get("source_table", "")) == "tabla_maestra_lstm_v6.csv" else r["model"]
        hit = ml_it[(ml_it.gas == r["gas"]) & (ml_it.model == tag) & (ml_it.config == r["config"])]
        if not hit.empty:
            out3.at[i, "R2_test"] = float(hit.iloc[0]["R2"])
            out3.at[i, "RMSE"] = float(hit.iloc[0]["RMSE"])
            out3.at[i, "n_test"] = int(hit.iloc[0]["n_scored"])
        else:
            print(f"  WARNING: no single-basis row (intersection/test) for {r['gas']} {tag}-{r['config']}; the own R2 is kept")
    out3.to_csv(s9 / "tabla_maestra_locked_v6.csv", index=False)
    print(f"  written tabla_maestra_locked_v6.csv ({len(out3)} rows)")
    print(out3[["gas", "model", "config", "R2_test", "R2_test_own", "selection"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
