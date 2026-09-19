"""
Random Forest ablation of the model inputs

Trains a Random Forest with the search protocol of s9 on four reduced
input sets (config.py): environment and time without gas signal (D,
D_strict) and sensor signals only (S, S1), by default on the valid rows
of Config A. Writes the ablation table tabla_maestra_ablacion_v6.csv.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REF_CSV_NAME, load_node_pipeline, versions_banner

# -------------------------------------------------------------------------
# Configuration constants
# -------------------------------------------------------------------------
GASES = ["CO", "NO2", "PM2.5"]


# -------------------------------------------------------------------------
# Training of the ablation configurations
# -------------------------------------------------------------------------
def main():
    """Train every pending cell; cells already in the output table are skipped."""
    ap = argparse.ArgumentParser(description="Random Forest ablation of the model inputs.")
    ap.add_argument("--node", type=int, choices=(3, 5), required=True)
    ap.add_argument("--gases", nargs="*", default=None)
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--n-iter", type=int, default=None, help="RF search iterations (default HP_RF_N_ITER)")
    ap.add_argument("--no-align", action="store_true", help="do not intersect the rows with Config A")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    print("=" * 70)
    print(f"  Ablation (RF) | Node {args.node}")
    print("=" * 70)
    versions_banner()

    cfg, utils = load_node_pipeline(args.node)
    import config as cfgmod
    from ml.data_loader import prepare_dataset, get_train_test_masks, get_feature_columns
    from ml.ols_refit import refit_ols_on_train
    from ml.trainer import train_all_combinations

    # Optional override of the search iterations (config and ml.models each hold the value)
    if args.n_iter is not None:
        cfgmod.HP_RF_N_ITER = args.n_iter
        import ml.models as mm
        mm.HP_RF_N_ITER = args.n_iter
    n_iter = cfgmod.HP_RF_N_ITER
    gases = args.gases or GASES
    configs = args.configs or list(cfg.ML_CONFIGS_ABLATION)
    log_path = cfg.RESULTS_S9 / "s9_ablation_v6.log"
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)
    logging.getLogger().setLevel(logging.INFO)

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
    refit_ols_on_train(df, train_mask)  # Adds the train-only OLS columns, as in s9
    master = pd.read_csv(cfg.RESULTS_S9 / "tabla_maestra_resultados.csv")

    out_csv = cfg.RESULTS_S9 / "tabla_maestra_ablacion_v6.csv"
    status_path = cfg.RESULTS_S9 / "e5_status.json"
    done_rows = pd.read_csv(out_csv).to_dict("records") if out_csv.exists() else []
    done = {(r["gas"], r["config"]) for r in done_rows}
    cells = [(g, c) for g in gases for c in configs if (g, c) not in done]
    print(f"  configs={configs} gases={gases} n_iter={n_iter} align={'A' if not args.no_align else 'none'}; "
          f"pending cells {len(cells)} (done {len(done)})")
    cell_min: list[float] = []

    def status(**kw):
        d = {"node": args.node, "cells_total": len(cells) + len(done), "cells_done": len(done),
             "elapsed_min": round((time.time() - t0) / 60, 1),
             "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}
        per = (sum(cell_min) / len(cell_min)) if cell_min else 12.0
        d["percent"] = round(100 * d["cells_done"] / max(d["cells_total"], 1), 1)
        d["eta_min"] = round(per * (d["cells_total"] - d["cells_done"]), 1)
        if not args.dry_run:
            status_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        return d

    for gas, config in cells:
        t_cell = time.time()
        feats = get_feature_columns(gas, config, df)
        featsA = get_feature_columns(gas, "A", df)
        ref_col = cfg.ML_REF_COLS[gas]
        valid_cfg = df[feats + [ref_col]].notna().all(axis=1)
        validA = df[featsA + [ref_col]].notna().all(axis=1)
        # Row alignment: training and test rows restricted to the valid rows of Config A,
        # so every ablation set and Config A are scored on the same rows
        tr, te = train_mask.copy(), test_mask.copy()
        align = "none"
        if not args.no_align:
            tr, te = tr & validA, te & validA
            align = "A"
        n_tr, n_te = int((valid_cfg & tr).sum()), int((valid_cfg & te).sum())
        # Random Forest of Config A in the s9 master table, as the comparison value
        a_row = master[(master.gas == gas) & (master.config == "A") & (master.model == "RF")]
        st = status(current_cell=f"{gas}/{config}")
        print(f"\n  [{gas}/{config}] features={len(feats)} {feats} | rows train={n_tr} test={n_te} "
              f"(Config A RF: {int(a_row.n_train.iloc[0]) if len(a_row) else '?'}/{int(a_row.n_test.iloc[0]) if len(a_row) else '?'}) "
              f"| {st['percent']}% ETA {st['eta_min']} min")
        if args.dry_run:
            continue
        res_df, _ = train_all_combinations(df, tr, te, cfg.RESULTS_S9_MODELS,
                                           models=["RF"], gases=[gas], configs=[config])
        if res_df.empty:
            print("  no result (not enough rows)")
            continue
        r = res_df.iloc[0].to_dict()
        r.update({"align_rows_to": align, "n_iter": n_iter,
                  "R2_test_configA_RF": float(a_row.R2_test.iloc[0]) if len(a_row) else float("nan")})
        done_rows.append(r)
        done.add((gas, config))
        pd.DataFrame(done_rows).to_csv(out_csv, index=False)
        cell_min.append((time.time() - t_cell) / 60)
        status(current_cell=None)
        print(f"  -> R2_test={r['R2_test']:.4f} RMSE={r['RMSE']:.3f} (Config A RF {r['R2_test_configA_RF']:.4f}) "
              f"in {cell_min[-1]:.1f} min")

    if args.dry_run:
        print(f"\n  dry-run OK ({(time.time() - t0) / 60:.1f} min)")
        return
    status(current_cell=None, finished=True)
    out = pd.DataFrame(done_rows)
    print(f"\n  written {out_csv} ({len(out)} rows) in {(time.time() - t0) / 60:.0f} min")
    print(out[["gas", "config", "n_features", "n_train", "n_test", "R2_test", "RMSE", "R2_test_configA_RF"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
