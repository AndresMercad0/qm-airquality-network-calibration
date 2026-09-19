"""
Step s3: quality control of the Alphasense cells

Reads the s2 CSV and sets to NaN the Alphasense concentrations outside
the plausibility limits, the Hampel spikes and the periods of thermal
instability. Writes a flagging report and the clean CSV, whose CO and
O3 columns are the comparators of the later calibration steps.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    NODE_ID, DATA_S2, DATA_S3, RESULTS_S3, COL_TS, COL_TEMP,
    ALPHA_HAMPEL_H, ALPHA_HAMPEL_K, ALPHA_BOUNDS,
    TEMP_DELTA_WINDOW, TEMP_DELTA_THRESH, TEMP_RECOVERY_WIN,
)
from utils import (
    hampel_filter, flag_bounds,
    flag_thermal_instability, count_thermal_events,
    compute_stats, first_last_ts, detect_csv, extract_dates_from_csv,
    build_output_name,
)


# -------------------------------------------------------------------------
# Quality control of the Alphasense cells (ug/m3 domain)
# -------------------------------------------------------------------------
def run_s3(input_csv: Path = None) -> Path:
    """
    Clean the Alphasense columns already converted to ug/m3: plausibility limits,
    Hampel filter and thermal instability. The cells are the comparators of the OLS
    step for CO and O3, so they are cleaned before any calibration.
    """
    if input_csv is None:
        input_csv = detect_csv(DATA_S2, f"{NODE_ID}_ALPHASENSE_UG_M3_wide_data_*.csv")
    if input_csv is None or not input_csv.exists():
        raise FileNotFoundError(f"No s2 CSV found in {DATA_S2}")

    print(f"s3, Alphasense QC: {input_csv.name}")
    df = pd.read_csv(input_csv, parse_dates=[COL_TS])
    n_total = len(df)
    print(f"  Rows: {n_total:,}")

    # Thermal mask (applies to every gas column)
    thermal_mask = flag_thermal_instability(
        df[COL_TEMP], TEMP_DELTA_WINDOW, TEMP_DELTA_THRESH, TEMP_RECOVERY_WIN)
    n_thermal_rows = int(thermal_mask.sum())
    n_thermal_evts = count_thermal_events(thermal_mask)
    print(f"  Thermal instability: {n_thermal_evts} events ({n_thermal_rows:,} rows)")

    # Report folder of this run
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_S3 / f"resultado_{run_ts}"
    result_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = []

    for col, (lo, hi) in ALPHA_BOUNDS.items():
        before = compute_stats(df[col], "before")
        all_flags = pd.Series(False, index=df.index)

        # Plausibility limits
        mask_bounds = flag_bounds(df[col], lo, hi)
        n_bounds = int(mask_bounds.sum())
        df.loc[mask_bounds, col] = np.nan
        all_flags |= mask_bounds

        # Hampel filter
        mask_hamp = hampel_filter(df[col], ALPHA_HAMPEL_H, ALPHA_HAMPEL_K)
        n_hampel = int(mask_hamp.sum())
        df.loc[mask_hamp, col] = np.nan
        all_flags |= mask_hamp

        # Thermal instability
        mask_therm = thermal_mask & (~df[col].isna())
        n_thermal = int(mask_therm.sum())
        df.loc[mask_therm, col] = np.nan
        all_flags |= mask_therm

        n_flagged = int(all_flags.sum())
        n_valid = n_total - int(df[col].isna().sum())
        after = compute_stats(df[col], "after")
        first_ts_val, last_ts_val = first_last_ts(df, all_flags)

        stats_rows.append({
            "column": col, "n_total": n_total,
            "n_bounds_flagged": n_bounds, "n_hampel_flagged": n_hampel,
            "n_thermal_flagged": n_thermal, "n_thermal_events": n_thermal_evts,
            "n_flagged_total": n_flagged,
            "pct_flagged": round(100.0 * n_flagged / n_total, 4),
            "n_valid": n_valid,
            "pct_valid": round(100.0 * n_valid / n_total, 4),
            **before, **after,
            "first_flag_ts": first_ts_val, "last_flag_ts": last_ts_val,
        })

    stats_df = pd.DataFrame(stats_rows)

    print("\nSummary (% flagged):")
    print(stats_df[["column", "n_bounds_flagged", "n_hampel_flagged",
                     "n_thermal_flagged", "pct_flagged"]].to_string(index=False))

    # Save the clean CSV
    DATA_S3.mkdir(parents=True, exist_ok=True)
    start_date, end_date = extract_dates_from_csv(df)
    out_name = build_output_name(NODE_ID, "QC_ALPHASENSE_UG_M3", start_date, end_date)
    out_csv = DATA_S3 / out_name
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name}")

    # Save the statistics and the Markdown report
    stats_df.to_csv(result_dir / "stats_detalladas.csv", index=False)

    lines = [
        f"# Cleaning report: {NODE_ID} ALPHASENSE (s3)\n",
        f"**Run at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Input:** {input_csv.name}  ",
        f"**Output:** {out_csv.name}  ",
        f"**Rows:** {n_total:,}  ",
        f"**Thermal events:** {n_thermal_evts} ({n_thermal_rows:,} rows)\n",
        "## By column",
        "| Column | Flagged | % | P99 before | P99 after |",
        "|---------|---------|---|-----------|-------------|",
    ]
    for _, r in stats_df.iterrows():
        p99_b = r.get("before_p99")
        p99_a = r.get("after_p99")
        b_str = f"{p99_b:.4f}" if p99_b is not None else "N/A"
        a_str = f"{p99_a:.4f}" if p99_a is not None else "N/A"
        lines.append(
            f"| {r['column']} | {int(r['n_flagged_total']):,} | {r['pct_flagged']:.2f}% "
            f"| {b_str} | {a_str} |")

    (result_dir / "reporte_limpieza.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Reports in: {result_dir}")

    return out_csv


if __name__ == "__main__":
    run_s3()
