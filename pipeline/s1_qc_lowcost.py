"""
Step s1: quality control of the low-cost sensors

Reads the raw wide CSV of the node and sets flagged readings to NaN:
saturation, stuck sensor and Hampel spikes on the gas channels, physical
limits and Hampel spikes on PM2.5 and PM10, physical limits on
temperature and humidity. Writes the clean CSV and a flagging report.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    NODE_ID, DATA_RAW, DATA_S1, RESULTS_S1, COL_TS,
    ADC_COLS, VOLT_COLS,
    ADC_SAT_MIN, ADC_SAT_MAX, OZ_VOLT_LO, OZ_VOLT_HI,
    STUCK_MIN, HAMPEL_H, HAMPEL_K,
    PM25_MAX, PM10_MAX, TEMP_LO, TEMP_HI, RH_LO, RH_HI,
    COL_PM25, COL_PM10, COL_TEMP, COL_RH,
)
from utils import (
    hampel_filter, flag_bounds, flag_stuck,
    flag_adc_saturation, flag_voltage_saturation,
    compute_stats, first_last_ts, detect_csv, extract_dates_from_csv,
    build_output_name,
)


# -------------------------------------------------------------------------
# Quality control of the low-cost sensors (raw domain)
# -------------------------------------------------------------------------
def run_s1(input_csv: Path = None) -> Path:
    """
    Clean the raw wide CSV of the node and report the rows flagged in each column.
    Flagged values are set to NaN. Return the path of the clean CSV.
    """
    if input_csv is None:
        input_csv = detect_csv(DATA_RAW, f"{NODE_ID}_RAW_wide_data_*.csv")
    if input_csv is None or not input_csv.exists():
        raise FileNotFoundError(f"No raw CSV found in {DATA_RAW}")

    print(f"s1, low-cost QC: {input_csv.name}")
    df = pd.read_csv(input_csv, parse_dates=[COL_TS])
    n_total = len(df)
    print(f"  Rows: {n_total:,}")

    # Report folder of this run
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_S1 / f"resultado_{run_ts}"
    result_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = []

    def make_row(col, before, n_sat, n_stuck, n_stuck_events, max_stuck_run,
                 n_hampel, n_bounds, all_flags):
        n_flagged = int(all_flags.sum())
        n_valid = n_total - int(df[col].isna().sum())
        after = compute_stats(df[col], "after")
        first_ts_val, last_ts_val = first_last_ts(df, all_flags)
        return {
            "column": col, "n_total": n_total,
            "n_sat_flagged": n_sat, "n_stuck_flagged": n_stuck,
            "n_stuck_events": n_stuck_events, "max_stuck_run": max_stuck_run,
            "n_hampel_flagged": n_hampel, "n_bounds_flagged": n_bounds,
            "n_flagged_total": n_flagged,
            "pct_flagged": round(100.0 * n_flagged / n_total, 4),
            "n_valid": n_valid,
            "pct_valid": round(100.0 * n_valid / n_total, 4),
            **before, **after,
            "first_flag_ts": first_ts_val, "last_flag_ts": last_ts_val,
        }

    # Gas channels: saturation, stuck sensor and Hampel filter (transient spikes)
    for col in ADC_COLS + VOLT_COLS:
        before = compute_stats(df[col], "before")
        all_flags = pd.Series(False, index=df.index)

        if col in ADC_COLS:
            mask_sat = flag_adc_saturation(df[col], ADC_SAT_MIN, ADC_SAT_MAX)
        else:
            mask_sat = flag_voltage_saturation(df[col], OZ_VOLT_LO, OZ_VOLT_HI)
        df.loc[mask_sat, col] = np.nan
        all_flags |= mask_sat

        mask_stuck, n_events, max_run = flag_stuck(df[col], STUCK_MIN)
        df.loc[mask_stuck, col] = np.nan
        all_flags |= mask_stuck

        mask_hamp = hampel_filter(df[col], HAMPEL_H, HAMPEL_K)
        df.loc[mask_hamp, col] = np.nan
        all_flags |= mask_hamp

        stats_rows.append(make_row(
            col, before, int(mask_sat.sum()), int(mask_stuck.sum()),
            n_events, max_run, int(mask_hamp.sum()), 0, all_flags))

    # PM2.5 and PM10: physical limits and Hampel filter
    for col, lo, hi in [(COL_PM25, 0, PM25_MAX), (COL_PM10, 0, PM10_MAX)]:
        before = compute_stats(df[col], "before")
        all_flags = pd.Series(False, index=df.index)

        mask_bounds = flag_bounds(df[col], lo, hi)
        df.loc[mask_bounds, col] = np.nan
        all_flags |= mask_bounds

        mask_hamp = hampel_filter(df[col], HAMPEL_H, HAMPEL_K)
        df.loc[mask_hamp, col] = np.nan
        all_flags |= mask_hamp

        stats_rows.append(make_row(
            col, before, 0, 0, 0, 0, int(mask_hamp.sum()),
            int(mask_bounds.sum()), all_flags))

    # Temperature and humidity: physical limits
    for col, lo, hi in [(COL_TEMP, TEMP_LO, TEMP_HI), (COL_RH, RH_LO, RH_HI)]:
        before = compute_stats(df[col], "before")
        all_flags = pd.Series(False, index=df.index)

        mask_bounds = flag_bounds(df[col], lo, hi)
        df.loc[mask_bounds, col] = np.nan
        all_flags |= mask_bounds

        stats_rows.append(make_row(
            col, before, 0, 0, 0, 0, 0, int(mask_bounds.sum()), all_flags))

    stats_df = pd.DataFrame(stats_rows)

    print("\nSummary (% flagged):")
    print(stats_df[["column", "n_sat_flagged", "n_stuck_flagged",
                     "n_hampel_flagged", "n_bounds_flagged",
                     "pct_flagged"]].to_string(index=False))

    # Save the clean CSV
    DATA_S1.mkdir(parents=True, exist_ok=True)
    start_date, end_date = extract_dates_from_csv(df)
    out_name = build_output_name(NODE_ID, "QC_RAW", start_date, end_date)
    out_csv = DATA_S1 / out_name
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv.name}")

    # Save the statistics and the Markdown report
    stats_df.to_csv(result_dir / "stats_detalladas.csv", index=False)

    lines = [
        f"# Cleaning report: {NODE_ID} RAW (s1)\n",
        f"**Run at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Input:** {input_csv.name}  ",
        f"**Output:** {out_csv.name}  ",
        f"**Rows:** {n_total:,}\n",
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
    run_s1()
