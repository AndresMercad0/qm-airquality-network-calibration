"""
Input table of the transfer figure

Keeps, from source_selected_transfer.csv, the direct transfer of the
validation-selected models scored on the destination test period, and
writes results/cross_node/best_transfer_source_selected_v6.csv.
"""

from pathlib import Path

import pandas as pd

# -------------------------------------------------------------------------
# Input and output tables
# -------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "cross_node"
TRANSFER_CSV = RESULTS_DIR / "source_selected_transfer.csv"
OUT_CSV = RESULTS_DIR / "best_transfer_source_selected_v6.csv"

SOURCE_LABEL = "locked_min_val_RMSE/source-selected/test"
GAS_ORDER = ["PM2.5", "CO", "NO2", "O3"]


# -------------------------------------------------------------------------
# Transfer of the validation-selected models, scored on the destination test period
# -------------------------------------------------------------------------
def main():
    """Write the table read by the cross-node transfer figure."""
    transfer = pd.read_csv(TRANSFER_CSV)
    sel = transfer[(transfer.selection_rule == "locked_min_val_RMSE")
                   & (transfer.scenario == "source-selected")
                   & (transfer.period == "test")]
    rows = pd.DataFrame({"direction": sel.direction, "gas": sel.gas, "model": sel.model, "config": sel.config,
                         "R2_local": sel.R2_local_test, "R2_transferred": sel.R2_transferred,
                         "delta_R2": sel.R2_transferred - sel.R2_local_test,
                         "source": SOURCE_LABEL})

    rows["gas"] = pd.Categorical(rows["gas"], categories=GAS_ORDER, ordered=True)
    rows = rows.sort_values(["direction", "gas"]).reset_index(drop=True)
    rows["gas"] = rows["gas"].astype(str)

    rows.to_csv(OUT_CSV, index=False)
    print(f"{OUT_CSV.name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
