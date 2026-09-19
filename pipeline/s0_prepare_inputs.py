"""
Step s0: preparation of the inputs

Crops the node file of the public dataset (qm-airquality-network-data)
to the study window and writes it to data/node_N/raw/. Timestamps are
compared as text because they are ISO 8601 in UTC.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    NODE_ID, DATA_RAW, DATASET_DIR, RAW_CSV_NAME,
    STUDY_START_UTC, STUDY_STOP_UTC,
)

log = logging.getLogger("s0")


# -------------------------------------------------------------------------
# Crop the published node file to the study window
# -------------------------------------------------------------------------
def crop_node_file(src, dst):
    """
    Copy the rows of the study window from the dataset file to the raw folder.
    Timestamps are ISO 8601 in UTC, so they are compared as text.
    """
    kept = 0
    with open(src, "r", encoding="utf-8", newline="") as fin, \
            open(dst, "w", encoding="utf-8", newline="") as fout:
        fout.write(fin.readline())
        for line in fin:
            ts = line.split(",", 1)[0]
            if STUDY_START_UTC <= ts < STUDY_STOP_UTC:
                fout.write(line)
                kept += 1
    return kept


def run_s0():
    """Prepare the raw input of this node and return its path."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    src = DATASET_DIR / f"{NODE_ID}.csv"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found: clone qm-airquality-network-data (with Git LFS) "
            "next to this repository or set AQMS_DATASET_DIR"
        )

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    dst = DATA_RAW / RAW_CSV_NAME
    kept = crop_node_file(src, dst)
    log.info("%s: %d rows from %s to %s (exclusive)", dst.name, kept, STUDY_START_UTC, STUDY_STOP_UTC)
    return dst


if __name__ == "__main__":
    run_s0()
