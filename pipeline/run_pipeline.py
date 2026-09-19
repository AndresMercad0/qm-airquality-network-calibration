"""
Orchestrator of the calibration pipeline

Runs steps s0 to s9 in sequence for the node given with --node, from the
preparation of the public dataset to the machine learning calibration,
and prints the run time of each step. --skip and --from leave steps out;
a skipped step from s0 to s4 is replaced by its CSV from an earlier run.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


# -------------------------------------------------------------------------
# Pipeline steps, run in this order
# -------------------------------------------------------------------------
STEPS = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9"]


def parse_args():
    parser = argparse.ArgumentParser(description="Calibration pipeline of one node (s0 to s9)")
    parser.add_argument("--node", required=True, choices=["3", "5"],
                        help="Node to process")
    parser.add_argument("--skip", nargs="+", default=[], metavar="sN",
                        help="Steps to skip (for example: --skip s0)")
    parser.add_argument("--from", dest="from_step", default=None, metavar="sN",
                        help="Run again from this step (for example: --from s4)")
    return parser.parse_args()


# -------------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------------
def run():
    """
    Run the steps in sequence and print the time taken by each one.
    When a step from s0 to s4 is skipped, its CSV from a previous run is used.
    """
    args = parse_args()

    from config import NODE_ID, DATA_RAW, DATA_S1, DATA_S2, DATA_S3, DATA_S4
    from utils import detect_csv

    skip_set = set(args.skip)
    from_step = args.from_step

    if from_step and from_step in STEPS:
        from_idx = STEPS.index(from_step)
        skip_set.update(STEPS[:from_idx])

    print("=" * 70)
    print(f"  CALIBRATION PIPELINE: {NODE_ID}")
    print("=" * 70)
    print(f"  Steps: {' -> '.join(STEPS)}")
    if skip_set:
        print(f"  Skipping: {', '.join(sorted(skip_set))}")
    print()

    tiempos = {}
    csv_s0 = csv_s1 = csv_s2 = csv_s3 = csv_s4 = csv_s5 = csv_s6 = csv_s8 = None
    csv_s9 = None

    # s0: input preparation
    if "s0" not in skip_set:
        t0 = time.time()
        from s0_prepare_inputs import run_s0
        csv_s0 = run_s0()
        tiempos["s0"] = time.time() - t0
        print()
    else:
        csv_s0 = detect_csv(DATA_RAW, f"{NODE_ID}_RAW_wide_data_*.csv")
        if csv_s0:
            print(f"  [skip s0] Using: {csv_s0.name}")

    # s1: quality control of the low-cost sensors
    if "s1" not in skip_set:
        t0 = time.time()
        from s1_qc_lowcost import run_s1
        csv_s1 = run_s1(csv_s0)
        tiempos["s1"] = time.time() - t0
        print()
    else:
        csv_s1 = detect_csv(DATA_S1, f"{NODE_ID}_QC_RAW_wide_data_*.csv")
        if csv_s1:
            print(f"  [skip s1] Using: {csv_s1.name}")

    # s2: Alphasense conversion
    if "s2" not in skip_set:
        t0 = time.time()
        from s2_convert_alphasense import run_s2
        csv_s2 = run_s2(csv_s1)
        tiempos["s2"] = time.time() - t0
        print()
    else:
        csv_s2 = detect_csv(DATA_S2, f"{NODE_ID}_ALPHASENSE_UG_M3_wide_data_*.csv")
        if csv_s2:
            print(f"  [skip s2] Using: {csv_s2.name}")

    # s3: quality control of the Alphasense cells
    if "s3" not in skip_set:
        t0 = time.time()
        from s3_qc_alphasense import run_s3
        csv_s3 = run_s3(csv_s2)
        tiempos["s3"] = time.time() - t0
        print()
    else:
        csv_s3 = detect_csv(DATA_S3, f"{NODE_ID}_QC_ALPHASENSE_UG_M3_wide_data_*.csv")
        if csv_s3:
            print(f"  [skip s3] Using: {csv_s3.name}")

    # s4: preprocessing (raw signal to ratio); reads the s1 output, not s2 or s3
    if "s4" not in skip_set:
        t0 = time.time()
        from s4_preprocess import run_s4
        csv_s4 = run_s4(csv_s1)
        tiempos["s4"] = time.time() - t0
        print()
    else:
        csv_s4 = detect_csv(DATA_S4, f"{NODE_ID}_PREPROCESS_wide_data_*.csv")
        if csv_s4:
            print(f"  [skip s4] Using: {csv_s4.name}")

    # s5: manufacturer curves
    if "s5" not in skip_set:
        t0 = time.time()
        from s5_calibrate_v1v2 import run_s5
        csv_s5 = run_s5(csv_s4, csv_s3)
        tiempos["s5"] = time.time() - t0
        print()

    # s6: OLS
    if "s6" not in skip_set:
        t0 = time.time()
        from s6_calibrate_v3 import run_s6
        csv_s6 = run_s6(csv_s4, csv_s3)
        tiempos["s6"] = time.time() - t0
        print()

    # s7: comparative evaluation by cost tier
    if "s7" not in skip_set:
        t0 = time.time()
        from s7_evaluate import run_s7
        run_s7()
        tiempos["s7"] = time.time() - t0
        print()

    # s8: final assembly
    if "s8" not in skip_set:
        t0 = time.time()
        from s8_assemble_final import run_s8
        csv_s8_fab, csv_s8_ols = run_s8(csv_s3, csv_s5, csv_s6)
        tiempos["s8"] = time.time() - t0
        print()

    # s9: machine learning calibration
    if "s9" not in skip_set:
        t0 = time.time()
        from s9_calibrate_ml import run_s9
        csv_s9 = run_s9()
        tiempos["s9"] = time.time() - t0
        print()

    # Run times
    print("=" * 70)
    print("  RUN TIMES")
    print("=" * 70)
    total = 0
    for step in STEPS:
        if step in tiempos:
            t = tiempos[step]
            total += t
            print(f"  {step}: {t:.1f}s")
    print(f"  TOTAL: {total:.1f}s ({total/60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    run()
