"""
Settings of Node 3 (QMUL, Mile End Road)

Constants that differ between the two nodes: reference station
(TH2), partner node of the transfer experiment (Node 5) and the
calibration sheet of the three Alphasense cells. Imported by config.py.
"""

# -------------------------------------------------------------------------
# Node, partner node and reference station
# -------------------------------------------------------------------------
NODE_ID = "node_3"
PARTNER_NODE_ID = "node_5"

REF_STATION_LABEL = "TH2_MER"
REF_STATION_SHORT = "TH2"
REF_CSV_PATTERN = "node_TH2_MER_wide_data_*.csv"
REF_CSV_NAME = "node_TH2_MER_wide_data_2025-04-16_to_2026-02-15.csv"

# -------------------------------------------------------------------------
# Alphasense calibration sheet of the three cells fitted to this node
# -------------------------------------------------------------------------
ALPHASENSE_CALIBRATION = {
    "co":  {"gain": 0.8,   "ez_we": 354, "ez_aux": 346, "sensitivity": 0.412, "no2_sens": 0},
    "no2": {"gain": -0.73, "ez_we": 230, "ez_aux": 233, "sensitivity": 0.288, "no2_sens": 0},
    "o3":  {"gain": -0.73, "ez_we": 231, "ez_aux": 238, "sensitivity": 0.343, "no2_sens": -644},
}
