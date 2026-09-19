"""
Settings of Node 5 (King Edward Memorial Park)

Constants that differ between the two nodes: reference station
(TH7), partner node of the transfer experiment (Node 3) and the
calibration sheet of the three Alphasense cells. Imported by config.py.
"""

# -------------------------------------------------------------------------
# Node, partner node and reference station
# -------------------------------------------------------------------------
NODE_ID = "node_5"
PARTNER_NODE_ID = "node_3"

REF_STATION_LABEL = "TH7_KEMP"
REF_STATION_SHORT = "TH7"
REF_CSV_PATTERN = "node_TH7_KEMP_wide_data_*.csv"
REF_CSV_NAME = "node_TH7_KEMP_wide_data_2025-04-16_to_2026-02-15.csv"

# -------------------------------------------------------------------------
# Alphasense calibration sheet of the three cells fitted to this node
# -------------------------------------------------------------------------
ALPHASENSE_CALIBRATION = {
    "co":  {"gain": 0.8,  "ez_we": 342, "ez_aux": 341, "sensitivity": 0.449,   "no2_sens": 0},
    "no2": {"gain": 0.8,  "ez_we": 247, "ez_aux": 245, "sensitivity": -0.303,  "no2_sens": 0},
    "o3":  {"gain": 0.8,  "ez_we": 231, "ez_aux": 227, "sensitivity": -0.283,  "no2_sens": -584.12},
}
