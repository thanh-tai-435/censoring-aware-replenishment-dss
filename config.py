# Layer 0: the only place cost/policy assumptions live. Every sensitivity run = edit here.
import os
DATA = os.environ.get("FRN_DATA", "data")   # input folder (train/eval parquet); p12 points it at LT windows
OUT = os.environ.get("FRN_OUT", "out")
W = list(range(6, 22))            # business hours covered by stock_hour6_22_cnt
SHRINK_K = 5.0                    # series hourly profile shrunk toward store profile (pseudo-units); tuned in p2
MIN_P_IN = 0.2                    # below this in-stock profile mass, scaling is unreliable -> fallback
QUANTILES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
CU_CO = [(1, 1), (3, 2), (7, 3), (4, 1), (9, 1), (19, 1)]   # (cu, co) grid -> critical fractiles
BASE_CU_CO = (4, 1)               # base case: lost margin + stockout penalty 4x spoilage cost
RECOVERY = True                   # ablation flag (p3/p4 run both anyway)
SEED = 0
