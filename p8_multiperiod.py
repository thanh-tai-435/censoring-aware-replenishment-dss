"""Multi-period perishable decision (shelf life L = 1..3) over a long horizon, scored on known true demand.
The 7-day eval week is too short for L > 1 (inventory starts empty, little ages out), so the semi-synthetic world
is re-split: models train on days 28-59, then order day-ahead for days 60-96 (37 days); the first 7 are warm-up."""
import numpy as np, pandas as pd
from config import OUT, SEED, BASE_CU_CO
from p6_semisynthetic import load, build_truth, experiment

d, H, S = load(10_000)
d["split"] = np.where(d.groupby(["store_id", "product_id"]).cumcount() < 60, "train", "eval")
pd.set_option("display.width", 220)
res_all, gap_all = [], []
for mode in ["global", "evening"]:
    true_h = build_truth(H, S, mode, np.random.default_rng(SEED))
    r = experiment(d, H, true_h, 1.2, Ls=(1, 2, 3), tag=f"_mp_{mode}", warmup=7)
    res_all.append(r["res"].assign(truth=mode)); gap_all.append(r["gap"].assign(truth=mode))
    print(f"[{mode}] world {r['world']}", flush=True)
res, gap = pd.concat(res_all), pd.concat(gap_all)
res.to_csv(f"{OUT}/multiperiod_kpi.csv", index=False)
gap.round(4).to_csv(f"{OUT}/multiperiod_headline.csv")

cu, co = BASE_CU_CO
print(f"\n=== 30 scored days, cu:co={cu}:{co} ===")
print(res[(res.cu == cu) & (res.co == co)].drop(columns=["cu", "co", "tau"]).round(4).to_string(index=False))
print("\n=== recovered vs censored cost change / share of gap to oracle closed ===")
print(gap.reset_index().pivot_table(index=["L", "tau"], columns="truth", values=["rec_vs_cens_cost", "gap_closed_to_oracle"]).round(3).to_string())
