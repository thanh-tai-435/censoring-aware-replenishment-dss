"""Replication on FreshRetailNet-LT (770 days, 2023-06 -> 2025-07, series of varying length).
The pipeline assumes 97 contiguous days per series, so LT is cut into 97-day windows (90 train + 7 eval) holding
every series complete in the window; each window runs p1 -> p2 -> p11 with its own data/output folders.
R  = official eval week (newest data); S1-S4 = seasonal windows (holidays: National Day, Spring Festival, ...)."""
import os, subprocess, sys
import pandas as pd

LT = "data/lt"
WINDOWS = {"R_2025_official": None, "S1_autumn_2023": "2023-09-15", "S2_winter_2024": "2023-12-20",
           "S3_spring_2024": "2024-04-01", "S4_summer_2024": "2024-06-25"}
KEYS = ["store_id", "product_id"]

missing = [n for n in WINDOWS if not os.path.exists(f"data/lt_win/{n}/train.parquet")]
if missing:   # the full LT table needs several GB; load it only to cut windows that do not exist yet
    tr = pd.read_parquet(f"{LT}/train.parquet")
    tr["dt_"] = pd.to_datetime(tr.dt)
    ev = pd.read_parquet(f"{LT}/eval.parquet")
for name, start in WINDOWS.items():
    folder = f"data/lt_win/{name}"
    if name in missing:
        if start is None:                                   # official split: last 90 train days + the eval week
            s = pd.Timestamp("2025-07-13") - pd.Timedelta(days=89)
            w_tr, w_ev = tr[tr.dt_ >= s], ev
        else:
            s = pd.Timestamp(start)
            w = tr[(tr.dt_ >= s) & (tr.dt_ < s + pd.Timedelta(days=97))]
            cut = s + pd.Timedelta(days=90)
            w_tr, w_ev = w[w.dt_ < cut], w[w.dt_ >= cut]
        ok = (w_tr.groupby(KEYS).size() == 90) & (w_ev.groupby(KEYS).size() == 7)
        ok = ok[ok].index
        keep = lambda x: x.set_index(KEYS).loc[lambda y: y.index.isin(ok)].reset_index()
        os.makedirs(folder, exist_ok=True)
        keep(w_tr).drop(columns="dt_", errors="ignore").to_parquet(f"{folder}/train.parquet")
        keep(w_ev).drop(columns="dt_", errors="ignore").to_parquet(f"{folder}/eval.parquet")
        print(name, "series", len(ok), flush=True)
if missing:
    del tr, ev

rows = []
for name in WINDOWS:
    out = f"out_lt/{name}"
    env = dict(os.environ, FRN_DATA=f"data/lt_win/{name}", FRN_OUT=out)
    if not os.path.exists(f"{out}/confidence_summary.csv"):
        for script in ["p1_audit.py", "p2_recovery.py", "p11_confidence.py"]:
            subprocess.run([sys.executable, "-W", "ignore", script], env=env, check=True,
                           stdout=open(f"{out}_{script}.log", "w") if os.path.isdir("out_lt") else None)
            print(name, script, "done", flush=True)
    s = pd.read_csv(f"{out}/confidence_summary.csv").assign(window=name)
    rv = pd.read_csv(f"{out}/recovery_validation.csv").set_index(["method", "severity"])
    d = pd.read_parquet(f"{out}/daily.parquet", columns=["censored"])
    s["series"] = len(d) // 97
    s["stockout_day_share"] = d.censored.mean()
    s["transplant_bias_naive"] = rv.loc[("naive (observed sales)", "all"), "bias"]
    s["transplant_bias_M1"] = rv.loc[("series profile k=5", "all"), "bias"]
    s["transplant_WAPE_M1"] = rv.loc[("series profile k=5", "all"), "WAPE"]
    rows.append(s)
res = pd.concat(rows)
os.makedirs("out", exist_ok=True)
res.round(4).to_csv("out/lt_summary.csv", index=False)
pd.set_option("display.width", 250)
print(res[res.tau == 0.8][["window", "truth", "series", "stockout_day_share", "transplant_bias_naive", "transplant_bias_M1",
                           "M1_bias", "cost_change_mean", "cost_change_ci_lo", "cost_change_ci_hi", "gap_closed_mean",
                           "gap_closed_ci_lo", "gap_closed_ci_hi", "rec_better_share"]].round(3).to_string(index=False))
