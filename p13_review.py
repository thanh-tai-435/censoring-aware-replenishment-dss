"""Reviewer stress tests, each run through the full decision pipeline (same forecaster, same simulator), seed 0,
the same random 10K series, single-period, tau in {0.5, 0.8, 0.95}:
  - stress truth: demand shocks + drifting intraday shape, filled hours independent of the same day's rate
  - recovery baselines: daily Tobit EM (no hourly labels) and flat-profile hourly scaling
  - alternative retailer policies: forecast + safety stock, adaptive multiplier
  - uncertainty: series bootstrap AND store-cluster bootstrap (series in a store are dependent), and
    day-to-day variation of the cost change over the 7 evaluation days."""
import numpy as np, pandas as pd
from config import OUT
from p4_simulate import simulate
from p6_semisynthetic import load, build_truth, experiment

Q, CU_CO, B = [0.5, 0.8, 0.95], [(1, 1), (4, 1), (19, 1)], 1000
SCEN = [("global", "mean", "m1"), ("donor", "mean", "m1"), ("stress", "mean", "m1"),
        ("global", "mean", "tobit"), ("global", "mean", "uniform"),
        ("stress", "mean", "tobit"), ("stress", "mean", "uniform"),
        ("global", "safety", "m1"), ("global", "adaptive", "m1")]

d, H, S = load(10_000, 0)
store = d.store_id.values[::97]
stores, inv = np.unique(store, return_inverse=True)
rng = np.random.default_rng(0)
idx_series = rng.integers(0, len(store), (B, len(store)))
# store-cluster bootstrap: draw stores with replacement; a series enters as often as its store is drawn
draw_counts = np.stack([np.bincount(rng.integers(0, len(stores), len(stores)), minlength=len(stores)) for _ in range(B)])
wt_cluster = draw_counts[:, inv]                                       # (B, n_series) weights

rows = []
for truth, pol, rec in SCEN:
    r = experiment(d, H, build_truth(H, S, truth, np.random.default_rng(0)), 1.2, quantiles=Q, cu_co=CU_CO,
                   Ls=(1,), tag=f"_rev_{truth}_{pol}_{rec}", seed=0, retailer_policy=pol, recovery=rec)
    out, recerr = r["out"], r["rec"].set_index(["estimate", "days"])
    D = out.demand_true.values
    for tau in Q:
        C = {}
        for col in ["cens", "rec", "oracle"]:
            sim = simulate(out[f"{col}_q{tau}"].values, D, 1)
            C[col] = tau * sim["lost"] + (1 - tau) * sim["waste"]          # (n_series, 7 days)
        cs = {k: v.sum(1) for k, v in C.items()}                         # per-series cost
        chg = lambda w: (w * cs["rec"]).sum(-1) / (w * cs["cens"]).sum(-1) - 1
        gap = lambda w: (w * (cs["cens"] - cs["rec"])).sum(-1) / (w * (cs["cens"] - cs["oracle"])).sum(-1)
        ws = np.stack([np.bincount(i, minlength=len(store)) for i in idx_series])
        per_day = C["rec"].sum(0) / C["cens"].sum(0) - 1                  # 7 values
        rows.append(dict(
            truth=truth, retailer=pol, recovery=rec, tau=tau,
            stockout_day_rate=r["world"]["stockout_day_rate"],
            rec_bias_cens_days=recerr.loc[("M1 recovered", "censored days"), "bias"],
            rec_wape_cens_days=recerr.loc[("M1 recovered", "censored days"), "WAPE"],
            cost_change=cs["rec"].sum() / cs["cens"].sum() - 1,
            series_ci_lo=np.percentile(chg(ws), 2.5), series_ci_hi=np.percentile(chg(ws), 97.5),
            cluster_ci_lo=np.percentile(chg(wt_cluster), 2.5), cluster_ci_hi=np.percentile(chg(wt_cluster), 97.5),
            gap_closed=(cs["cens"] - cs["rec"]).sum() / (cs["cens"] - cs["oracle"]).sum(),
            gap_cluster_ci_lo=np.percentile(gap(wt_cluster), 2.5), gap_cluster_ci_hi=np.percentile(gap(wt_cluster), 97.5),
            day_change_min=per_day.min(), day_change_max=per_day.max()))
    print(pd.DataFrame(rows[-3:])[["truth", "retailer", "recovery", "tau", "stockout_day_rate", "rec_bias_cens_days",
                                   "cost_change", "cluster_ci_lo", "cluster_ci_hi", "gap_closed",
                                   "day_change_min", "day_change_max"]].round(3).to_string(index=False), flush=True)

res = pd.DataFrame(rows)
res.round(4).to_csv(f"{OUT}/review_tests.csv", index=False)
print(f"\nstores in sample: {len(stores)}; series: {len(store)}")
