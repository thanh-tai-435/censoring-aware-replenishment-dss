"""Second round of reviewer stress tests, each through the full decision pipeline (same forecaster, same simulator),
seed 0, the same random 10K series as p13, single-period, tau in {0.5, 0.8, 0.95}:
  - rolling origin: five 7-day evaluation windows (days 62-96), pooled with a two-way store x day bootstrap
  - richer synthetic retailer: order lead time, stock capacity, deliveries every other day, deferred demand,
    substitution within the store, and all of them at once
  - Kaplan-Meier recovery baseline (global and stress truth)
  - clustered intervals for the main run by store, city and category.
Writes out/review2_tests.csv (one row per scenario x tau) and out/review2_rolling.csv (pooled rolling origin)."""
import numpy as np, pandas as pd
from config import OUT
from p4_simulate import simulate
from p6_semisynthetic import load, build_truth, experiment, retailer_ext

Q, CU_CO, B = [0.5, 0.8, 0.95], [(1, 1), (4, 1), (19, 1)], 1000
ORIGINS = [62, 69, 76, 83, 90]

d, H, S = load(10_000, 0)
first = d.iloc[::97]
rng = np.random.default_rng(0)


def cluster_weights(labels):
    """(B, n_series) weights of a bootstrap that resamples whole clusters with replacement."""
    u, inv = np.unique(labels, return_inverse=True)
    return np.stack([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(B)])[:, inv]


W_STORE, W_CITY, W_CAT = (cluster_weights(first[c].values) for c in ["store_id", "city_id", "first_category_id"])

# substitution partner: another sampled series of the same store (none for stores with one sampled series)
partner, share = np.arange(len(first)), np.zeros(len(first))
for _, ix in first.reset_index(drop=True).groupby("store_id").groups.items():
    ix = np.array(ix)
    if len(ix) > 1:
        partner[ix], share[ix] = np.roll(rng.permutation(ix), 1), 1.0

true_global = build_truth(H, S, "global")
cap = np.percentile(true_global.sum(2), 75, axis=1)             # shelf sized to a 75th-percentile day


def costs(out, tau):
    """Per-series, per-day expected cost (cu + co = 1) of the three planners, shape (n_series, 7)."""
    D = out.demand_true.values
    return {c: (lambda s: tau * s["lost"] + (1 - tau) * s["waste"])(simulate(out[f"{c}_q{tau}"].values, D, 1))
            for c in ["cens", "rec", "oracle"]}


def summary(C, wts):
    cs = {k: v.sum(1) for k, v in C.items()}
    chg = lambda w: (w * cs["rec"]).sum(-1) / (w * cs["cens"]).sum(-1) - 1
    gap = lambda w: (w * (cs["cens"] - cs["rec"])).sum(-1) / (w * (cs["cens"] - cs["oracle"])).sum(-1)
    r = dict(cost_change=chg(np.ones(len(cs["rec"]))), gap_closed=gap(np.ones(len(cs["rec"]))))
    for nm, w in wts.items():
        r[f"{nm}_ci_lo"], r[f"{nm}_ci_hi"] = np.percentile(chg(w), [2.5, 97.5])
        r[f"gap_{nm}_ci_lo"], r[f"gap_{nm}_ci_hi"] = np.percentile(gap(w), [2.5, 97.5])
    per_day = C["rec"].sum(0) / C["cens"].sum(0) - 1
    return dict(r, day_change_min=per_day.min(), day_change_max=per_day.max())


SCEN = [  # name, truth, recovery, retailer_fn, eval_start
    *[(f"rolling_{o}", "global", "m1", None, o) for o in ORIGINS],
    ("lead_time_2d", "global", "m1", lambda th: retailer_ext(th, 1.2, lag=2), None),
    ("capacity_p75", "global", "m1", lambda th: retailer_ext(th, 1.2, cap=cap), None),
    ("delivery_every_2d", "global", "m1", lambda th: retailer_ext(th, 1.2, every=2), None),
    ("deferred_30", "global", "m1", lambda th: retailer_ext(th, 1.2, defer=0.3), None),
    ("substitution_30", "global", "m1", lambda th: retailer_ext(th, 1.2, subst=0.3 * share, partner=partner), None),
    ("all_combined", "global", "m1", lambda th: retailer_ext(th, 1.2, lag=2, cap=cap, every=2, defer=0.3,
                                                             subst=0.3 * share, partner=partner), None),
    ("km_global", "global", "km", None, None),
    ("km_stress", "stress", "km", None, None)]

rows, roll = [], {tau: {c: [] for c in ["cens", "rec", "oracle"]} for tau in Q}
for name, truth, rec, fn, start in SCEN:
    th = true_global if truth == "global" else build_truth(H, S, truth, np.random.default_rng(0))
    r = experiment(d, H, th, 1.2, quantiles=Q, cu_co=CU_CO, Ls=(1,), tag=f"_rev2_{name}", seed=0,
                   recovery=rec, retailer_fn=fn, eval_start=start)
    recerr = r["rec"].set_index(["estimate", "days"])
    for tau in Q:
        C = costs(r["out"], tau)
        wts = dict(store=W_STORE) if start is None or start != 90 else dict(store=W_STORE, city=W_CITY, category=W_CAT)
        rows.append(dict(scenario=name, truth=truth, recovery=rec, eval_start=start or 90, tau=tau,
                         stockout_day_rate=r["world"]["stockout_day_rate"], observed_share=r["world"]["observed_share"],
                         rec_bias_cens_days=recerr.loc[("M1 recovered", "censored days"), "bias"],
                         rec_wape_cens_days=recerr.loc[("M1 recovered", "censored days"), "WAPE"], **summary(C, wts)))
        if start is not None:
            for c in C:
                roll[tau][c].append(C[c])
    print(pd.DataFrame(rows[-3:])[["scenario", "tau", "stockout_day_rate", "rec_bias_cens_days", "cost_change",
                                   "store_ci_lo", "store_ci_hi", "gap_closed"]].round(3).to_string(index=False), flush=True)
pd.DataFrame(rows).round(4).to_csv(f"{OUT}/review2_tests.csv", index=False)

# pooled rolling origin: 35 evaluation days; two-way bootstrap resampling stores and days independently
pooled = []
for tau in Q:
    C = {c: np.concatenate(v, 1) for c, v in roll[tau].items()}                 # (n_series, 35)
    nd = C["rec"].shape[1]
    wd = np.stack([np.bincount(rng.integers(0, nd, nd), minlength=nd) for _ in range(B)])
    two = lambda X: np.einsum("bi,ij,bj->b", W_STORE, X, wd)
    chg, gap = two(C["rec"]) / two(C["cens"]) - 1, two(C["cens"] - C["rec"]) / two(C["cens"] - C["oracle"])
    per_window = [C["rec"][:, i:i + 7].sum() / C["cens"][:, i:i + 7].sum() - 1 for i in range(0, nd, 7)]
    per_day = C["rec"].sum(0) / C["cens"].sum(0) - 1
    pooled.append(dict(tau=tau, days=nd, cost_change=C["rec"].sum() / C["cens"].sum() - 1,
                       twoway_ci_lo=np.percentile(chg, 2.5), twoway_ci_hi=np.percentile(chg, 97.5),
                       gap_closed=(C["cens"] - C["rec"]).sum() / (C["cens"] - C["oracle"]).sum(),
                       gap_twoway_ci_lo=np.percentile(gap, 2.5), gap_twoway_ci_hi=np.percentile(gap, 97.5),
                       window_change_min=min(per_window), window_change_max=max(per_window),
                       day_change_min=per_day.min(), day_change_max=per_day.max(), days_better=(per_day < 0).mean()))
pd.DataFrame(pooled).round(4).to_csv(f"{OUT}/review2_rolling.csv", index=False)
print(pd.DataFrame(pooled).round(3).to_string(index=False))
