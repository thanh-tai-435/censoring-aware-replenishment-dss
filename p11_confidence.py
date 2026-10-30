"""Uncertainty for the headline numbers: 3 seeds (different random 10K-series subsets + LightGBM seeds) x
truth construction {global, donor}, each with a 95% series-bootstrap CI. Costs normalized so cu + co = 1
(cu = tau, co = 1 - tau), making cost levels comparable across critical fractiles. Single-period (L=1)."""
import numpy as np, pandas as pd
from config import OUT
from p4_simulate import simulate
from p6_semisynthetic import load, build_truth, experiment

Q = [0.5, 0.8, 0.95]
CU_CO = [(1, 1), (4, 1), (19, 1)]
B = 1000
POL = {"cens": "censored", "rec": "recovered", "oracle": "oracle"}


def series_cost(out, col, tau):
    """Per-series normalized cost and demand over the eval week (rows sorted by series, 7 days each)."""
    D = out.demand_true.values
    sim = simulate(out[f"{col}_q{tau}"].values, D, 1)
    return (tau * sim["lost"] + (1 - tau) * sim["waste"]).sum(1), D.reshape(-1, 7).sum(1)


rows = []
for seed in [0, 1, 2]:
    d, H, S = load(10_000, seed)
    for mode in ["global", "donor"]:
        r = experiment(d, H, build_truth(H, S, mode, np.random.default_rng(seed)), 1.2, quantiles=Q, cu_co=CU_CO,
                       Ls=(1,), tag=f"_ci_{mode}_{seed}", seed=seed)
        rec = r["rec"].set_index(["estimate", "days"])
        rng = np.random.default_rng(seed)
        n = len(r["out"]) // 7
        idx = rng.integers(0, n, (B, n))
        for tau in Q:
            C = {}
            for col in POL:
                C[col], Dn = series_cost(r["out"], col, tau)
            boot_chg = C["rec"][idx].sum(1) / C["cens"][idx].sum(1) - 1
            boot_gap = (C["cens"] - C["rec"])[idx].sum(1) / (C["cens"] - C["oracle"])[idx].sum(1)
            rows.append(dict(
                seed=seed, truth=mode, tau=tau,
                M1_bias_cens_days=rec.loc[("M1 recovered", "censored days"), "bias"],
                cost_cens=C["cens"].sum() / Dn.sum(), cost_rec=C["rec"].sum() / Dn.sum(), cost_oracle=C["oracle"].sum() / Dn.sum(),
                cost_change=C["rec"].sum() / C["cens"].sum() - 1,
                cost_change_lo=np.percentile(boot_chg, 2.5), cost_change_hi=np.percentile(boot_chg, 97.5),
                gap_closed=(C["cens"] - C["rec"]).sum() / (C["cens"] - C["oracle"]).sum(),
                gap_closed_lo=np.percentile(boot_gap, 2.5), gap_closed_hi=np.percentile(boot_gap, 97.5),
                rec_better_share=(C["rec"] < C["cens"]).mean()))     # share of series where recovery lowers cost
        print(pd.DataFrame(rows[-3:])[["seed", "truth", "tau", "cost_change", "cost_change_lo", "cost_change_hi",
                                       "gap_closed", "gap_closed_lo", "gap_closed_hi"]].round(3).to_string(index=False), flush=True)

res = pd.DataFrame(rows)
res.round(4).to_csv(f"{OUT}/confidence_runs.csv", index=False)
agg = res.groupby(["truth", "tau"]).agg(
    M1_bias=("M1_bias_cens_days", "mean"),
    cost_cens=("cost_cens", "mean"), cost_rec=("cost_rec", "mean"), cost_oracle=("cost_oracle", "mean"),
    cost_change_mean=("cost_change", "mean"), cost_change_sd=("cost_change", "std"),
    cost_change_ci_lo=("cost_change_lo", "min"), cost_change_ci_hi=("cost_change_hi", "max"),
    gap_closed_mean=("gap_closed", "mean"), gap_closed_sd=("gap_closed", "std"),
    gap_closed_ci_lo=("gap_closed_lo", "min"), gap_closed_ci_hi=("gap_closed_hi", "max"),
    rec_better_share=("rec_better_share", "mean"))
agg.round(4).to_csv(f"{OUT}/confidence_summary.csv")
pd.set_option("display.width", 250)
print("\n=== mean over 3 seeds; CI = widest 95% series-bootstrap bound across seeds ===")
print(agg.round(3).to_string())
