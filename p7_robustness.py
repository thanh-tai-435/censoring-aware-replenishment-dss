"""Robustness of the semi-synthetic result: vary how the truth is built and how often the retailer stocks out.
Same random 10K series for every scenario (incl. the base), fewer quantiles, single-period (L=1)."""
import numpy as np, pandas as pd
from config import OUT, SEED
from p6_semisynthetic import load, build_truth, experiment

SCENARIOS = [("global", 1.2), ("noise", 1.2), ("evening", 1.2), ("global", 1.1), ("global", 1.4), ("evening", 1.4)]
Q, CU_CO = [0.5, 0.8, 0.95], [(1, 1), (4, 1), (19, 1)]

d, H, S = load(10_000)
rows = []
for mode, k in SCENARIOS:
    true_h = build_truth(H, S, mode, np.random.default_rng(SEED))
    r = experiment(d, H, true_h, k, quantiles=Q, cu_co=CU_CO, Ls=(1,), tag=f"_rob_{mode}_{k}")
    rec, acc, res = r["rec"].set_index(["estimate", "days"]), r["acc"].set_index("model"), r["res"]
    row = dict(truth=mode, k=k, **{n: round(v, 3) for n, v in r["world"].items()},
               M1_bias_cens_days=rec.loc[("M1 recovered", "censored days"), "bias"],
               sales_bias_cens_days=rec.loc[("censored sales", "censored days"), "bias"])
    for mdl in ["cens", "rec", "oracle"]:
        row[f"cover0.8_{mdl}"] = acc.loc[mdl, "cover_0.8"]
    for (L, tau), g in r["gap"].iterrows():
        row[f"cost_chg_tau{tau}"] = g.rec_vs_cens_cost
        row[f"gap_closed_tau{tau}"] = g.gap_closed_to_oracle
    b = res[res.tau == 0.8].set_index("policy")
    for p, lab in [("censored quantile (no recovery)", "cens"), ("recovered quantile (proposed)", "rec")]:
        row[f"fill_{lab}"], row[f"waste_{lab}"] = b.loc[p, "fill_rate"], b.loc[p, "waste_rate"]
    rows.append(row)
    print(pd.Series(row).to_string(), "\n", flush=True)

tab = pd.DataFrame(rows)
tab.round(4).to_csv(f"{OUT}/robustness.csv", index=False)
pd.set_option("display.width", 250)
print(tab.set_index(["truth", "k"]).T.round(3).to_string())
