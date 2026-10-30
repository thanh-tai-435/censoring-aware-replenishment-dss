"""Phase 4-5 / Layers 4-5 (M3 + simulator): critical-fractile orders, inventory simulation, KPIs,
baselines, ablations, cu:co + shelf-life sensitivity, per-city replication, frontier plot.
Truth A = uncensored eval days, D = observed sales (exact, conservative: low-demand days).
Truth B = all eval days, D = recovered demand (estimate; validated in p2)."""
import numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import OUT, QUANTILES, CU_CO, BASE_CU_CO

T = 7                                            # eval days per series; rows sorted by (series, day)


def order(policy, tau):
    """Layer 4: order-up-to level per row for a policy at critical fractile tau."""
    z = norm.ppf(tau)
    return {
        "P1 naive: ma7 of sales": f.sales_ma7,
        "P1b sales ma7 + safety stock": f.sales_ma7 + z * f.sales_sd7,
        "P2 sales quantile (no recovery)": f[f"sales_q{tau}"],
        "P3a demand point (q0.5)": f["demand_q0.5"],
        "P3b demand ma7 + safety stock": f.demand_ma7 + z * f.demand_sd7,
        "P3 demand quantile (proposed)": f[f"demand_q{tau}"],
    }[policy].fillna(0).clip(lower=0).values


POLICIES = ["P1 naive: ma7 of sales", "P1b sales ma7 + safety stock", "P2 sales quantile (no recovery)",
            "P3a demand point (q0.5)", "P3b demand ma7 + safety stock", "P3 demand quantile (proposed)"]


def simulate(S_target, D, L, T=T):
    """Base-stock with FIFO aging: each day top up to S_target, sell oldest first, units older than L days spoil.
    L=1 is the single-period perishable newsvendor. Returns ordered, sold, waste, lost arrays (N, T)."""
    S_target, D = S_target.reshape(-1, T), D.reshape(-1, T).astype(np.float64)   # float32 truth breaks unit balance
    inv = np.zeros((len(D), L))                       # inv[:, a] = units of age a (0 = fresh today)
    out = {k: np.zeros(D.shape) for k in ["ordered", "sold", "waste", "lost"]}
    for t in range(T):
        q = np.maximum(S_target[:, t] - inv.sum(1), 0)
        inv[:, 0] += q
        need = D[:, t].copy()
        for a in range(L - 1, -1, -1):           # oldest first
            s = np.minimum(inv[:, a], need); inv[:, a] -= s; need -= s
        out["ordered"][:, t], out["sold"][:, t], out["lost"][:, t] = q, D[:, t] - need, need
        out["waste"][:, t] = inv[:, L - 1]       # oldest bucket expires tonight
        inv = np.concatenate([np.zeros((len(D), 1)), inv[:, :L - 1]], 1)
    # unit conservation: everything ordered is sold, spoiled, or still on hand
    assert np.allclose(out["ordered"].sum(1), out["sold"].sum(1) + out["waste"].sum(1) + inv.sum(1))
    assert np.allclose(out["sold"] + out["lost"], D)
    return out


def kpis(sim, D, mask, cu, co, T=T):
    m = mask.reshape(-1, T)
    s = {k: v[m].sum() for k, v in sim.items()}
    Dsum = D.reshape(-1, T)[m].sum()
    return dict(cost_per_unit_demand=(cu * s["lost"] + co * s["waste"]) / Dsum,
                waste_rate=s["waste"] / max(s["ordered"], 1e-9), fill_rate=s["sold"] / Dsum,
                order_bias=(s["ordered"] - Dsum) / Dsum)


if __name__ == "__main__":
    f = pd.read_parquet(f"{OUT}/forecasts.parquet").sort_values(["store_id", "product_id", "dt"])
    TRUTH = {"A": (f.sale_amount.values, (f.censored == 0).values), "B": (f.demand.values, np.ones(len(f), bool))}
    rows = []
    for truth, (D, mask) in TRUTH.items():
        for L in ([1] if truth == "A" else [1, 2, 3]):    # A has gaps in time -> single-period only
            for cu, co in CU_CO:
                tau = round(cu / (cu + co), 2)
                for p in POLICIES:
                    sim = simulate(order(p, tau), D, L)
                    rows.append(dict(truth=truth, L=L, cu=cu, co=co, tau=tau, policy=p, **kpis(sim, D, mask, cu, co)))
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/kpi_all.csv", index=False)

    cu, co = BASE_CU_CO
    base = res[(res.cu == cu) & (res.co == co)]
    pd.set_option("display.width", 200)
    print(f"=== KPI table, base cu:co = {cu}:{co} (tau={cu/(cu+co):.2f}) ===")
    print(base.drop(columns=["cu", "co", "tau"]).round(4).to_string(index=False))

    # headline: proposed vs no-recovery (same order rule) across cost ratios
    piv = res.pivot_table(index=["truth", "L", "tau"], columns="policy", values=["cost_per_unit_demand", "fill_rate", "waste_rate"])
    hl = pd.DataFrame({
        "cost_change_P3_vs_P2": piv["cost_per_unit_demand"]["P3 demand quantile (proposed)"] / piv["cost_per_unit_demand"]["P2 sales quantile (no recovery)"] - 1,
        "cost_change_P3_vs_P1": piv["cost_per_unit_demand"]["P3 demand quantile (proposed)"] / piv["cost_per_unit_demand"]["P1 naive: ma7 of sales"] - 1,
        "fill_P2": piv["fill_rate"]["P2 sales quantile (no recovery)"], "fill_P3": piv["fill_rate"]["P3 demand quantile (proposed)"],
        "waste_P2": piv["waste_rate"]["P2 sales quantile (no recovery)"], "waste_P3": piv["waste_rate"]["P3 demand quantile (proposed)"]})
    hl.round(4).to_csv(f"{OUT}/headline.csv")
    print("\n=== P3 (recovered) vs P2 (observed), same critical-fractile rule ===")
    print(hl.round(4).to_string())

    # per-city replication at base cost, L=1
    city = []
    for truth, (D, mask) in TRUTH.items():
        tau = round(cu / (cu + co), 2)
        for p in ["P2 sales quantile (no recovery)", "P3 demand quantile (proposed)"]:
            sim = simulate(order(p, tau), D, 1)
            for c in np.unique(f.city_id):
                cm = mask & (f.city_id.values == c)
                city.append(dict(truth=truth, city=c, policy=p[:2], **kpis(sim, D, cm, cu, co)))
    city = pd.DataFrame(city).pivot_table(index=["truth", "city"], columns="policy", values=["cost_per_unit_demand", "fill_rate"])
    city[("cost_change", "")] = city[("cost_per_unit_demand", "P3")] / city[("cost_per_unit_demand", "P2")] - 1
    city.round(4).to_csv(f"{OUT}/city_replication.csv")
    print("\n=== per-city cost change P3 vs P2 (base cost, L=1) ===")
    print(city["cost_change"].unstack(0).round(3).describe().round(3).to_string())

    # frontier: cost and waste vs fill rate across tau, L=1
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for j, truth in enumerate(["A", "B"]):
        r = res[(res.truth == truth) & (res.L == 1)]
        for p in POLICIES:
            s = r[r.policy == p].sort_values("tau")
            ax[0, j].plot(s.fill_rate, s.waste_rate, "o-", label=p, ms=3)
            ax[1, j].plot(s.tau, s.cost_per_unit_demand, "o-", label=p, ms=3)
        ax[0, j].set(title=f"Truth {truth}: waste vs service (L=1)", xlabel="fill rate", ylabel="waste rate")
        ax[1, j].set(title=f"Truth {truth}: cost vs critical fractile", xlabel="tau = cu/(cu+co)", ylabel="cost per unit demand")
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{OUT}/frontier.png", dpi=130)
