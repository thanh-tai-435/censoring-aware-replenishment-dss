"""Baseline 4: the dataset authors' published forecaster (SSA, similar-scenario average) from frn-50k-baseline
(Apache-2.0, demand_forecasting/SSA/ssa_forecasting.py), ported 1:1 to numpy. Their TimesNet recovery needs a GPU
and is not rerun; its reported numbers are cited from the technical report instead.
A) real data: SSA on observed sales vs on M1-recovered demand, under the authors' accuracy protocol and our KPIs.
B) semi-synthetic world (known truth): SSA point orders vs the LightGBM quantile planners."""
import numpy as np, pandas as pd
from config import OUT, CU_CO
from p4_simulate import simulate, kpis
from p6_semisynthetic import load, build_truth, censored_world

ENC = 90


def ssa(d, col):
    """Authors' SSA: day-ahead-free 7-day forecast = softmax-weighted average of the 90 history days, weighted by
    recency, weekday/holiday and rain similarity, and discount closeness, times a cross-series weekday ratio."""
    a = d[["dt", "holiday_flag", "precpt", "discount", col]].copy()
    dt = pd.to_datetime(a.dt)
    a["time_idx"] = (dt - dt.min()).dt.days
    a["dow"] = dt.dt.day_of_week
    a.loc[a.dt.astype(str).isin(["2024-05-02", "2024-05-03", "2024-05-04"]), "holiday_flag"] = 0
    a.loc[a.holiday_flag == 1, "dow"] = 5
    a.loc[(a.holiday_flag == 0) & (a.dow.isin([5, 6])), "dow"] = 0
    a["is_precpt"] = (a.precpt > 3.5).astype(int)
    xc = a[["time_idx", "discount", col]].values.astype(float).reshape(-1, 97, 3)
    xk = a[["dow", "holiday_flag", "is_precpt"]].values.reshape(-1, 97, 3)
    HOL = 5
    dist = 18 - xc[:, :ENC, 0][:, None, :] / 5
    disc = 1 - np.abs(xc[:, ENC:, 1][..., None] - xc[:, :ENC, 1][:, None, :])
    ddow, edow = xk[:, ENC:, 0][..., None], xk[:, :ENC, 0][:, None, :]
    dpre, epre = xk[:, ENC:, 2][..., None], xk[:, :ENC, 2][:, None, :]
    dist = np.where((ddow == HOL).astype(int) - (edow == HOL).astype(int) == 0, dist - 3, dist)
    dist = np.where((ddow != HOL) & (edow == HOL), dist + 3, dist)
    dist = np.where(ddow - edow == 0, dist - 3, dist)
    dist = np.where((dpre - epre == 0) & (dpre == 1), dist - 3, dist)
    dist = np.where(dpre - epre != 0, dist + 3, dist)
    dist = np.clip(dist, 1, 18)
    w = (18 - dist) * disc ** 2
    w = np.exp(w) / np.exp(w).sum(-1, keepdims=True)
    y = xc[:, :ENC, 2:3]
    idx = edow.squeeze(1)[..., None] == np.arange(6)
    mean_sale = np.nanmean(np.where(idx, y, np.nan), axis=1) + 0.001
    season = np.nanmedian(mean_sale[..., None] / mean_sale[:, None, :], axis=0)
    ratio = season[ddow, edow]
    return np.maximum(((w * ratio) @ y).reshape(-1), 0)


def frn_metric(df, pred, truth="sale_amount"):
    """Authors' protocol: uncensored eval days only, WAPE and WPE (bias) computed per date then averaged."""
    m = df.stock_hour6_22_cnt == 0
    g = pd.DataFrame({"dt": df.dt[m], "y": df[truth][m], "p": np.asarray(pred)[m.values]}).groupby("dt")
    return dict(wape=g.apply(lambda s: (s.p - s.y).abs().sum() / s.y.sum()).mean(),
                wpe=g.apply(lambda s: (s.p - s.y).sum() / s.y.sum()).mean())


def kpi_rows(D, orders, cu_co=CU_CO, T=7):
    rows = []
    for cu, co in cu_co:
        tau = round(cu / (cu + co), 2)
        for name, fn in orders.items():
            sim = simulate(np.clip(np.nan_to_num(fn(tau)), 0, None), D, 1, T=T)
            rows.append(dict(tau=tau, policy=name, **kpis(sim, D, np.ones(len(D), bool), cu, co, T=T)))
    return pd.DataFrame(rows)


pd.set_option("display.width", 220)
# ---- A) real data
rd = pd.read_parquet(f"{OUT}/recovered.parquet")
ev = (rd.split == "eval").values
f = pd.read_parquet(f"{OUT}/forecasts.parquet")
assert (f.product_id.values == rd.product_id.values[ev]).all() and (f.dt.values == rd.dt.values[ev]).all()
f["ssa_sales"], f["ssa_demand"] = ssa(rd, "sale_amount"), ssa(rd, "demand")
f["stock_hour6_22_cnt"] = rd.stock_hour6_22_cnt.values[ev]
acc = pd.DataFrame([dict(forecast=n, **frn_metric(f, f[c])) for n, c in [
    ("SSA on observed sales (authors)", "ssa_sales"), ("SSA on M1-recovered demand", "ssa_demand"),
    ("LightGBM q0.5 on observed sales", "sales_q0.5"), ("LightGBM q0.5 on M1-recovered demand", "demand_q0.5")]])
acc.round(4).to_csv(f"{OUT}/baseline_frn_protocol.csv", index=False)
print("=== authors' protocol (uncensored eval days, per-date WAPE / WPE) ===")
print(acc.round(4).to_string(index=False))

orders = {"SSA on sales, point (authors)": lambda tau: f.ssa_sales.values,
          "SSA on recovered, point": lambda tau: f.ssa_demand.values,
          "P2 sales quantile (no recovery)": lambda tau: f[f"sales_q{tau}"].values,
          "P3 demand quantile (proposed)": lambda tau: f[f"demand_q{tau}"].values}
real = []
for truth, D, m in [("A", f.sale_amount.values, (f.censored == 0).values), ("B", f.demand.values, None)]:
    if m is None:
        real.append(kpi_rows(D, orders).assign(truth=truth))
    else:   # A: only uncensored days carry an exact truth -> score just those rows (single-period, rows independent)
        sub = {k: (lambda fn: (lambda tau: fn(tau)[m]))(fn) for k, fn in orders.items()}
        real.append(kpi_rows(D[m], sub, T=1).assign(truth=truth))

# ---- B) semi-synthetic world, same construction as p6 main (global truth, k=1.2)
d, H, S = load()
sd, _, _ = censored_world(d, H, build_truth(H, S, "global"), 1.2)
sev = (sd.split == "eval").values
sf = pd.read_parquet(f"{OUT}/semi_forecasts.parquet")
assert np.allclose(sf.sales_c.values, sd.sales_c.values[sev])
sf["ssa_cens"], sf["ssa_rec"], sf["ssa_true"] = ssa(sd, "sales_c"), ssa(sd, "demand_rec"), ssa(sd, "demand_true")
semi = kpi_rows(sf.demand_true.values, {
    "SSA on censored, point (authors)": lambda tau: sf.ssa_cens.values,
    "SSA on recovered, point": lambda tau: sf.ssa_rec.values,
    "SSA on true demand, point": lambda tau: sf.ssa_true.values,
    "censored quantile (no recovery)": lambda tau: sf[f"cens_q{tau}"].values,
    "recovered quantile (proposed)": lambda tau: sf[f"rec_q{tau}"].values,
    "oracle quantile (true-demand history)": lambda tau: sf[f"oracle_q{tau}"].values}).assign(truth="semi-synthetic")
semi_acc = pd.DataFrame([dict(forecast=c, bias=(sf[c] - sf.demand_true).sum() / sf.demand_true.sum(),
                              WAPE=(sf[c] - sf.demand_true).abs().sum() / sf.demand_true.sum())
                         for c in ["ssa_cens", "ssa_rec", "ssa_true", "cens_q0.5", "rec_q0.5", "oracle_q0.5"]])
print("\n=== semi-synthetic, point accuracy vs true demand ===")
print(semi_acc.round(4).to_string(index=False))

res = pd.concat(real + [semi])
res.to_csv(f"{OUT}/baseline_kpi.csv", index=False)
print("\n=== cost per unit demand by critical fractile (L=1) ===")
print(res.pivot_table(index=["truth", "policy"], columns="tau", values="cost_per_unit_demand").round(3).to_string())
print("\n=== fill / waste at tau=0.8 ===")
print(res[res.tau == 0.8][["truth", "policy", "fill_rate", "waste_rate"]].round(3).to_string(index=False))
