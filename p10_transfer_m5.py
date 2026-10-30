"""Transfer run on M5 (Walmart, daily, no stockout labels, no hourly data): does the DESIGN carry over?
Layers M2 (forecaster), M3 (order rule) and the simulator are reused unchanged; only Layer 2 (M1) is swapped,
because the hourly-profile trick needs intraday data. M1-daily = censored-likelihood (Tobit-style) recovery:
per-series Poisson demand with weekday factors, fitted by EM on the censored likelihood, then
D_hat = E[D | D >= S] on stockout days.
Truth: high-volume FOODS series (mean >= 2/day, <= 10% zero days) over the last 97 days, taken as uncensored
demand; the same synthetic retailer as p6 (stock = k * mean observed last 7 days) censors it at daily level."""
import numpy as np, pandas as pd
from scipy.stats import poisson
from config import OUT, CU_CO, BASE_CU_CO, QUANTILES
from p3_forecast import forecast, pinball_table
from p4_simulate import simulate, kpis

T, ENC = 97, 90
s = pd.read_csv("data/m5/sales_train_evaluation.csv")
cal = pd.read_csv("data/m5/calendar.csv").set_index("d")
days = [f"d_{i}" for i in range(1942 - T, 1942)]
Y = s[days].values.astype(float)
keep = (s.cat_id == "FOODS").values & (Y.mean(1) >= 2) & ((Y == 0).mean(1) <= 0.1)
s, Y = s[keep].reset_index(drop=True), Y[keep]
N = len(s)
c = cal.loc[days]
dow = pd.to_datetime(c.date).dt.day_of_week.values


def retailer(D, k):
    obs, cens = np.zeros_like(D), np.zeros(D.shape, bool)
    for t in range(T):
        C = k * (D[:, :7].mean(1) if t < 7 else obs[:, t - 7:t].mean(1))
        obs[:, t], cens[:, t] = np.minimum(D[:, t], C), D[:, t] >= C
    return obs, cens


def m1_daily(S, cens, train, iters=30):
    """EM on the censored Poisson likelihood: lambda_it = a_i * w_dow(t). E-step on censored days uses
    E[D | D >= s] = lambda * P(D >= s-1) / P(D >= s)."""
    w = np.ones(7); D = S.copy()
    for _ in range(iters):
        a = (D * train).sum(1) / (w[dow] * train).sum(1)
        sel = [train[0] & (dow == j) for j in range(7)]
        w = np.array([D[:, sj].sum() / (a.sum() * sj.sum()) for sj in sel])
        w = w / w.mean()
        lam = a[:, None] * w[dow][None, :]
        s_ = np.ceil(S)
        tail = poisson.sf(s_ - 1, lam)                         # P(D >= s)
        ED = lam * poisson.sf(s_ - 2, lam) / np.maximum(tail, 1e-12)
        D = np.where(cens, np.maximum(S, ED), S)
    return D


train = np.zeros((N, T), bool); train[:, :ENC] = True
rows, results = [], {}
for k in [1.1, 1.2, 1.4]:
    obs, cens = retailer(Y, k)
    rec = m1_daily(obs, cens, train)
    m = cens
    rows.append(dict(k=k, stockout_day_rate=cens.mean(), observed_share=obs.sum() / Y.sum(),
                     sales_bias_cens_days=(obs[m] - Y[m]).sum() / Y[m].sum(),
                     M1daily_bias_cens_days=(rec[m] - Y[m]).sum() / Y[m].sum(),
                     M1daily_WAPE_cens_days=np.abs(rec[m] - Y[m]).sum() / Y[m].sum()))
    print(rows[-1], flush=True)

    # Layer 1 contract: the same flat daily table the FRN pipeline uses (column names mapped)
    d = pd.DataFrame({
        "store_id": np.repeat(s.store_id.astype("category").cat.codes.values, T),
        "product_id": np.repeat(np.arange(N), T),
        "dt": np.tile(c.date.values, N), "split": np.tile(np.where(np.arange(T) < ENC, "train", "eval"), N),
        "city_id": np.repeat(s.state_id.astype("category").cat.codes.values, T),
        "first_category_id": np.repeat(s.dept_id.astype("category").cat.codes.values, T),
        "second_category_id": np.repeat(s.dept_id.astype("category").cat.codes.values, T),
        "third_category_id": np.repeat(s.item_id.astype("category").cat.codes.values, T),
        "management_group_id": np.repeat(s.store_id.astype("category").cat.codes.values, T),
        "dow": np.tile(dow, N), "holiday_flag": np.tile(c.event_name_1.notna().astype(int).values, N),
        "activity_flag": np.concatenate([c[f"snap_{st}"].values for st in s.state_id]),
        "discount": 1.0, "precpt": 0.0, "avg_temperature": 0.0, "avg_humidity": 0.0, "avg_wind_level": 0.0,
        # ponytail: no price/weather covariates; sell_prices.csv (200MB) would add a real discount feature
        "sales_c": obs.ravel(), "demand_rec": rec.ravel(), "demand_true": Y.ravel(), "censored": cens.ravel().astype(np.int8)})
    out = d.loc[d.split == "eval", ["store_id", "product_id", "dt", "demand_true", "sales_c"]].copy()
    forecast(d, {"cens": "sales_c", "rec": "demand_rec", "oracle": "demand_true"}, out, tag=f"_m5_{k}")
    acc = pinball_table(out, [("true demand", out.demand_true >= 0, out.demand_true)], ["cens", "rec", "oracle"])
    D = out.demand_true.values
    for L in [1]:
        for cu, co in CU_CO:
            tau = round(cu / (cu + co), 2)
            for name, col in [("censored quantile (no recovery)", "cens"), ("recovered quantile (M1-daily)", "rec"),
                              ("oracle quantile", "oracle")]:
                sim = simulate(out[f"{col}_q{tau}"].values, D, L)
                results.setdefault("kpi", []).append(dict(k=k, tau=tau, policy=name,
                                                          **kpis(sim, D, np.ones(len(D), bool), cu, co)))
    results.setdefault("acc", []).append(acc.assign(k=k))

pd.set_option("display.width", 220)
world = pd.DataFrame(rows); world.round(4).to_csv(f"{OUT}/m5_recovery.csv", index=False)
kpi = pd.DataFrame(results["kpi"]); kpi.to_csv(f"{OUT}/m5_kpi.csv", index=False)
acc = pd.concat(results["acc"]); acc.to_csv(f"{OUT}/m5_accuracy.csv", index=False)
print(f"\nM5 transfer: {N} FOODS series")
print(world.round(3).to_string(index=False))
print(acc[["k", "model", "bias_q0.5", "WAPE_q0.5", "cover_0.8"]].round(3).to_string(index=False))
cc = kpi.pivot_table(index=["k", "tau"], columns="policy", values="cost_per_unit_demand")
gap = pd.DataFrame({"rec_vs_cens_cost": cc["recovered quantile (M1-daily)"] / cc["censored quantile (no recovery)"] - 1,
                    "gap_closed_to_oracle": (cc["censored quantile (no recovery)"] - cc["recovered quantile (M1-daily)"])
                                            / (cc["censored quantile (no recovery)"] - cc["oracle quantile"])})
gap.round(4).to_csv(f"{OUT}/m5_headline.csv")
print(gap.round(3).to_string())
