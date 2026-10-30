"""Step 1 of the honest evaluation: semi-synthetic censoring with known ground truth.
FreshRetailNet has no uncensored series (every series has >= 11 stockout days), so:
  1. True world: real hourly sales on in-stock hours; stocked-out hours filled with the GLOBAL hourly profile
     (deliberately not M1's series-shrunk profile, so recovery is not graded against its own assumption).
  2. Synthetic retailer: daily stock C_t = k * mean(observed sales, last 7 days) -> the self-reinforcing loop.
     Stock depletes hour by hour against true demand, producing endogenous hourly stockout masks.
  3. Three planners see the censored world: train on censored sales / M1-recovered demand / true demand (oracle).
  4. Score orders on TRUE eval-week demand (no truth-by-estimate).
p7_robustness.py reruns this with other truth constructions and retailer k."""
import numpy as np, pandas as pd
from config import OUT, W, SHRINK_K, MIN_P_IN, QUANTILES, CU_CO, BASE_CU_CO, SEED
from p2_recovery import profiles, recover, series_fallback
from p3_forecast import forecast, pinball_table
from p4_simulate import simulate, kpis

T = 97


def load(n_series=None, seed=SEED):
    """Daily table + hourly arrays, optionally for a random subset of series (same subset for every call)."""
    d = pd.read_parquet(f"{OUT}/daily.parquet")
    z = np.load(f"{OUT}/hourly.npz")
    N = len(d) // T
    idx = np.arange(N) if n_series is None or n_series >= N else np.sort(np.random.default_rng(seed).choice(N, n_series, replace=False))
    rows = (idx[:, None] * T + np.arange(T)).ravel()
    H, S = z["H"].reshape(N, T, 24)[idx], z["S"].reshape(N, T, 24)[idx]
    return d.iloc[rows].reset_index(drop=True), H, S


def build_truth(H, S, mode="global", rng=None):
    """True hourly W-demand. In-stock hours = real sales; stocked-out hours filled by mode:
    global  = global profile x the day's in-stock rate
    noise   = global, times mean-one lognormal noise (sigma 0.5) per filled hour
    evening = global, but 17-21h demand 1.5x what the profile implies (adversarial: most stockouts are late)
    donor   = the real hourly sales of a random uncensored day of the same series, NOT scaled to the day's rate
              (independent of M1's proportionality assumption)."""
    Hw, Mw = H[..., W], S[..., W].astype(np.float32)
    gp = Hw[Mw.sum(2) == 0].sum(0); gp = gp / gp.sum()
    P_in = (gp * (1 - Mw)).sum(2)
    rate = (Hw * (1 - Mw)).sum(2) / np.maximum(P_in, 1e-9)        # implied daily W-demand
    ok = P_in >= MIN_P_IN
    fb = np.nan_to_num(np.nanmedian(np.where(ok, rate, np.nan), 1))
    rate = np.where(ok, rate, np.maximum(fb[:, None], Hw.sum(2)))
    fill = gp * rate[..., None]
    if mode == "donor":
        U = Mw.sum(2) == 0                                       # uncensored days per series
        order = np.argsort(~U, axis=1, kind="stable")             # uncensored day indices first
        cnt = U.sum(1, keepdims=True)
        pick = order[np.arange(len(U))[:, None], (rng.random(U.shape) * np.maximum(cnt, 1)).astype(int)]
        donor = Hw[np.arange(len(U))[:, None], pick]
        fill = np.where(cnt[..., None] > 0, donor, fill)          # no uncensored day -> global fill
    if mode == "noise":
        fill = fill * rng.lognormal(-0.125, 0.5, fill.shape)
    elif mode == "evening":
        fill = fill * np.where(np.array(W) >= 17, 1.5, 1.0)
    return np.where(Mw == 1, np.maximum(Hw, fill), Hw).astype(np.float32)


def retailer(true_h, k):
    """Daily stock = k * mean observed W-sales of the last 7 days; depletes hour by hour."""
    N = len(true_h)
    cum = np.cumsum(true_h, 2)
    obs_h = np.zeros_like(true_h); mask = np.zeros_like(true_h)
    for t in range(T):
        hist = true_h[:, :7].sum(2).mean(1) if t < 7 else obs_h[:, t - 7:t].sum(2).mean(1)
        C = k * hist[:, None]
        obs_h[:, t] = np.diff(np.minimum(cum[:, t], C), prepend=0, axis=1)
        mask[:, t] = np.concatenate([np.zeros((N, 1)), cum[:, t, :-1]], 1) >= C   # empty at start of hour
    return obs_h, mask


def censored_world(d, H, true_h, k):
    """What the planner sees: retailer-censored sales, plus M1 recovery of it. Returns d with
    sales_c / demand_rec / demand_true / censored columns, the recovery-error table and world stats."""
    N = len(true_h)
    night = H.sum(2) - H[..., W].sum(2)                        # sales outside 6-21h, taken as uncensored
    store = d.store_id.values.reshape(N, T)[:, 0]
    is_train = (d.split == "train").to_numpy().reshape(N, T)
    true_d = night + true_h.sum(2)
    obs_h, mask = retailer(true_h, k)
    obs_d = night + obs_h.sum(2)
    cens = mask.sum(2) > 0

    # M1 recovery on the censored world (train-only profiles, like the real pipeline)
    p = profiles(obs_h, mask, store, is_train, SHRINK_K)
    D0, f0 = recover(obs_h, mask, obs_d, p, np.zeros(N))
    rec_d, _ = recover(obs_h, mask, obs_d, p, series_fallback(D0, f0, is_train))
    rec = []
    for name, est in [("censored sales", obs_d), ("M1 recovered", rec_d)]:
        for lab, m in [("censored days", cens), ("all days", np.ones_like(cens))]:
            rec.append(dict(estimate=name, days=lab, bias=(est[m] - true_d[m]).sum() / true_d[m].sum(),
                            WAPE=np.abs(est[m] - true_d[m]).sum() / true_d[m].sum()))
    world = dict(stockout_day_rate=cens.mean(), observed_share=obs_d.sum() / true_d.sum())
    d = d.assign(sales_c=obs_d.ravel(), demand_rec=rec_d.ravel(), demand_true=true_d.ravel(),
                 censored=cens.ravel().astype(np.int8))
    return d, pd.DataFrame(rec), world


def q_at(out, prefix, tau, quantiles):
    """Quantile at any tau by linear interpolation over the trained grid (clamped at the grid ends)."""
    qs = np.array(quantiles)
    X = out[[f"{prefix}_q{q}" for q in quantiles]].values
    j = np.clip(np.searchsorted(qs, tau), 1, len(qs) - 1)
    w = np.clip((tau - qs[j - 1]) / (qs[j] - qs[j - 1]), 0, 1)
    return pd.Series(X[:, j - 1] * (1 - w) + X[:, j] * w, index=out.index)


def experiment(d, H, true_h, k, quantiles=QUANTILES, cu_co=CU_CO, Ls=(1, 2, 3), tag="_semi", warmup=0, seed=SEED):
    """warmup: first eval days excluded from KPIs (inventory starts empty); matters for long horizons, L>1."""
    N = len(true_h)
    d, rec, world = censored_world(d, H, true_h, k)

    # forecasts for three planners
    out =d.loc[d.split == "eval", ["store_id", "product_id", "dt", "city_id", "demand_true", "sales_c"]].copy()
    forecast(d, {"cens": "sales_c", "rec": "demand_rec", "oracle": "demand_true"}, out, tag=tag, quantiles=quantiles, seed=seed)
    acc = pinball_table(out, [("true demand, all eval days", out.demand_true >= 0, out.demand_true)],
                        ["cens", "rec", "oracle"], quantiles=quantiles)

    # decisions scored on true demand
    # shelf-life-adjusted fractile: a leftover unit gets L chances to sell, so spoilage cost is ~co/L per day
    # ponytail: first-order heuristic; exact multi-period perishable base-stock is a DP, add if reviewers ask
    tau_L = lambda cu, co, L: cu / (cu + co / L)
    pol = {"naive ma7 of censored sales": lambda cu, co, L: out.cens_ma7,
           "censored quantile (no recovery)": lambda cu, co, L: q_at(out, "cens", tau_L(cu, co, 1), quantiles),
           "recovered quantile (proposed)": lambda cu, co, L: q_at(out, "rec", tau_L(cu, co, 1), quantiles),
           "oracle quantile (true-demand history)": lambda cu, co, L: q_at(out, "oracle", tau_L(cu, co, 1), quantiles),
           "censored quantile, shelf-life-adjusted": lambda cu, co, L: q_at(out, "cens", tau_L(cu, co, L), quantiles),
           "recovered quantile, shelf-life-adjusted": lambda cu, co, L: q_at(out, "rec", tau_L(cu, co, L), quantiles),
           "oracle quantile, shelf-life-adjusted": lambda cu, co, L: q_at(out, "oracle", tau_L(cu, co, L), quantiles)}
    Te = len(out) // N                                         # simulated days per series
    D, allm = out.demand_true.values, np.tile(np.arange(Te) >= warmup, N)
    rows = []
    for L in Ls:
        for cu, co in cu_co:
            tau = round(cu / (cu + co), 2)
            for name, fn in pol.items():
                sim = simulate(fn(cu, co, L).fillna(0).clip(lower=0).values, D, L, T=Te)
                rows.append(dict(L=L, cu=cu, co=co, tau=tau, policy=name, **kpis(sim, D, allm, cu, co, T=Te)))
    res = pd.DataFrame(rows)
    c = res.pivot_table(index=["L", "tau"], columns="policy", values="cost_per_unit_demand")
    gap = pd.DataFrame({
        "rec_vs_cens_cost": c["recovered quantile (proposed)"] / c["censored quantile (no recovery)"] - 1,
        "gap_closed_to_oracle": (c["censored quantile (no recovery)"] - c["recovered quantile (proposed)"])
                                / (c["censored quantile (no recovery)"] - c["oracle quantile (true-demand history)"])})
    return dict(world=world, rec=rec, out=out, acc=acc, res=res, gap=gap)


if __name__ == "__main__":
    d, H, S = load()
    true_h = build_truth(H, S, "global")
    target_rate = (S[..., W].sum(2) > 0).mean()
    calib = []
    for k in [1.0, 1.2, 1.4, 1.6, 1.8]:
        _, m = retailer(true_h, k)
        calib.append((k, (m.sum(2) > 0).mean()))
        print(f"k={k}: stockout-day rate {calib[-1][1]:.3f} (real {target_rate:.3f})", flush=True)
    K = min(calib, key=lambda c: abs(c[1] - target_rate))[0]
    r = experiment(d, H, true_h, K)
    print(f"chosen k={K}; world: {r['world']}")
    r["rec"].to_csv(f"{OUT}/semi_recovery.csv", index=False)
    r["out"].to_parquet(f"{OUT}/semi_forecasts.parquet")
    r["acc"].to_csv(f"{OUT}/semi_accuracy.csv", index=False)
    r["res"].to_csv(f"{OUT}/semi_kpi.csv", index=False)
    r["gap"].round(4).to_csv(f"{OUT}/semi_headline.csv")
    pd.set_option("display.width", 200)
    cu, co = BASE_CU_CO
    print(r["rec"].round(4).to_string(index=False))
    print(r["acc"].set_index("model").T.round(4).to_string())
    print(f"\n=== semi-synthetic KPIs vs TRUE demand, cu:co={cu}:{co} ===")
    print(r["res"][(r["res"].cu == cu) & (r["res"].co == co)].drop(columns=["cu", "co", "tau"]).round(4).to_string(index=False))
    print("\n=== recovered vs censored (same order rule); share of censored->oracle cost gap closed ===")
    print(r["gap"].round(3).to_string())
