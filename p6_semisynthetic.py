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
              (independent of M1's proportionality assumption)
    stress  = donor hours times a day-level demand shock (mean-one lognormal, sigma 0.5, plus 2x spikes on 5% of
              days) and a slowly drifting intraday shape: neither the level nor the shape of the filled hours
              follows the same day's in-stock rate or a stationary profile."""
    Hw, Mw = H[..., W], S[..., W].astype(np.float32)
    gp = Hw[Mw.sum(2) == 0].sum(0); gp = gp / gp.sum()
    P_in = (gp * (1 - Mw)).sum(2)
    rate = (Hw * (1 - Mw)).sum(2) / np.maximum(P_in, 1e-9)        # implied daily W-demand
    ok = P_in >= MIN_P_IN
    fb = np.nan_to_num(np.nanmedian(np.where(ok, rate, np.nan), 1))
    rate = np.where(ok, rate, np.maximum(fb[:, None], Hw.sum(2)))
    fill = gp * rate[..., None]
    if mode in ("donor", "stress"):
        U = Mw.sum(2) == 0                                       # uncensored days per series
        order = np.argsort(~U, axis=1, kind="stable")             # uncensored day indices first
        cnt = U.sum(1, keepdims=True)
        pick = order[np.arange(len(U))[:, None], (rng.random(U.shape) * np.maximum(cnt, 1)).astype(int)]
        donor = Hw[np.arange(len(U))[:, None], pick]
        fill = np.where(cnt[..., None] > 0, donor, fill)          # no uncensored day -> global fill
    if mode == "stress":
        N_, T_ = Mw.shape[:2]
        shock = rng.lognormal(-0.125, 0.5, (N_, T_)) * np.where(rng.random((N_, T_)) < 0.05, 2.0, 1.0)
        h = (np.array(W) - 13.5) / 8                            # -1 (morning) .. +1 (evening)
        tilt = 1 + 0.4 * np.sin(2 * np.pi * np.arange(T_) / 45)[:, None] * h[None, :]   # drifts over ~45 days
        fill = fill * shock[..., None] * (tilt / tilt.mean(1, keepdims=True))[None]
    if mode == "noise":
        fill = fill * rng.lognormal(-0.125, 0.5, fill.shape)
    elif mode == "evening":
        fill = fill * np.where(np.array(W) >= 17, 1.5, 1.0)
    return np.where(Mw == 1, np.maximum(Hw, fill), Hw).astype(np.float32)


def retailer(true_h, k, policy="mean"):
    """Stock set each morning from the retailer's own OBSERVED W-sales of the last 7 days; depletes hour by hour.
    mean     = k * mean                                  (base case)
    safety   = mean + 1.28 * s.d.                        (forecast-plus-safety-stock, ~90% normal service)
    adaptive = m_t * mean, m_t up 15% after a stockout day, down 3% after a day without one, in [0.8, 2.5]."""
    N = len(true_h)
    cum = np.cumsum(true_h, 2)
    obs_h = np.zeros_like(true_h); mask = np.zeros_like(true_h)
    m = np.full(N, k, dtype=np.float64)
    for t in range(T):
        win = true_h[:, :7].sum(2) if t < 7 else obs_h[:, t - 7:t].sum(2)
        if policy == "safety":
            C = win.mean(1) + 1.28 * win.std(1)
        elif policy == "adaptive":
            if t > 0:
                m = np.clip(np.where(mask[:, t - 1].any(1), m * 1.15, m * 0.97), 0.8, 2.5)
            C = m * win.mean(1)
        else:
            C = k * win.mean(1)
        C = C[:, None]
        obs_h[:, t] = np.diff(np.minimum(cum[:, t], C), prepend=0, axis=1)
        mask[:, t] = np.concatenate([np.zeros((N, 1)), cum[:, t, :-1]], 1) >= C   # empty at start of hour
    return obs_h, mask


def retailer_ext(true_h, k=1.2, lag=0, cap=None, every=1, defer=0.0, subst=0.0, partner=None):
    """Richer synthetic retailer, simulated hour by hour. Returns (realised true demand, observed sales, mask):
    deferred and substituted demand depend on the retailer's own stockouts, so the truth is endogenous.
    lag    = order lead time: day t's stock is set from observed sales up to day t-1-lag
    cap    = per-series stock capacity (shelf / order limit)
    every  = deliveries every `every` days, order-up-to k*every*mean; stock carries over between deliveries
    defer  = share of yesterday's lost demand that comes back today (spread over today's hourly shape)
    subst  = share (scalar or per series) of the partner series' lost demand (same store, previous hour) that
             switches to this series; partner = index of that series (None = no substitution)
    With defaults it reproduces retailer(true_h, k)."""
    N, T_, Hn = true_h.shape
    true_h = true_h.astype(np.float64).copy()
    obs_h, mask = np.zeros_like(true_h), np.zeros_like(true_h)
    left, lost_prev, lost_h = np.zeros(N), np.zeros(N), np.zeros(N)
    for t in range(T_):
        if defer and t:
            sh = true_h[:, t]; tot = sh.sum(1, keepdims=True)
            true_h[:, t] += defer * lost_prev[:, None] * np.where(tot > 0, sh / np.maximum(tot, 1e-12), 1 / Hn)
        hi = t - lag
        win = true_h[:, :7].sum(2) if hi < 7 else obs_h[:, hi - 7:hi].sum(2)
        if t % every == 0:
            stock = np.maximum(k * every * win.mean(1), left if every > 1 else 0)
            if cap is not None:
                stock = np.minimum(stock, cap)
        else:
            stock = left
        lost = np.zeros(N)
        for h in range(Hn):
            if partner is not None:
                true_h[:, t, h] += subst * lost_h[partner]
            dem = true_h[:, t, h]
            mask[:, t, h] = stock <= 1e-9                          # empty at the start of the hour
            sold = np.minimum(stock, dem)
            obs_h[:, t, h], stock = sold, stock - sold
            lost_h = dem - sold; lost += lost_h
        left, lost_prev = stock, lost
    return true_h.astype(np.float32), obs_h.astype(np.float32), mask.astype(np.float32)


def km_daily(S, cens, train):
    """Recovery baseline: per-series Kaplan-Meier (product-limit) estimate of the daily demand distribution from
    train days, censored days right-censored at sales; a censored day gets E[D | D >= S] under that estimate.
    Mass left beyond the largest observation sits on it (Efron), so the upper tail is understated by design."""
    D = S.astype(np.float64).copy()
    for i in range(len(S)):
        x, c = S[i, train[i]], cens[i, train[i]]
        o = np.lexsort((c, x)); x, c = x[o], c[o]                  # ties: events before censorings
        at_risk = len(x) - np.arange(len(x))
        surv = np.cumprod(np.where(c, 1.0, 1 - 1 / at_risk))
        mass = -np.diff(np.concatenate([[1.0], surv]))             # jump at each event
        mass[-1] += surv[-1]                                       # Efron: remaining mass on the max
        for t in np.where(cens[i])[0]:
            s = S[i, t]; m = x >= s
            D[i, t] = max(s, (x[m] * mass[m]).sum() / mass[m].sum()) if mass[m].sum() > 0 else s
    return D


def tobit_daily(S, cens, train, dow, iters=30):
    """Recovery baseline that ignores the hourly labels: a daily Tobit (censored Normal) model,
    D_it ~ N(a_i * w_dow(t), s_i^2), fitted by EM on the censored likelihood (Eq. 1); censored days get
    E[D | D >= S]. Only WHICH days were censored is used, not when in the day the stockout began."""
    from scipy.stats import norm
    D, D2 = S.copy(), S ** 2
    w = np.ones(7)
    for _ in range(iters):
        a = (D * train).sum(1) / np.maximum((w[dow] * train).sum(1), 1e-9)
        sel = [train[0] & (dow == j) for j in range(7)]
        w = np.array([D[:, s].sum() / max(a.sum() * s.sum(), 1e-9) for s in sel]); w = w / w.mean()
        mu = a[:, None] * w[dow][None, :]
        sd = np.sqrt(np.maximum(((D2 - 2 * mu * D + mu ** 2) * train).sum(1) / train.sum(1), 1e-6))[:, None]
        z = (S - mu) / sd
        lam = norm.pdf(z) / np.maximum(norm.sf(z), 1e-12)             # inverse Mills ratio
        ED = mu + sd * lam
        ED2 = mu ** 2 + sd ** 2 + sd * (S + mu) * lam
        D, D2 = np.where(cens, np.maximum(S, ED), S), np.where(cens, np.maximum(S ** 2, ED2), S ** 2)
    return D


def censored_world(d, H, true_h, k, retailer_policy="mean", recovery="m1", retailer_fn=None):
    """What the planner sees: retailer-censored sales, plus recovery of it. Returns d with
    sales_c / demand_rec / demand_true / censored columns, the recovery-error table and world stats.
    recovery: m1 (hourly profile scaling, the proposed layer), uniform (same scaling with a flat intraday
    profile), tobit (daily censored-Normal EM, no hourly information), km (daily Kaplan-Meier), or a callable
    f(obs_h, mask, obs_d, is_train) -> daily demand (e.g. a deep imputation model).
    retailer_fn: true_h -> (realised true_h, obs_h, mask) replaces the built-in retailer."""
    N = len(true_h)
    night = H.sum(2) - H[..., W].sum(2)                        # sales outside 6-21h, taken as uncensored
    store = d.store_id.values.reshape(N, T)[:, 0]
    is_train = (d.split == "train").to_numpy().reshape(N, T)
    if retailer_fn is not None:
        true_h, obs_h, mask = retailer_fn(true_h)
    else:
        obs_h, mask = retailer(true_h, k, retailer_policy)
    true_d = night + true_h.sum(2)
    obs_d = night + obs_h.sum(2)
    cens = mask.sum(2) > 0

    # recovery on the censored world (train-only parameters, like the real pipeline)
    if callable(recovery):
        rec_d = recovery(obs_h, mask, obs_d, is_train)
    elif recovery == "km":
        rec_d = km_daily(obs_d, cens, is_train)
    elif recovery == "tobit":
        rec_d = tobit_daily(obs_d, cens, is_train, d.dow.values.reshape(N, T)[0])
    else:
        p = profiles(obs_h, mask, store, is_train, SHRINK_K)
        if recovery == "uniform":
            p = np.full_like(p, 1 / p.shape[1])
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


def experiment(d, H, true_h, k, quantiles=QUANTILES, cu_co=CU_CO, Ls=(1, 2, 3), tag="_semi", warmup=0, seed=SEED,
               retailer_policy="mean", recovery="m1", retailer_fn=None, eval_start=None, reuse=None):
    """warmup: first eval days excluded from KPIs (inventory starts empty); matters for long horizons, L>1.
    eval_start: rolling origin, evaluate days [eval_start, eval_start+7) with training on the days before it.
    reuse: 'out' of an earlier run on the same world and seed; its cens/oracle forecasts, which do not depend on
    the recovery layer, are copied instead of refitted."""
    N = len(true_h)
    if eval_start is not None:
        t = np.tile(np.arange(T), N)
        d = d.assign(split=np.where(t < eval_start, "train", np.where(t < eval_start + 7, "eval", "none")))
    d, rec, world = censored_world(d, H, true_h, k, retailer_policy, recovery, retailer_fn)

    # forecasts for three planners
    out =d.loc[d.split == "eval", ["store_id", "product_id", "dt", "city_id", "demand_true", "sales_c"]].copy()
    targets = {"cens": "sales_c", "rec": "demand_rec", "oracle": "demand_true"}
    if reuse is not None:
        assert (reuse.sales_c.values == out.sales_c.values).all() and (reuse.demand_true.values == out.demand_true.values).all()
        for c in [c for c in reuse.columns if c.startswith(("cens_", "oracle_"))]:
            out[c] = reuse[c].values
        targets = {"rec": "demand_rec"}
    forecast(d, targets, out, tag=tag, quantiles=quantiles, seed=seed)
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
