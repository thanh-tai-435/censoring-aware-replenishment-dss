"""Phase 2 / Layer 2 (M1): latent-demand recovery by scaling the in-stock hourly profile
over stocked-out hours, validated on uncensored days with real stockout masks transplanted.
Output: out/recovered.parquet = daily table + demand, imputed_share, recovery_flag."""
import numpy as np, pandas as pd
from config import OUT, W, SHRINK_K, MIN_P_IN, SEED

def profiles(Hw, Mw, store, fit_days, k):
    """Hourly demand-share profile per series from uncensored fit days, shrunk toward its store's profile."""
    use = fit_days & (Mw.sum(2) == 0)
    hs = (Hw * use[..., None]).sum(1)                                   # (N, 16) series hourly totals
    st = pd.DataFrame(hs).groupby(store).transform("sum").values         # store hourly totals
    p_store = st / np.maximum(st.sum(1, keepdims=True), 1e-9)
    if k == np.inf:
        return p_store
    p = (hs + k * p_store) / (hs.sum(1, keepdims=True) + k)
    return p / np.maximum(p.sum(1, keepdims=True), 1e-9)


def recover(H, M, S, p, fallback):
    """D = S + rate_in_stock * stocked-out profile mass. fallback (N,) used where too little is in stock."""
    P_in = (p[:, None, :] * (1 - M)).sum(2)
    obs_in = (H * (1 - M)).sum(2)
    lost = obs_in * (1 - P_in) / np.maximum(P_in, 1e-9)
    ok = (P_in >= MIN_P_IN) | (M.sum(2) == 0)
    D = np.where(ok, S + np.where(M.sum(2) > 0, lost, 0), np.maximum(S, fallback[:, None]))
    flag = np.where(M.sum(2) == 0, 0, np.where(ok, 1, 2))
    return D, flag


def series_fallback(D, flag, days):
    """Median of this series' reliably-recovered demand on the given days."""
    x = np.where(days & (flag < 2), D, np.nan)
    return np.nan_to_num(np.nanmedian(x, 1))


if __name__ == "__main__":
    d = pd.read_parquet(f"{OUT}/daily.parquet")
    z = np.load(f"{OUT}/hourly.npz")
    N, T = d.groupby(["store_id", "product_id"]).ngroups, 97     # p1 sorted: every series has 97 days
    Hw = z["H"][:, W].reshape(N, T, len(W))
    Mw = z["S"][:, W].reshape(N, T, len(W)).astype(np.float32)  # 1 = out of stock
    sales = d.sale_amount.values.reshape(N, T)
    store = d.store_id.values.reshape(N, T)[:, 0]
    is_train = (d.split == "train").to_numpy().reshape(N, T)


    # ---- validation: fit profiles on train days 0-59, test on uncensored days 60-89 with transplanted real masks
    rng = np.random.default_rng(SEED)
    fit = np.zeros((N, T), bool); fit[:, :60] = True
    test = np.zeros((N, T), bool); test[:, 60:90] = True
    test &= (Mw.sum(2) == 0)
    real_masks = Mw.reshape(-1, len(W))[Mw.reshape(-1, len(W)).sum(1) > 0]
    art = np.zeros_like(Mw)
    art[test] = real_masks[rng.integers(0, len(real_masks), test.sum())]
    H_obs = Hw * (1 - art)
    S_obs = sales - (Hw * art).sum(2)
    truth = sales

    rows = []

    def score(name, D, flag=None):
        t, e = truth[test], D[test]
        sev = art[test].sum(1)
        for lo, hi, lab in [(1, 16, "all"), (1, 4, "1-4h out"), (5, 10, "5-10h out"), (11, 15, "11-15h out"), (16, 16, "full day")]:
            m = (sev >= lo) & (sev <= hi)
            rows.append(dict(method=name, severity=lab, n=int(m.sum()),
                             WAPE=np.abs(e[m] - t[m]).sum() / t[m].sum(), bias=(e[m] - t[m]).sum() / t[m].sum()))

    score("naive (observed sales)", S_obs)
    gl = Hw[fit & (Mw.sum(2) == 0)].sum(0); gl = np.tile(gl / gl.sum(), (N, 1))
    for name, p in [("global profile", gl), ("store profile", profiles(Hw, Mw, store, fit, np.inf))] + \
                   [(f"series profile k={k}", profiles(Hw, Mw, store, fit, k)) for k in [0.5, 2, 5, 20]]:
        D0, f0 = recover(Hw, Mw, sales, p, np.zeros(N))            # real history in fit window -> fallback level
        fb = series_fallback(D0, f0, fit)
        D, flag = recover(H_obs, art, S_obs, p, fb)
        score(name, D)
    val = pd.DataFrame(rows)
    val.to_csv(f"{OUT}/recovery_validation.csv", index=False)
    print(val.pivot(index="method", columns="severity", values="bias").round(3).to_string())
    print(val.pivot(index="method", columns="severity", values="WAPE").round(3).to_string())
    print(val[val.method == "naive (observed sales)"][["severity", "n"]].to_string(index=False))

    # ---- final recovery: profiles on all train days, applied to every day (eval days use train-only profiles)
    p = profiles(Hw, Mw, store, is_train, SHRINK_K)
    D0, f0 = recover(Hw, Mw, sales, p, np.zeros(N))
    D, flag = recover(Hw, Mw, sales, p, series_fallback(D0, f0, is_train))
    d["demand"] = D.ravel()
    d["recovery_flag"] = flag.ravel()                       # 0 observed, 1 profile-scaled, 2 series fallback
    d["imputed_share"] = np.where(d.demand > 0, (d.demand - d.sale_amount) / d.demand, 0)  # uncertainty proxy
    d.to_parquet(f"{OUT}/recovered.parquet")
    print(f"\nrecovered/observed total: {d.demand.sum() / d.sale_amount.sum():.4f}  "
          f"observed underestimates by {1 - d.sale_amount.sum() / d.demand.sum():.2%}")
    print(d.recovery_flag.value_counts(normalize=True).round(4).to_string())
