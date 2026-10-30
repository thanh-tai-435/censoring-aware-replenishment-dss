"""Exact benchmark for the shelf-life-adjusted fractile (Eq. 5) on small instances.
Perishable inventory with lifetime L days, zero lead time, FIFO issuing, lost sales, i.i.d. integer demand
(negative binomial), cost cu per lost unit and co per expired unit, cu + co = 1 -- the same dynamics as p4_simulate.
The optimal policy is computed by relative value iteration over the full age-structured state; three base-stock
rules (order up to S, given the total on hand) are evaluated exactly on the same Markov chain:
  tau_L     : S = F^-1(cu / (cu + co / L))   (the heuristic used in the paper)
  tau       : S = F^-1(cu / (cu + co))       (single-period rule, ignores carry-over)
  best base : the best constant S, found by search
Writes out/shelflife_dp.csv: long-run average cost per unit demand and gap to the optimum."""
import itertools
import numpy as np, pandas as pd
from scipy.stats import nbinom
from config import OUT


def demand(mu, cv):
    var = (cv * mu) ** 2
    r = mu ** 2 / (var - mu)
    p = nbinom.pmf(np.arange(int(nbinom.ppf(0.9999, r, r / (r + mu))) + 1), r, r / (r + mu))
    return p / p.sum()


def model(L, pd_, M, tau):
    """States: on-hand stock by age (oldest first), L-1 buckets with total <= M. Returns state list, and for every
    (state, order q) the expected one-day cost and the next-state index for each demand value (cost = inf marks an infeasible q)."""
    states = [s for s in itertools.product(range(M + 1), repeat=L - 1) if sum(s) <= M]
    radix = (M + 1) ** np.arange(L - 1)
    code2idx = np.full((M + 1) ** (L - 1), -1)
    code2idx[[int(np.dot(s, radix)) for s in states]] = np.arange(len(states))
    nD = len(pd_)
    cost = np.full((len(states), M + 1), np.inf)
    nxt = np.zeros((len(states), M + 1, nD), dtype=np.int64)
    for i, s in enumerate(states):
        for q in range(M + 1 - sum(s)):
            need = np.arange(nD)                        # every demand value at once
            left = []
            for a, x in enumerate(list(s) + [q]):       # oldest first
                take = np.minimum(x, need); need = need - take; left.append(x - take)
            cost[i, q] = (tau * need + (1 - tau) * left[0]) @ pd_   # oldest bucket expires tonight
            nxt[i, q] = code2idx[np.stack(left[1:], 1) @ radix]
    return states, cost, nxt


def rvi(cost, nxt, pd_, policy=None, tol=1e-9, it=20000):
    """Relative value iteration (aperiodicity transform 0.5). policy: fixed action per state, or None = optimise."""
    h = np.zeros(cost.shape[0])
    if policy is not None:
        ar = np.arange(len(h)); cost, nxt = cost[ar, policy][:, None], nxt[ar, policy][:, None]
    for _ in range(it):
        Th = (cost + h[nxt] @ pd_).min(1)
        diff = Th - h
        h = 0.5 * h + 0.5 * (Th - Th[0])
        if diff.max() - diff.min() < tol:
            break
    return (diff.max() + diff.min()) / 2


def base_stock(states, S, M):
    return np.array([min(max(S - sum(s), 0), M - sum(s)) for s in states])


rows = []
for L, mu, cv, tau in itertools.product([2, 3], [3, 6, 10], [0.6, 1.0], [0.5, 0.8, 0.95]):
    pd_ = demand(mu, cv)
    F = np.cumsum(pd_)
    M = int(np.searchsorted(F, 0.9995)) + 2
    states, cost, nxt = model(L, pd_, M, tau)
    g_opt = rvi(cost, nxt, pd_)
    q = lambda t: int(np.searchsorted(F, t - 1e-12))          # F^-1(t) for a discrete F
    tau_L = tau / (tau + (1 - tau) / L)
    g = {S: rvi(cost, nxt, pd_, base_stock(states, S, M)) for S in range(M + 1)}
    S_best = min(g, key=g.get)
    rows.append(dict(L=L, mu=mu, cv=cv, tau=tau, opt=g_opt / mu, tauL=g[q(tau_L)] / mu, single=g[q(tau)] / mu,
                     best_base=g[S_best] / mu, S_tauL=q(tau_L), S_single=q(tau), S_best=S_best))
    r = rows[-1]
    print(f"L={L} mu={mu} cv={cv} tau={tau}: opt {r['opt']:.4f}  tau_L {r['tauL']:.4f}  single {r['single']:.4f}"
          f"  best-base {r['best_base']:.4f}", flush=True)
res = pd.DataFrame(rows)
for c in ["tauL", "single", "best_base"]:
    res[f"gap_{c}"] = res[c] / res.opt - 1
res.round(5).to_csv(f"{OUT}/shelflife_dp.csv", index=False)
print(res.groupby("L")[["gap_tauL", "gap_single", "gap_best_base"]].agg(["mean", "max"]).round(4).to_string())
