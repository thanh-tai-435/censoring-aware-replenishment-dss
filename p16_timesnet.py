"""Decision-level comparison with the dataset authors' deep-imputation recovery (TimesNet, frn-50k-baseline).
TimesNet replaces Layer 2 in the semi-synthetic world; forecaster, order rule and simulator are unchanged, so the
result is directly comparable with M1, which is re-run on the same world: the comparison is paired. TimesNet needs a GPU
for all 10K series (about 10 min per epoch on a T4; a 2,000-series subsample takes about 12 min per epoch on a laptop CPU).
Training windows follow the authors (days 0-29, 30-59, 60-89); each evaluation day t is imputed from the window
t-29..t only, so no later observation leaks into the lag features. TimesNet runs in .venv-dl (p16_timesnet_worker.py).
TIMESNET_MODE: full (default) | export (impute only, save out/timesnet_imp_{truth}_n{N}.npz; run on a GPU, e.g. Kaggle)
| import (load those files instead of training; the world hash must match), so LightGBM can run on a CPU machine.
Writes out/review_timesnet.csv (all series; subsamples get an _n{N} suffix)."""
import hashlib, os, subprocess, sys, time
import numpy as np, pandas as pd
from config import OUT, W
from p4_simulate import simulate
from p6_semisynthetic import load, build_truth, experiment, censored_world, T

N_SERIES = int(os.environ.get("TIMESNET_SERIES", 10_000))
EPOCHS = os.environ.get("TIMESNET_EPOCHS", "5")              # authors: 5; smaller only for smoke tests
TRUTHS = os.environ.get("TIMESNET_TRUTHS", "global,stress").split(",")
DL_PY = os.environ.get("DL_PY") or os.path.join(".venv-dl", "Scripts" if os.name == "nt" else "bin", "python")
MODE = os.environ.get("TIMESNET_MODE", "full")
Q, CU_CO, B = [0.5, 0.8, 0.95], [(1, 1), (4, 1), (19, 1)], 1000
EVAL = range(90, 97)

d, H, S = load(N_SERIES, 0)
N = len(H)
night = H.sum(2) - H[..., W].sum(2)
cov = d[["discount", "holiday_flag", "precpt", "avg_temperature"]].values.astype(float).reshape(N, T, 4)


def samples(obs_h, mask, starts):
    """Authors' input: (n, 30*16, 6) = sales with stocked-out hours missing, 4 covariates scaled by their window
    max (+0.1), hour index / 15."""
    x = np.concatenate([np.where(mask[:, s:s + 30] == 1, np.nan, obs_h[:, s:s + 30]) for s in starts])
    c = np.concatenate([cov[:, s:s + 30] for s in starts]); c = c / (c.max(1, keepdims=True) + 0.1)
    hr = np.broadcast_to(np.arange(16)[None, None, :, None] / 15, x.shape + (1,))
    X = np.concatenate([x[..., None], np.broadcast_to(c[:, :, None, :], x.shape + (4,)), hr], -1)
    return X.reshape(len(x), 480, 6).astype(np.float32)


def world_hash(obs_h, mask):
    return hashlib.sha1(np.ascontiguousarray(obs_h, np.float64).tobytes() + np.ascontiguousarray(mask, np.int8).tobytes()).hexdigest()


def imputed(obs_h, mask, truth):
    """(N, T, 16) TimesNet imputation of business hours: trained here (full/export) or loaded (import)."""
    f = f"{OUT}/timesnet_imp_{truth}_n{N}.npz"
    if MODE == "import":
        while not os.path.exists(f):                          # lets the M1 runs start before the GPU export lands
            print("waiting for", f, flush=True); time.sleep(120)
        with np.load(f) as z:
            assert str(z["hash"]) == world_hash(obs_h, mask), f"{f} was imputed on a different world"
            return z["imp"]
    fin, fout = f"{OUT}/timesnet_in_{os.getpid()}.npz", f"{OUT}/timesnet_out_{os.getpid()}.npz"   # pid: truths may run in parallel
    np.savez(fin, train=samples(obs_h, mask, [0, 30, 60]), evalw=samples(obs_h, mask, [t - 29 for t in EVAL]))
    subprocess.run([DL_PY, "p16_timesnet_worker.py", fin, fout, EPOCHS], check=True)
    with np.load(fout) as z:                                   # closes the file (Windows cannot delete open files)
        imp = np.concatenate([np.concatenate(list(z["train"].reshape(3, N, 30, 16)), 1),          # days 0-89
                              z["evalw"].reshape(len(EVAL), N, 30, 16)[:, :, -1].transpose(1, 0, 2)], 1)  # 90-96
    for g in (fin, fout):
        os.remove(g)
    if MODE == "export":
        np.savez(f, imp=imp.astype(np.float32), hash=world_hash(obs_h, mask))
    return imp


def timesnet_recovery(truth):
    return lambda obs_h, mask, obs_d, is_train: night + np.where(mask == 1, np.maximum(imputed(obs_h, mask, truth), 0), obs_h).sum(2)


if MODE == "export":                                          # no forecasting: the world plus TimesNet only
    for truth in TRUTHS:
        censored_world(d, H, build_truth(H, S, truth, np.random.default_rng(0)), 1.2, recovery=timesnet_recovery(truth))
        print("exported", truth, flush=True)
    sys.exit()


rows = []
for truth, rec in [(t, m) for t in TRUTHS for m in ["m1", "timesnet"]]:
    th = build_truth(H, S, truth, np.random.default_rng(0))
    r = experiment(d, H, th, 1.2, quantiles=Q, cu_co=CU_CO, Ls=(1,), tag=f"_timesnet_{truth}_{rec}", seed=0,
                   recovery=timesnet_recovery(truth) if rec == "timesnet" else "m1",
                   reuse=r["out"] if rec == "timesnet" else None)    # same world: cens/oracle forecasts from the M1 run
    out, recerr = r["out"], r["rec"].set_index(["estimate", "days"])
    store = d.store_id.values[::T]
    u, inv = np.unique(store, return_inverse=True)
    rng = np.random.default_rng(0)
    w = np.stack([np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)) for _ in range(B)])[:, inv]
    for tau in Q:
        cs = {c: (lambda s: tau * s["lost"] + (1 - tau) * s["waste"])(simulate(out[f"{c}_q{tau}"].values,
              out.demand_true.values, 1)).sum(1) for c in ["cens", "rec", "oracle"]}
        chg = (w * cs["rec"]).sum(1) / (w * cs["cens"]).sum(1) - 1
        gap = (w * (cs["cens"] - cs["rec"])).sum(1) / (w * (cs["cens"] - cs["oracle"])).sum(1)
        rows.append(dict(truth=truth, recovery=rec, n_series=N, tau=tau, stockout_day_rate=r["world"]["stockout_day_rate"],
                         rec_bias_cens_days=recerr.loc[("M1 recovered", "censored days"), "bias"],
                         rec_wape_cens_days=recerr.loc[("M1 recovered", "censored days"), "WAPE"],
                         cost_change=cs["rec"].sum() / cs["cens"].sum() - 1,
                         store_ci_lo=np.percentile(chg, 2.5), store_ci_hi=np.percentile(chg, 97.5),
                         gap_closed=(cs["cens"] - cs["rec"]).sum() / (cs["cens"] - cs["oracle"]).sum(),
                         gap_store_ci_lo=np.percentile(gap, 2.5), gap_store_ci_hi=np.percentile(gap, 97.5)))
    print(pd.DataFrame(rows[-3:]).round(3).to_string(index=False), flush=True)
suffix = ("_smoke" if EPOCHS != "5" else "") + ("" if N_SERIES >= 10_000 else f"_n{N}") + ("" if len(TRUTHS) == 2 else f"_{TRUTHS[0]}")
pd.DataFrame(rows).round(4).to_csv(f"{OUT}/review_timesnet{suffix}.csv", index=False)
