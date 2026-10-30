"""Phase 3 / Layer 3 (M2): one global LightGBM quantile model per (target, quantile), day-ahead.
Target 'sales' = censored observed sales (baseline), 'demand' = recovered latent demand (proposed).
Output: out/forecasts.parquet, eval days only, columns {target}_q{q} + naive ma7/sd7 per target."""
import numpy as np, pandas as pd, lightgbm as lgb
from config import OUT, QUANTILES, SEED

T = 97                     # p1 sorted rows by (series, day); every series has 97 days
CAT = ["city_id", "first_category_id", "second_category_id", "third_category_id", "management_group_id"]
EXO = ["dow", "discount", "holiday_flag", "activity_flag", "precpt", "avg_temperature", "avg_humidity", "avg_wind_level",
       "cens_lag1", "cens_rate7"]  # ponytail: weather treated as known day-ahead (perfect forecast); state in paper


def lag_feats(y):
    y = pd.DataFrame(y.reshape(-1, T))
    s1 = y.shift(1, axis=1)
    f = {f"lag{k}": y.shift(k, axis=1) for k in [1, 2, 3, 7, 14]}
    for w in [7, 14, 28]:
        f[f"ma{w}"] = s1.T.rolling(w).mean().T
    f["sd7"] = s1.T.rolling(7).std().T
    return pd.DataFrame({k: v.values.ravel() for k, v in f.items()})


def pinball_table(out, truths, models, quantiles=QUANTILES):
    """Secondary metrics: median WAPE/bias, pinball loss and coverage per quantile. truths = [(name, mask, y)]."""
    rows = []
    for truth_name, m, y in truths:
        for target in models:
            r = dict(truth=truth_name, model=target, n=int(m.sum()))
            med = out.loc[m, f"{target}_q0.5"]
            r["WAPE_q0.5"] = np.abs(med - y[m]).sum() / y[m].sum()
            r["bias_q0.5"] = (med - y[m]).sum() / y[m].sum()
            for q in quantiles:
                e = y[m] - out.loc[m, f"{target}_q{q}"]
                r[f"pinball_{q}"] = np.maximum(q * e, (q - 1) * e).mean()
                r[f"cover_{q}"] = (y[m] <= out.loc[m, f"{target}_q{q}"]).mean()
            rows.append(r)
    return pd.DataFrame(rows)


def explain_shap(m, X, name):
    """Layer 6 knowledge discovery: TreeSHAP of the median model on a 5K sample of eval rows (LightGBM native, no extra deps).
    Saves mean |SHAP| ranking and mean SHAP per value quintile of the business drivers (direction of effect)."""
    X = X.sample(min(len(X), 5000), random_state=SEED)   # ponytail: exact TreeSHAP costs trees*leaves*depth^2 per row
    sv = pd.DataFrame(m.predict(X, pred_contrib=True)[:, :-1], columns=X.columns, index=X.index)
    sv.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").to_csv(f"{OUT}/shap{name}.csv")
    dep = []
    for c in ["discount", "activity_flag", "holiday_flag", "precpt", "avg_temperature", "cens_rate7", "dow"]:
        b = pd.qcut(X[c].astype(float).rank(method="first"), 5, labels=False) if X[c].nunique() > 7 else X[c]
        g = pd.DataFrame({"bin": b, "value": X[c].astype(float), "shap": sv[c]}).groupby("bin")
        dep.append(g.agg(value_mean=("value", "mean"), shap_mean=("shap", "mean"), n=("shap", "size")).reset_index().assign(feature=c))
    pd.concat(dep).round(4).to_csv(f"{OUT}/shap_dependence{name}.csv", index=False)


def forecast(d, targets, out, tag="", quantiles=QUANTILES, explain=False, seed=SEED):
    """Fit one quantile model per (target, q) on train days >= 28 and add eval-day predictions to out.
    d: daily table sorted by (series, day) with a 'censored' column; targets: {name: column}."""
    g = d.groupby(["store_id", "product_id"], sort=False)
    t = g.cumcount().values
    d = d.assign(cens_lag1=g.censored.shift(1),
                 cens_rate7=pd.Series(lag_feats(d.censored.values.astype(float))["ma7"].values, index=d.index))
    train_rows = ((t >= 28) & (d.split == "train")).values
    eval_rows = (d.split == "eval").values
    for target, col in targets.items():
        F = pd.concat([lag_feats(d[col].values), d[CAT + EXO].reset_index(drop=True)], axis=1)
        for c in CAT:
            F[c] = F[c].astype("category")
        out[f"{target}_ma7"], out[f"{target}_sd7"] = F.loc[eval_rows, "ma7"].values, F.loc[eval_rows, "sd7"].values
        ds = lgb.Dataset(F[train_rows], d.loc[train_rows, col].values, free_raw_data=False)
        for q in quantiles:
            m = lgb.train(dict(objective="quantile", alpha=q, learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                               bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.8, seed=seed, verbose=-1), ds, 400)
            out[f"{target}_q{q}"] = np.maximum(m.predict(F[eval_rows]), 0)
            print(tag, target, q, "done", flush=True)
            if q == 0.5:
                imp = pd.Series(m.feature_importance("gain"), F.columns).sort_values(ascending=False)
                imp.to_csv(f"{OUT}/importance{tag}_{target}.csv")
                if explain:
                    explain_shap(m, F[eval_rows], f"{tag}_{target}")
        qc = [f"{target}_q{q}" for q in quantiles]
        out[qc] = np.sort(out[qc].values, axis=1)           # no quantile crossing
    return out


if __name__ == "__main__":
    d = pd.read_parquet(f"{OUT}/recovered.parquet")
    out = d.loc[d.split == "eval", ["store_id", "product_id", "dt", "city_id", "first_category_id",
                                    "sale_amount", "demand", "censored"]].copy()
    forecast(d, {"sales": "sale_amount", "demand": "demand"}, out, explain=True)
    out.to_parquet(f"{OUT}/forecasts.parquet")
    # (A) uncensored eval days' sales, (B) recovered demand on all days
    acc = pinball_table(out, [("A uncensored days, D=sales", out.censored == 0, out.sale_amount),
                              ("B all days, D=recovered", out.censored >= 0, out.demand)], ["sales", "demand"])
    acc.to_csv(f"{OUT}/forecast_accuracy.csv", index=False)
    print(acc.T.to_string())
