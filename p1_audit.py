"""Phase 1 / Layer 1: data audit + daily table with censoring mask and hourly arrays."""
import os
import numpy as np, pandas as pd
from config import DATA, OUT, W

os.makedirs(OUT, exist_ok=True)
frames = []
for split in ["train", "eval"]:
    d = pd.read_parquet(f"{DATA}/{split}.parquet")
    d["split"] = split
    frames.append(d)
d = pd.concat(frames, ignore_index=True)
d["dt"] = pd.to_datetime(d["dt"])
d = d.sort_values(["store_id", "product_id", "dt"]).reset_index(drop=True)

H = np.stack(d.pop("hours_sale").values).astype(np.float32)
S = np.stack(d.pop("hours_stock_status").values).astype(np.int8)
np.savez_compressed(f"{OUT}/hourly.npz", H=H, S=S)
d["dow"] = d.dt.dt.day_of_week
d["censored"] = (d.stock_hour6_22_cnt > 0).astype(np.int8)
d.to_parquet(f"{OUT}/daily.parquet")

Sw = S[:, W]
lines = [
    f"rows {len(d):,}  series {d.groupby(['store_id','product_id']).ngroups:,}  "
    f"days train {d[d.split=='train'].dt.nunique()} eval {d[d.split=='eval'].dt.nunique()}",
    f"cities {d.city_id.nunique()} stores {d.store_id.nunique()} products {d.product_id.nunique()} cat1 {d.first_category_id.nunique()}",
    f"hourly sum == sale_amount: {np.allclose(H.sum(1), d.sale_amount)}",
    f"stock_hour6_22_cnt == sum(status[6:22]): {(Sw.sum(1) == d.stock_hour6_22_cnt).all()}",
    f"share of sales in stockout-flagged hours: {H[S == 1].sum() / H.sum():.3%}  (status 1 = out of stock)",
    f"censored days (any stockout 6-21h): {d.censored.mean():.1%}  full-day stockout: {(d.stock_hour6_22_cnt == 16).mean():.1%}",
    f"zero-sale days: {(d.sale_amount == 0).mean():.1%}",
    f"censored day mean sales {d.sale_amount[d.censored == 1].mean():.3f} vs uncensored {d.sale_amount[d.censored == 0].mean():.3f}",
    "hourly sales share 0-23h: " + " ".join(f"{x:.3f}" for x in H.sum(0) / H.sum()),
    "hourly stockout rate 0-23h: " + " ".join(f"{x:.2f}" for x in S.mean(0)),
    "",
    "stockout-day rate by city:\n" + d.groupby("city_id").censored.mean().round(3).to_string(),
    "",
    "stockout-day rate by stockout hours (cnt distribution):\n" + d.stock_hour6_22_cnt.value_counts(normalize=True).sort_index().round(3).to_string(),
    "",
    "covariates:\n" + d[["discount", "holiday_flag", "activity_flag", "precpt", "avg_temperature"]].describe().round(3).to_string(),
]
open(f"{OUT}/audit.txt", "w").write("\n".join(lines))
print("\n".join(lines[:8]))
