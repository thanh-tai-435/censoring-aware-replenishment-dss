"""Phase 7 / Layer 6 (M4): where availability leaks. Lost demand = recovered demand - observed sales."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import OUT, W

d = pd.read_parquet(f"{OUT}/recovered.parquet")
S = np.load(f"{OUT}/hourly.npz")["S"][:, W]
d["lost"] = d.demand - d.sale_amount
first = np.where(S.any(1), S.argmax(1) + W[0], -1)
d["stockout_start"] = pd.cut(first, [-2, -0.5, 9.5, 13.5, 17.5, 22], labels=["none", "6-9h", "10-13h", "14-17h", "18-21h"])
# stockout pattern: opening gap (empty at 6h, restocked later) is a delivery-timing issue, not an order-size one
M = S.astype(bool); inn = ~M; idx = np.arange(len(W))
fi = np.where(inn.any(1), inn.argmax(1), 99)
li = np.where(inn.any(1), len(W) - 1 - inn[:, ::-1].argmax(1), -1)
gap = (M & (idx >= fi[:, None]) & (idx <= li[:, None])).any(1)
d["stockout_pattern"] = np.select(
    [~M.any(1), fi == 99, M[:, 0] & M[:, -1], M[:, 0], M[:, -1] & ~gap, M[:, -1] & gap],
    ["none", "full day", "opening gap + sold out", "opening gap only", "sold out", "sold out + midday gap"],
    "midday gap only")
d["discount_bin"] = pd.cut(d.discount, [0, 0.8, 0.9, 0.99, 2], labels=["<0.8", "0.8-0.9", "0.9-0.99", "no discount"])
d["rain_bin"] = pd.cut(d.precpt, [-1, 1, 5, 15, 100], labels=["dry", "light", "moderate", "heavy"])

TOTAL = d.lost.sum()
out = []
for col in ["city_id", "first_category_id", "activity_flag", "holiday_flag", "discount_bin", "rain_bin", "dow", "stockout_start", "stockout_pattern"]:
    g = d.groupby(col, observed=True).agg(demand=("demand", "sum"), lost=("lost", "sum"), days=("lost", "size"),
                                          stockout_day_rate=("censored", "mean"))
    g["lost_rate"] = g.lost / g.demand                  # share of this segment's demand not served
    g["share_of_all_lost"] = g.lost / TOTAL             # where the lost units are
    out.append(g.reset_index().rename(columns={col: "level"}).assign(dimension=col))
diag = pd.concat(out)[["dimension", "level", "days", "stockout_day_rate", "lost_rate", "share_of_all_lost"]]
diag.round(4).to_csv(f"{OUT}/diagnostics.csv", index=False)
pd.set_option("display.width", 200)
print(f"overall lost rate {TOTAL / d.demand.sum():.2%} of latent demand")
for dim, g in diag.groupby("dimension", sort=False):
    print(f"\n-- {dim}"); print(g.drop(columns="dimension").sort_values("share_of_all_lost", ascending=False).head(10).round(3).to_string(index=False))

# availability-leak concentration: what share of series accounts for 50% / 80% of lost demand
s = d.groupby(["store_id", "product_id"]).lost.sum().sort_values(ascending=False)
c = s.cumsum() / s.sum()
print(f"\nseries holding 50% of lost demand: {(c < 0.5).mean():.1%}; 80%: {(c < 0.8).mean():.1%}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, dim in zip(ax, ["city_id", "first_category_id"]):
    g = diag[diag.dimension == dim].sort_values("share_of_all_lost", ascending=False).head(15)
    a.bar(g.level.astype(str), g.share_of_all_lost)
    a2 = a.twinx(); a2.plot(g.level.astype(str), g.lost_rate, "o", color="C3")
    a.set(title=f"Lost demand by {dim}", ylabel="share of all lost units"); a2.set_ylabel("lost rate (red)")
fig.tight_layout(); fig.savefig(f"{OUT}/diagnostics.png", dpi=130)
