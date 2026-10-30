"""Paper exhibits -> out/figures/*.pdf (vector, for the manuscript) + *.png (preview).
Fig 1 architecture | Fig 2 cost-service frontier | Fig 3 gap closed across scenarios | Fig 4 diagnostics.
One color per entity in every figure (validated categorical slots 1-4); every series also has a direct label,
marker and line style, so nothing relies on color alone (aqua/yellow are < 3:1 on the surface)."""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from config import OUT

FIG = f"{OUT}/figures"; os.makedirs(FIG, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"   # white: figures sit on a white journal page
REC, CENS, ORACLE, BASE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # recovered / no recovery / oracle / SSA-naive
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "legend.frameon": False, "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def panel(ax, letter):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, color=INK)


# ---------------- Fig 1: architecture
def box(ax, x, y, w, h, title, sub, fc="#eef3fb", ec=REC):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontweight="bold", fontsize=9, color=INK)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7, color=INK2)


def arrow(ax, p, q, text=None, style="-|>", ls="-", color=INK2, rad=0.0, tx=0.08):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, lw=1.1, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2, text, fontsize=7, color=INK2, va="center", style="italic")


fig, ax = plt.subplots(figsize=(7.2, 6.6)); ax.set_xlim(-0.15, 10.15); ax.set_ylim(1.2, 10.7); ax.axis("off")
W, H, X = 4.9, 0.95, 2.55
layers = [("Layer 1 - Data & censoring mask", "hourly sales & stockout labels, promo, weather"),
          ("Layer 2 - M1 Latent-demand recovery", "in-stock hourly profile scaled over stocked-out hours"),
          ("Layer 3 - M2 Probabilistic forecaster", "one global LightGBM quantile model, day-ahead"),
          ("Layer 4 - M3 Decision layer", "critical fractile q* = F^-1(cu/(cu+co)), shelf-life adjusted"),
          ("Layer 5 - Simulation harness", "FIFO perishable inventory, scored on true demand"),
          ("Layer 6 - M4 Diagnostics", "lost-demand accounting, stockout patterns, SHAP")]
contracts = ["daily table + censoring mask", "demand, recovery flag, imputed share", "predictive quantiles",
             "order quantity q*", "cost, waste, fill rate, bias"]
ys = [9.2 - i * 1.55 for i in range(6)]
for (t, s), y in zip(layers, ys):
    box(ax, X, y, W, H, t, s)
for i, c in enumerate(contracts):
    arrow(ax, (X + W / 2, ys[i]), (X + W / 2, ys[i + 1] + H), c)
box(ax, 0.0, 5.3, 2.1, 1.7, "Layer 0 - Config", "cu, co, shelf life L,\nservice target,\nrecovery on/off", fc="#f4f3f0", ec=INK2)
arrow(ax, (2.1, 6.6), (X, ys[1] + H / 2), color=INK2)      # recovery on/off -> M1
arrow(ax, (2.1, 5.6), (X, ys[3] + H / 2), color=INK2)      # cost & service -> M3
box(ax, 7.8, 4.3, 2.2, 1.7, "Store", "order q* -> shelf stock\n-> sales capped by\nstock (censored)", fc="#fdf0ea", ec=CENS)
arrow(ax, (X + W, ys[3] + H / 2), (7.8, 4.9), color=CENS)
arrow(ax, (8.9, 6.0), (X + W, ys[0] + H / 2), color=CENS, ls="--", rad=0.25)
ax.text(8.9, 4.05, "dashed: re-censoring loop,\ncensored sales feed\nthe next forecast", fontsize=7, color=INK2,
        style="italic", ha="center", va="top")
ax.text(5.0, 10.45, "Seven swappable layers, one flat (store, SKU, day) table between each", ha="center", fontsize=9, color=INK)
save(fig, "fig1_architecture")

# ---------------- Fig 2: cost-service frontier (semi-synthetic world, known truth, L=1, all 50K series)
k = pd.read_csv(f"{OUT}/semi_kpi.csv"); k = k[k.L == 1].copy()
k["norm_cost"] = k.tau * (1 - k.fill_rate) + (1 - k.tau) * k.waste_rate * (1 + k.order_bias)   # cu + co = 1
b = pd.read_csv(f"{OUT}/baseline_kpi.csv"); b = b[b.truth == "semi-synthetic"].copy()
b["norm_cost"] = b.tau * (1 - b.fill_rate) + (1 - b.tau) * b.waste_rate * (1 + b.order_bias)
POL = [("recovered quantile (proposed)", "Recovered + quantile (proposed)", REC, "o", "-"),
       ("censored quantile (no recovery)", "Censored sales + quantile", CENS, "s", "-"),
       ("oracle quantile (true-demand history)", "Oracle (true demand)", ORACLE, "^", "--")]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.3))
for key, lab, c, m, ls in POL:
    s = k[k.policy == key].sort_values("tau")
    axs[0].plot(s.fill_rate, s.waste_rate, ls, color=c, lw=2, marker=m, ms=5, mec=SURF, mew=1)
    axs[1].plot(s.tau, s.norm_cost, ls, color=c, lw=2, marker=m, ms=5, mec=SURF, mew=1)
ssa = b[b.policy == "SSA on censored, point (authors)"]
axs[0].plot(ssa.fill_rate.iloc[0], ssa.waste_rate.iloc[0], "D", color=BASE, ms=7, mec=INK2, mew=0.8)
axs[0].annotate("SSA point forecast\n(dataset authors)", (ssa.fill_rate.iloc[0], ssa.waste_rate.iloc[0]),
                xytext=(0.69, 0.30), fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
s = ssa.sort_values("tau"); axs[1].plot(s.tau, s.norm_cost, ":", color=BASE, lw=2, marker="D", ms=5, mec=INK2, mew=0.6)
# direct labels at the right end of each line
for key, lab, c, m, ls in POL:
    s = k[k.policy == key].sort_values("tau")
    x, y, ha = {"recovered quantile (proposed)": (s.fill_rate.iloc[-1] - 0.012, s.waste_rate.iloc[-1] + 0.012, "right"),
                "oracle quantile (true-demand history)": (s.fill_rate.iloc[-1] + 0.008, s.waste_rate.iloc[-1] - 0.03, "left"),
                "censored quantile (no recovery)": (0.715, 0.077, "left")}[key]
    axs[0].text(x, y, lab.split(" (")[0].replace(" + quantile", ""), fontsize=7.5, color=INK, va="center", ha=ha)
axs[0].set(xlabel="Fill rate (share of true demand served)", ylabel="Waste rate (spoiled / ordered)", xlim=(0.65, 1.08))
axs[0].set_title("Waste needed for a given service level", fontsize=9, color=INK, loc="left")
axs[1].set(xlabel="Critical fractile  tau = cu / (cu + co)", ylabel="Expected cost per unit demand (cu + co = 1)")
axs[1].set_title("Cost across planner cost settings", fontsize=9, color=INK, loc="left")
handles = [plt.Line2D([], [], color=c, lw=2, marker=m, ls=ls, label=lab) for _, lab, c, m, ls in POL] + \
          [plt.Line2D([], [], color=BASE, lw=2, marker="D", ls=":", label="SSA point forecast (authors)")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
panel(axs[0], "a"); panel(axs[1], "b"); fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, "fig2_frontier")

# ---------------- Fig 3: share of censored->oracle cost gap closed, tau = 0.8, across all checks
rows = []
ci = pd.read_csv(f"{OUT}/confidence_summary.csv"); ci = ci[ci.tau == 0.8].set_index("truth")
for t, lab in [("global", "Global-profile truth"), ("donor", "Donor truth (independent of M1)")]:
    r = ci.loc[t]
    rows.append(("Main (3 seeds, 95% CI)", lab, r.gap_closed_mean, r.gap_closed_ci_lo, r.gap_closed_ci_hi,
                 r.cost_change_mean, r.cost_change_ci_lo, r.cost_change_ci_hi))
rb = pd.read_csv(f"{OUT}/robustness.csv")
for _, r in rb.iterrows():
    rows.append(("Robustness (10K series)", f"{r.truth} truth, k={r.k} ({r.stockout_day_rate:.0%} stockout days)",
                 r["gap_closed_tau0.8"], np.nan, np.nan, r["cost_chg_tau0.8"], np.nan, np.nan))
mp = pd.read_csv(f"{OUT}/multiperiod_kpi.csv"); mp = mp[(mp.tau == 0.8) & (mp.truth == "global")]
for L in [1, 2, 3]:
    c = mp[mp.L == L].set_index("policy").cost_per_unit_demand
    for adj, sfx in [("", ""), (", shelf-life-adjusted", ", adjusted rule")]:
        if L == 1 and adj:
            continue
        cc, rr, oo = c[f"censored quantile{adj or ' (no recovery)'}"], c[f"recovered quantile{adj or ' (proposed)'}"], \
                     c[f"oracle quantile{adj or ' (true-demand history)'}"]
        rows.append(("Shelf life, 30-day run", f"L = {L} day{'s' if L > 1 else ''}{sfx}", (cc - rr) / (cc - oo), np.nan, np.nan,
                     rr / cc - 1, np.nan, np.nan))
m5 = pd.read_csv(f"{OUT}/m5_headline.csv"); m5 = m5[m5.tau == 0.8]
for _, r in m5.iterrows():
    rows.append(("Transfer: M5 (daily, Tobit M1)", f"k={r.k}", r.gap_closed_to_oracle, np.nan, np.nan, r.rec_vs_cens_cost, np.nan, np.nan))
g = pd.DataFrame(rows, columns=["group", "label", "gap", "glo", "ghi", "chg", "clo", "chi"])
g.to_csv(f"{FIG}/fig3_data.csv", index=False)

fig, axs = plt.subplots(1, 2, figsize=(7.4, 5.6), sharey=True, gridspec_kw=dict(wspace=0.08))
y, yt, yl, gy = 0, [], [], []
for grp, s in g.groupby("group", sort=False):
    gy.append((grp, y)); y += 1
    for _, r in s.iterrows():
        yt.append(y); yl.append(r.label); r_y = y
        for ax, v, lo, hi in [(axs[0], r.gap, r.glo, r.ghi), (axs[1], -r.chg, -r.chi, -r.clo)]:
            if not np.isnan(lo):
                ax.plot([lo, hi], [r_y, r_y], color=REC, lw=2.2, solid_capstyle="round")
            ax.plot(v, r_y, "o", color=REC, ms=6, mec=SURF, mew=1.2)
            ax.text(v + (0.012 if ax is axs[0] else 0.012), r_y - 0.32, f"{v:.0%}", fontsize=7, color=INK2)
        y += 1
    y += 0.4
axs[0].invert_yaxis()                          # shared y: invert once
for ax in axs:
    ax.grid(axis="y", visible=False)
for grp, gy_ in gy:
    axs[0].text(-0.02, gy_, grp, transform=axs[0].get_yaxis_transform(), ha="right", fontweight="bold", fontsize=8, color=INK)
axs[0].set_yticks(yt, yl, fontsize=7.5); axs[0].tick_params(axis="y", length=0)
axs[0].set(xlim=(0, 1.08), xlabel="Share of censored-to-oracle cost gap closed")
axs[1].set(xlim=(0, 0.8), xlabel="Cost reduction vs. same system, recovery off")
axs[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); axs[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
axs[0].set_title("Gap to full-visibility oracle closed", fontsize=9, loc="left", color=INK)
axs[1].set_title("Cost reduction from recovery", fontsize=9, loc="left", color=INK)
fig.text(0.5, -0.01, "All rows: critical fractile tau = 0.8 (cu:co = 4:1), scored on known true demand. Main rows: 95% series-bootstrap CI (narrower than the marker).", ha="center", fontsize=7.5, color=INK2)
panel(axs[0], "a"); panel(axs[1], "b")
save(fig, "fig3_gap_closed")

# ---------------- Fig 4: diagnostics (real data)
dg = pd.read_csv(f"{OUT}/diagnostics.csv")
fig, axs = plt.subplots(1, 3, figsize=(7.6, 3.4), gridspec_kw=dict(width_ratios=[1.15, 1, 1.15], wspace=0.75))
p = dg[(dg.dimension == "stockout_pattern") & (dg.level != "none")].sort_values("share_of_all_lost")
axs[0].barh(p.level, p.share_of_all_lost, color=REC, height=0.62, edgecolor=SURF, linewidth=2)
for i, v in enumerate(p.share_of_all_lost):
    axs[0].text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=7, color=INK2)
axs[0].set(xlabel="Share of all lost demand", xlim=(0, 0.8)); axs[0].grid(axis="y", visible=False)
axs[0].set_title("Where lost demand comes from", fontsize=9, loc="left", color=INK)
axs[0].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0)); axs[0].tick_params(axis="y", labelsize=7.5)

c = dg[dg.dimension == "city_id"].sort_values("lost_rate")
axs[1].barh("City " + c.level.astype(str), c.lost_rate, color=REC, height=0.62, edgecolor=SURF, linewidth=1)
axs[1].set(xlabel="Lost demand / latent demand", xlim=(0, 0.32)); axs[1].grid(axis="y", visible=False)
axs[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0)); axs[1].tick_params(axis="y", labelsize=6.5)
axs[1].set_title("Lost-demand rate by city", fontsize=9, loc="left", color=INK)

sa, sd = pd.read_csv(f"{OUT}/shap_dependence_sales.csv"), pd.read_csv(f"{OUT}/shap_dependence_demand.csv")
def eff(df):
    f = lambda feat, bins: df[(df.feature == feat) & df.bin.isin(bins)].shap_mean.mean()
    return [f("holiday_flag", [1]) - f("holiday_flag", [0]), f("discount", [0]) - f("discount", [3, 4]),
            f("dow", [5, 6]) - f("dow", [0, 1, 2, 3, 4]), f("cens_rate7", [4]) - f("cens_rate7", [0])]
labs = ["Holiday", "Deep discount", "Weekend", "Recent\nstockouts"]
ea, ed = eff(sa), eff(sd); yy = np.arange(len(labs))
axs[2].barh(yy - 0.19, ea, 0.36, color=CENS, edgecolor=SURF, linewidth=1.5, label="Trained on censored sales")
axs[2].barh(yy + 0.19, ed, 0.36, color=REC, edgecolor=SURF, linewidth=1.5, label="Trained on recovered demand")
axs[2].set_yticks(yy, labs, fontsize=7.5); axs[2].invert_yaxis(); axs[2].grid(axis="y", visible=False)
axs[2].axvline(0, color=INK2, lw=0.8)
axs[2].set(xlabel="SHAP effect on median forecast")
axs[2].set_title("Effects the censored model misreads", fontsize=9, loc="left", color=INK)
axs[2].legend(fontsize=6.8, loc="upper center", bbox_to_anchor=(0.4, -0.2))
for a, l in zip(axs, "abc"):
    panel(a, l)
save(fig, "fig4_diagnostics")
print("saved:", sorted(os.listdir(FIG)))
