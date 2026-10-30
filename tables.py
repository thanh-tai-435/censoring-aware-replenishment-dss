"""Manuscript numbers and tables, generated from out/*.csv so no figure is ever typed by hand.
Writes paper/numbers.tex (\newcommand macros used in the text) and paper/tables/*.tex.
Rerun after any experiment; then recompile the manuscript."""
import os
import numpy as np, pandas as pd
from config import OUT
from p6_semisynthetic import load, build_truth, censored_world, retailer
from config import W

P = "paper"; os.makedirs(f"{P}/tables", exist_ok=True)
pct = lambda x, d=1: f"{100 * x:.{d}f}"
spct = lambda x, d=1: f"{100 * x:+.{d}f}".replace("-", "$-$")          # signed, typeset minus
N = {}                                                              # macro name -> text


def ncost(df):
    """Expected cost per unit true demand with cu + co = 1 (cu = tau)."""
    return df.tau * (1 - df.fill_rate) + (1 - df.tau) * df.waste_rate * (1 + df.order_bias)


# ---- data facts (real FreshRetailNet)
d = pd.read_parquet(f"{OUT}/recovered.parquet", columns=["censored", "sale_amount", "demand"])
N["StockoutDayShare"] = pct(d.censored.mean())
N["LostShare"] = pct(1 - d.sale_amount.sum() / d.demand.sum())

# ---- recovery validity
rv = pd.read_csv(f"{OUT}/recovery_validation.csv").set_index(["method", "severity"])
naive, m1 = rv.loc[("naive (observed sales)", "all")], rv.loc[("series profile k=5", "all")]
N["TransNaiveBias"], N["TransMOneBias"] = spct(naive.bias), spct(m1.bias)
N["TransNaiveWAPE"], N["TransMOneWAPE"] = pct(naive.WAPE), pct(m1.WAPE)

# semi-synthetic recovery error for both truth constructions + the k=1.0 collapse (10K-series sample, seed 0)
dd, H, S = load(10_000)
rec_rows = []
for mode in ["global", "donor"]:
    _, rec, _ = censored_world(dd, H, build_truth(H, S, mode, np.random.default_rng(0)), 1.2)
    r = rec.set_index(["estimate", "days"])
    for est in ["censored sales", "M1 recovered"]:
        rec_rows.append((f"Semi-synthetic, {mode} truth", est[0].upper() + est[1:], r.loc[(est, "censored days"), "bias"], r.loc[(est, "censored days"), "WAPE"]))
obs_h, mask = retailer(build_truth(H, S, "global"), 1.0)
true_h = build_truth(H, S, "global")
N["CollapseStockoutDays"] = pct((mask.sum(2) > 0).mean(), 0)
N["CollapseObservedShare"] = pct(obs_h.sum() / true_h.sum(), 0)

t1 = [("Real data, transplanted real masks", "Censored sales", naive.bias, naive.WAPE),
      ("Real data, transplanted real masks", "M1 recovered", m1.bias, m1.WAPE)] + rec_rows
with open(f"{P}/tables/tab_recovery.tex", "w") as f:
    f.write("\\begin{tabular*}{\\tblwidth}{@{}LLRR@{}}\n\\toprule\nGround truth & Estimate & Bias (\\%) & WAPE (\\%)\\\\\n\\midrule\n")
    for i, (g, e, b, w) in enumerate(t1):
        f.write(f"{g if i % 2 == 0 else ''} & {e} & {spct(b)} & {pct(w)}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular*}\n")

# ---- main decision table: semi-synthetic, all 50K series, tau = 0.8, L = 1
k = pd.read_csv(f"{OUT}/semi_kpi.csv"); k = k[(k.L == 1) & (k.tau == 0.8)]
b = pd.read_csv(f"{OUT}/baseline_kpi.csv"); b = b[(b.truth == "semi-synthetic") & (b.tau == 0.8)]
rows = [("Moving average of censored sales (status quo)", k[k.policy == "naive ma7 of censored sales"]),
        ("SSA point forecast, censored sales (dataset authors)", b[b.policy == "SSA on censored, point (authors)"]),
        ("Quantile forecast, censored sales (recovery off)", k[k.policy == "censored quantile (no recovery)"]),
        ("Quantile forecast, recovered demand (proposed)", k[k.policy == "recovered quantile (proposed)"]),
        ("Quantile forecast, true demand (oracle)", k[k.policy == "oracle quantile (true-demand history)"])]
cost = {}
with open(f"{P}/tables/tab_main.tex", "w") as f:
    f.write("\\begin{tabular*}{\\tblwidth}{@{}LRRRR@{}}\n\\toprule\nPolicy & Cost & Fill rate (\\%) & Waste rate (\\%) & Order bias (\\%)\\\\\n\\midrule\n")
    for lab, r in rows:
        r = r.iloc[0]; c = ncost(r); cost[lab] = c
        name = f"\\textbf{{{lab}}}" if "proposed" in lab else lab
        f.write(f"{name} & {c:.3f} & {pct(r.fill_rate)} & {pct(r.waste_rate)} & {spct(r.order_bias)}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular*}\n")
labs = [l for l, _ in rows]
N["MainCostVsSSA"] = pct(1 - cost[labs[3]] / cost[labs[1]], 0)
N["MainCostVsNaive"] = pct(1 - cost[labs[3]] / cost[labs[0]], 0)
N["MainFillRec"] = pct(k[k.policy == "recovered quantile (proposed)"].fill_rate.iloc[0])
N["MainFillCens"] = pct(k[k.policy == "censored quantile (no recovery)"].fill_rate.iloc[0])
kk = pd.read_csv(f"{OUT}/semi_kpi.csv"); kk = kk[kk.L == 1]
N["MainOrderBiasCens"] = spct(k[k.policy == "censored quantile (no recovery)"].order_bias.iloc[0])
kr = k[k.policy == "recovered quantile (proposed)"].iloc[0]
N["MainOrderBiasRec"], N["MainWasteShareRec"], N["MainLostShareRec"] = spct(kr.order_bias), pct(kr.waste_rate * (1 + kr.order_bias)), pct(1 - kr.fill_rate)
N["OracleCost"] = f"{cost[labs[4]]:.3f}"
semi_sales = [b_ for g_, e_, b_, w_ in rec_rows if e_ == "Censored sales"]
N["SemiSalesBiasHi"], N["SemiSalesBiasLo"] = spct(max(semi_sales), 0), spct(min(semi_sales), 0)
N["CensFillCeiling"] = pct(kk[kk.policy == "censored quantile (no recovery)"].fill_rate.max())

# ---- uncertainty table (3 seeds x series bootstrap)
ci = pd.read_csv(f"{OUT}/confidence_summary.csv")
with open(f"{P}/tables/tab_uncertainty.tex", "w") as f:
    f.write("\\begin{tabular*}{\\tblwidth}{@{}LRRRRR@{}}\n\\toprule\n"
            "Truth & $\\tau$ & M1 bias (\\%) & Cost change (\\%) [95\\% CI] & Gap closed (\\%) [95\\% CI] & Series better (\\%)\\\\\n\\midrule\n")
    for _, r in ci.sort_values(["truth", "tau"], ascending=[False, True]).iterrows():
        f.write(f"{r.truth} & {r.tau:.2f} & {spct(r.M1_bias)} & {spct(r.cost_change_mean)} [{spct(r.cost_change_ci_lo)}, {spct(r.cost_change_ci_hi)}] & "
                f"{pct(r.gap_closed_mean)} [{pct(r.gap_closed_ci_lo)}, {pct(r.gap_closed_ci_hi)}] & {pct(r.rec_better_share)}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular*}\n")
g, dn = ci[(ci.truth == "global") & (ci.tau == 0.8)].iloc[0], ci[(ci.truth == "donor") & (ci.tau == 0.8)].iloc[0]
N["CostCut"], N["CostCutLo"], N["CostCutHi"] = pct(-g.cost_change_mean, 0), pct(-g.cost_change_ci_hi, 0), pct(-g.cost_change_ci_lo, 0)
N["CostCutDonor"] = pct(-dn.cost_change_mean, 0)
N["GapGlobal"], N["GapDonor"] = pct(g.gap_closed_mean, 0), pct(dn.gap_closed_mean, 0)
N["GapGlobalLo"], N["GapGlobalHi"] = pct(g.gap_closed_ci_lo, 0), pct(g.gap_closed_ci_hi, 0)
N["GapDonorLo"], N["GapDonorHi"] = pct(dn.gap_closed_ci_lo, 0), pct(dn.gap_closed_ci_hi, 0)
N["MOneBiasGlobal"], N["MOneBiasDonor"] = spct(g.M1_bias), spct(dn.M1_bias)
N["SeriesBetter"] = pct(min(g.rec_better_share, dn.rec_better_share), 0)

# ---- robustness range, transfer, shelf life
rb = pd.read_csv(f"{OUT}/robustness.csv")
N["RobGapMin"], N["RobGapMax"] = pct(rb["gap_closed_tau0.8"].min(), 0), pct(rb["gap_closed_tau0.8"].max(), 0)
m5 = pd.read_csv(f"{OUT}/m5_headline.csv"); m5 = m5[m5.tau == 0.8]
N["MFiveGapMin"], N["MFiveGapMax"] = pct(m5.gap_closed_to_oracle.min(), 0), pct(m5.gap_closed_to_oracle.max(), 0)
mp = pd.read_csv(f"{OUT}/multiperiod_kpi.csv"); mp = mp[(mp.tau == 0.8) & (mp.truth == "global")]
POLS = [("censored quantile (no recovery)", "Censored, single-period rule"), ("recovered quantile (proposed)", "Recovered, single-period rule"),
        ("oracle quantile (true-demand history)", "Oracle, single-period rule"),
        ("censored quantile, shelf-life-adjusted", "Censored, shelf-life-adjusted rule"),
        ("recovered quantile, shelf-life-adjusted", "Recovered, shelf-life-adjusted rule"),
        ("oracle quantile, shelf-life-adjusted", "Oracle, shelf-life-adjusted rule")]
mpc = mp.pivot_table(index="policy", columns="L", values="cost_per_unit_demand")
with open(f"{P}/tables/tab_shelflife.tex", "w") as f:
    f.write("\\begin{tabular*}{\\tblwidth}{@{}LRRR@{}}\n\\toprule\nPolicy & $L=1$ & $L=2$ & $L=3$\\\\\n\\midrule\n")
    for key, lab in POLS:
        f.write(f"{lab} & " + " & ".join(f"{mpc.loc[key, L] / 5:.3f}" for L in [1, 2, 3]) + "\\\\\n")   # /5: cu+co=1 at 4:1
    f.write("\\bottomrule\n\\end{tabular*}\n")
N["ShelfAdjCutThree"] = pct(1 - mpc.loc["recovered quantile, shelf-life-adjusted", 3] / mpc.loc["recovered quantile (proposed)", 3], 0)
N["ShelfCombinedCutThree"] = pct(1 - mpc.loc["recovered quantile, shelf-life-adjusted", 3] / mpc.loc["censored quantile (no recovery)", 3], 0)

# ---- exact DP benchmark for the shelf-life rule (p15)
dp = pd.read_csv(f"{OUT}/shelflife_dp.csv")
dp["closed"] = (dp.single - dp.tauL) / (dp.single - dp.opt)          # share of the single-period rule's excess removed
with open(f"{P}/tables/tab_dp.tex", "w") as f:
    f.write("\\begin{tabular*}{\\tblwidth}{@{}LLRRRR@{}}\n\\toprule\n"
            "$L$ & $\\tau$ & Single-period rule & Shelf-life rule $\\tau_L$ & Best base-stock & Excess removed by $\\tau_L$\\\\\n\\midrule\n")
    for (L, tau), g_ in dp.groupby(["L", "tau"]):
        f.write(f"{L} & {tau:.2f} & {pct(g_.gap_single.mean())} & {pct(g_.gap_tauL.mean())} & {pct(g_.gap_best_base.mean())} & {pct(g_.closed.mean(), 0)}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular*}\n")
N["DPBestBaseMax"] = pct(dp.gap_best_base.max())
hi_ = dp[dp.tau >= 0.8]
N["DPTauLHighMean"], N["DPTauLHighMax"] = pct(hi_.gap_tauL.mean()), pct(hi_.gap_tauL.max(), 0)
N["DPTauLLowMax"] = pct(dp[dp.tau == 0.5].gap_tauL.max(), 0)
N["DPClosedMin"], N["DPClosedMax"], N["DPClosedMedian"] = pct(dp.closed.min(), 0), pct(dp.closed.max(), 0), pct(dp.closed.median(), 0)
N["DPSingleMean"] = pct(dp.gap_single.mean(), 0)
N["DPTauLHighCVMax"] = pct(dp[(dp.tau >= 0.8) & (dp.cv == 1.0)].gap_tauL.max())

# ---- authors' accuracy protocol
fp = pd.read_csv(f"{OUT}/baseline_frn_protocol.csv")
with open(f"{P}/tables/tab_protocol.tex", "w") as f:
    f.write("\\begin{tabular*}{\\tblwidth}{@{}LRR@{}}\n\\toprule\nForecast & WAPE (\\%) & WPE (\\%)\\\\\n\\midrule\n")
    for _, r in fp.iterrows():
        f.write(f"{r.forecast} & {pct(r.wape)} & {spct(r.wpe)}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular*}\n")
N["SSAWPE"] = spct(fp.wpe.iloc[0])
N["SSAWAPE"], N["SSAWPEtwo"] = pct(fp.wape.iloc[0], 2), spct(fp.wpe.iloc[0], 2)

# ---- diagnostics + SHAP
dg = pd.read_csv(f"{OUT}/diagnostics.csv").set_index(["dimension", "level"])
N["SoldOutShare"] = pct(dg.loc[("stockout_pattern", "sold out"), "share_of_all_lost"] + dg.loc[("stockout_pattern", "sold out + midday gap"), "share_of_all_lost"], 0)
N["OpeningGapShare"] = pct(dg.loc[("stockout_pattern", "opening gap only"), "share_of_all_lost"] + dg.loc[("stockout_pattern", "opening gap + sold out"), "share_of_all_lost"], 0)
N["FullDayShare"] = pct(dg.loc[("stockout_pattern", "full day"), "share_of_all_lost"], 0)
sa, sd = pd.read_csv(f"{OUT}/shap_dependence_sales.csv"), pd.read_csv(f"{OUT}/shap_dependence_demand.csv")
eff = lambda df, ft, hi, lo: df[(df.feature == ft) & df.bin.isin(hi)].shap_mean.mean() - df[(df.feature == ft) & df.bin.isin(lo)].shap_mean.mean()
N["HolidayUnder"] = pct(1 - eff(sa, "holiday_flag", [1], [0]) / eff(sd, "holiday_flag", [1], [0]), 0)
N["DiscountUnder"] = pct(1 - eff(sa, "discount", [0], [3, 4]) / eff(sd, "discount", [0], [3, 4]), 0)
sa8, sd8 = pd.read_csv(f"{OUT}/shap_dependence_sales_q0.8.csv"), pd.read_csv(f"{OUT}/shap_dependence_demand_q0.8.csv")
N["HolidayUnderQ"] = pct(1 - eff(sa8, "holiday_flag", [1], [0]) / eff(sd8, "holiday_flag", [1], [0]), 0)
N["DiscountUnderQ"] = pct(1 - eff(sa8, "discount", [0], [3, 4]) / eff(sd8, "discount", [0], [3, 4]), 0)

# ---- method facts: profile-shrinkage sensitivity and retailer calibration
rva = pd.read_csv(f"{OUT}/recovery_validation.csv"); rva = rva[rva.severity == "all"].set_index("method").WAPE
ser = rva[rva.index.str.startswith("series profile")]
N["KappaWAPEMin"], N["KappaWAPEMax"] = pct(ser.min()), pct(ser.max())
N["StoreProfileWAPE"], N["GlobalProfileWAPE"] = pct(rva["store profile"]), pct(rva["global profile"])
rb12 = pd.read_csv(f"{OUT}/robustness.csv").set_index(["truth", "k"])
N["RetailerStockoutDays"] = pct(rb12.loc[("global", 1.2), "stockout_day_rate"], 0)

# ---- extra numbers for the Results discussion
for sev, nm in [("1-4h out", "Short"), ("11-15h out", "Long"), ("full day", "Full")]:
    N[f"Sev{nm}Bias"], N[f"Sev{nm}WAPE"] = spct(rv.loc[("series profile k=5", sev), "bias"]), pct(rv.loc[("series profile k=5", sev), "WAPE"])
sacc = pd.read_csv(f"{OUT}/semi_accuracy.csv").set_index("model")
for m, nm in [("cens", "Cens"), ("rec", "Rec"), ("oracle", "Oracle")]:
    N[f"Semi{nm}WAPE"], N[f"Semi{nm}Bias"] = pct(sacc.loc[m, "WAPE_q0.5"]), spct(sacc.loc[m, "bias_q0.5"])
    N[f"Semi{nm}Cover"] = pct(sacc.loc[m, "cover_0.8"], 0)
bb = pd.read_csv(f"{OUT}/baseline_kpi.csv"); bb = bb[(bb.truth == "semi-synthetic") & (bb.tau == 0.8)].set_index("policy")
N["SSATrueCost"] = f"{ncost(bb.loc['SSA on true demand, point']):.3f}"
N["SSATrueFill"] = pct(bb.loc["SSA on true demand, point", "fill_rate"])
N["RecCost"] = f"{cost[labs[3]]:.3f}"
N["CensCost"] = f"{cost[labs[2]]:.3f}"
hi_ = ci[ci.tau == 0.95].set_index("truth")
N["HighTauCut"] = pct(-hi_.loc["global", "cost_change_mean"], 0)
lo = ci[ci.tau == 0.5].set_index("truth")
N["LowTauCut"] = pct(-lo.loc["global", "cost_change_mean"], 1)
N["LowTauGapDonor"], N["LowTauGapGlobal"] = pct(lo.loc["donor", "gap_closed_mean"], 0), pct(lo.loc["global", "gap_closed_mean"], 0)
rbi = rb.set_index(["truth", "k"])
N["RobCutLowCens"], N["RobCutHighCens"] = pct(-rbi.loc[("global", 1.4), "cost_chg_tau0.8"], 0), pct(-rbi.loc[("global", 1.1), "cost_chg_tau0.8"], 0)
N["RobStockLow"], N["RobStockHigh"] = pct(rbi.loc[("global", 1.4), "stockout_day_rate"], 0), pct(rbi.loc[("global", 1.1), "stockout_day_rate"], 0)
N["EveningBias"] = spct(rbi.loc[("evening", 1.2), "M1_bias_cens_days"])
N["EveningGap"] = pct(rbi.loc[("evening", 1.2), "gap_closed_tau0.8"], 0)
mpf = mp.pivot_table(index="policy", columns="L", values="fill_rate")
mpw = mp.pivot_table(index="policy", columns="L", values="waste_rate")
N["ShelfFillSingle"] = pct(mpf.loc["recovered quantile (proposed)", 3])
N["ShelfFillAdj"] = pct(mpf.loc["recovered quantile, shelf-life-adjusted", 3])
N["ShelfWasteAdj"] = pct(mpw.loc["recovered quantile, shelf-life-adjusted", 3])
N["ShelfAdjCutTwo"] = pct(1 - mpc.loc["recovered quantile, shelf-life-adjusted", 2] / mpc.loc["recovered quantile (proposed)", 2], 0)
N["ShelfCensAdjCutThree"] = pct(1 - mpc.loc["censored quantile, shelf-life-adjusted", 3] / mpc.loc["censored quantile (no recovery)", 3], 0)
m5r = pd.read_csv(f"{OUT}/m5_recovery.csv")
N["MFiveBiasMin"], N["MFiveBiasMax"] = spct(m5r.M1daily_bias_cens_days.max()), spct(m5r.M1daily_bias_cens_days.min())
N["MFiveSalesBiasMin"], N["MFiveSalesBiasMax"] = spct(m5r.sales_bias_cens_days.max()), spct(m5r.sales_bias_cens_days.min())
m5a = pd.read_csv(f"{OUT}/m5_accuracy.csv").set_index(["k", "model"])
N["MFiveCoverCens"], N["MFiveCoverRec"], N["MFiveCoverOracle"] = (pct(m5a.loc[(1.2, m), "cover_0.8"], 0) for m in ["cens", "rec", "oracle"])
N["MFiveSeries"] = "1,601"
hd = pd.read_csv(f"{OUT}/headline.csv").set_index(["truth", "L", "tau"])
N["TruthACostChangeAbs"] = pct(abs(hd.loc[("A", 1, 0.8), "cost_change_P3_vs_P2"]))
fa = pd.read_csv(f"{OUT}/forecast_accuracy.csv").set_index(["truth", "model"])
N["TruthACoverSales"] = pct(fa.loc[("A uncensored days, D=sales", "sales"), "cover_0.5"])
N["TruthACoverDemand"] = pct(fa.loc[("A uncensored days, D=sales", "demand"), "cover_0.5"])
N["RealUncensDayMean"] = f"{d.sale_amount[d.censored == 0].mean():.3f}"
N["RealCensDayDemandMean"] = f"{d.demand[d.censored == 1].mean():.3f}"
for t, nm in [("sales", "Sales"), ("demand", "Demand")]:
    s = pd.read_csv(f"{OUT}/shap_{t}_q0.8.csv", index_col=0).iloc[:, 0]          # the quantile that sets the base-case order
    N[f"ShapCens{nm}"], N[f"ShapCens{nm}Rank"] = f"{s['cens_rate7']:.3f}", str(list(s.index).index("cens_rate7") + 1)
city = dg.loc["city_id"]
N["CityLostMin"], N["CityLostMax"] = pct(city.lost_rate.min()), pct(city.lost_rate.max())
dl = pd.read_parquet(f"{OUT}/recovered.parquet", columns=["store_id", "product_id", "sale_amount", "demand"])
sl = (dl.demand - dl.sale_amount).groupby([dl.store_id, dl.product_id]).sum().sort_values(ascending=False)
N["SeriesHalfLost"] = pct(((sl.cumsum() / sl.sum()) < 0.5).mean(), 0)

# ---- reviewer stress tests (p13): stress truth, recovery baselines, retailer policies, store-cluster CIs
rt = pd.read_csv(f"{OUT}/review_tests.csv")
r8 = rt[rt.tau == 0.8].set_index(["truth", "retailer", "recovery"])
r2 = pd.read_csv(f"{OUT}/review2_tests.csv"); r28 = r2[r2.tau == 0.8].set_index("scenario")
ro = pd.read_csv(f"{OUT}/review2_rolling.csv").set_index("tau")
tn = pd.read_csv(f"{OUT}/review_timesnet.csv"); tn8 = tn[tn.tau == 0.8].set_index(["truth", "recovery"])
row = lambda r, lo, hi, glo, ghi: (f"{pct(r.stockout_day_rate, 0)} & {spct(r.rec_bias_cens_days)} & {spct(r.cost_change)} [{spct(r[lo])}, {spct(r[hi])}] & "
                                   f"{pct(r.gap_closed, 0)} [{pct(r[glo], 0)}, {pct(r[ghi], 0)}] & {spct(r.day_change_min, 0)} to {spct(r.day_change_max, 0)}")
with open(f"{P}/tables/tab_review.tex", "w") as f:
    f.write("\\begin{tabular*}{\\tblwidth}{@{}LLRRRRR@{}}\n\\toprule\n"
            "Truth & Retailer & Stockout days & Rec.\\ bias & Cost change [CI] & Gap closed [CI] & Daily range\\\\\n\\midrule\n")
    for key, tr, rp in [(("global", "mean", "m1"), "Global", "Sales multiple"), (("donor", "mean", "m1"), "Donor", "Sales multiple"),
                        (("stress", "mean", "m1"), "Stress", "Sales multiple"), (("global", "safety", "m1"), "Global", "Safety stock"),
                        (("global", "adaptive", "m1"), "Global", "Adaptive")]:
        f.write(f"{tr} & {rp} & " + row(r8.loc[key], "cluster_ci_lo", "cluster_ci_hi", "gap_cluster_ci_lo", "gap_cluster_ci_hi") + "\\\\\n")
    for key, rp in [("lead_time_2d", "Lead time"), ("capacity_p75", "Capacity"), ("delivery_every_2d", "Delivery every 2 days"),
                    ("deferred_30", "Deferred demand"), ("substitution_30", "Substitution"), ("all_combined", "All five")]:
        f.write(f"Global & {rp} & " + row(r28.loc[key], "store_ci_lo", "store_ci_hi", "gap_store_ci_lo", "gap_store_ci_hi") + "\\\\\n")
    rp8, w90 = ro.loc[0.8], r28.loc["rolling_90"]
    f.write(f"Global & Rolling, 35 days$^{{a}}$ & {pct(r2[r2.scenario.str.startswith('rolling')].stockout_day_rate.mean(), 0)} & {spct(w90.rec_bias_cens_days)} & "
            f"{spct(rp8.cost_change)} [{spct(rp8.twoway_ci_lo)}, {spct(rp8.twoway_ci_hi)}] & {pct(rp8.gap_closed, 0)} [{pct(rp8.gap_twoway_ci_lo, 0)}, {pct(rp8.gap_twoway_ci_hi, 0)}] & "
            f"{spct(rp8.day_change_min, 0)} to {spct(rp8.day_change_max, 0)}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular*}\n")

# recovery methods on the same data, forecaster and simulator (global and stress truth, tau = 0.8)
REC = [("M1 (proposed)", "hour of stockout, profile", ("global", "mean", "m1"), ("stress", "mean", "m1")),
       ("Flat-profile scaling", "hour of stockout", ("global", "mean", "uniform"), ("stress", "mean", "uniform")),
       ("Daily Tobit", "censored days only", ("global", "mean", "tobit"), ("stress", "mean", "tobit")),
       ("Kaplan--Meier", "censored days only", "km_global", "km_stress"),
       ("TimesNet (dataset authors)$^{a}$", "missing hours", ("tn", "global", "timesnet"), ("tn", "stress", "timesnet")),
       ("M1, same subsample$^{a}$", "hour of stockout, profile", ("tn", "global", "m1"), ("tn", "stress", "m1"))]
def rec_row(key):
    if key[0] == "tn" and key[1:] not in tn8.index:              # run not available (see p16 notes)
        return None, "\\multicolumn{2}{c}{not run$^{b}$}"
    if key[0] == "tn":
        r = tn8.loc[key[1:]]; lo, hi = "gap_store_ci_lo", "gap_store_ci_hi"
    elif isinstance(key, tuple):
        r = r8.loc[key]; lo, hi = "gap_cluster_ci_lo", "gap_cluster_ci_hi"
    else:
        r = r28.loc[key]; lo, hi = "gap_store_ci_lo", "gap_store_ci_hi"
    return r, f"{spct(r.rec_bias_cens_days)} & {pct(r.gap_closed, 0)} [{pct(r[lo], 0)}, {pct(r[hi], 0)}]"
with open(f"{P}/tables/tab_recmethods.tex", "w") as f:
    f.write("\\begin{tabular*}{\\tblwidth}{@{}LLRRRR@{}}\n\\toprule\n"
            " & & \\multicolumn{2}{c}{Global truth} & \\multicolumn{2}{c}{Stress truth}\\\\\n"
            "Recovery & Information used & Rec.\\ bias & Gap closed [CI] & Rec.\\ bias & Gap closed [CI]\\\\\n\\midrule\n")
    for lab, info, kg, ks in REC:
        f.write(f"{lab} & {info} & {rec_row(kg)[1]} & {rec_row(ks)[1]}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular*}\n")
for lab, nm, kg, ks in [("km", "KM", "km_global", "km_stress"), ("tn", "TimesNet", ("tn", "global", "timesnet"), ("tn", "stress", "timesnet")),
                        ("m1sub", "MOneSub", ("tn", "global", "m1"), ("tn", "stress", "m1"))]:
    for key, sfx in [(kg, ""), (ks, "Stress")]:
        r, _ = rec_row(key)
        if r is None:
            continue
        N[f"{nm}{sfx}Gap"], N[f"{nm}{sfx}Bias"], N[f"{nm}{sfx}Cut"] = pct(r.gap_closed, 0), spct(r.rec_bias_cens_days), pct(-r.cost_change, 0)
for key, nm in [("lead_time_2d", "Lead"), ("capacity_p75", "Cap"), ("delivery_every_2d", "Deliv"), ("deferred_30", "Defer"),
                ("substitution_30", "Subst"), ("all_combined", "Comb")]:
    r = r28.loc[key]
    N[f"{nm}Gap"], N[f"{nm}Cut"], N[f"{nm}Bias"], N[f"{nm}StockDays"] = pct(r.gap_closed, 0), pct(-r.cost_change, 0), spct(r.rec_bias_cens_days), pct(r.stockout_day_rate, 0)
ext = r28.loc[["lead_time_2d", "capacity_p75", "delivery_every_2d", "deferred_30", "substitution_30", "all_combined"]]
N["ExtGapMin"], N["ExtGapMax"] = pct(ext.gap_closed.min(), 0), pct(ext.gap_closed.max(), 0)
N["ExtBiasMin"], N["ExtBiasMax"] = spct(ext.rec_bias_cens_days.min()), spct(ext.rec_bias_cens_days.max())
rw = r28.loc[[f"rolling_{o}" for o in [62, 69, 76, 83, 90]]]
N["RollCutMin"], N["RollCutMax"] = pct(-rw.cost_change.max(), 0), pct(-rw.cost_change.min(), 0)
N["RollGapMin"], N["RollGapMax"] = pct(rw.gap_closed.min(), 0), pct(rw.gap_closed.max(), 0)
N["RollCut"], N["RollCutLo"], N["RollCutHi"] = pct(-rp8.cost_change, 0), pct(-rp8.twoway_ci_hi, 0), pct(-rp8.twoway_ci_lo, 0)
N["RollGap"], N["RollGapLo"], N["RollGapHi"] = pct(rp8.gap_closed, 0), pct(rp8.gap_twoway_ci_lo, 0), pct(rp8.gap_twoway_ci_hi, 0)
N["RollDaysBetter"] = pct(rp8.days_better, 0)
N["TwoWayWidthRatio"] = f"{(rp8.twoway_ci_hi - rp8.twoway_ci_lo) / (w90.store_ci_hi - w90.store_ci_lo):.0f}"
N["RollDayCutMin"], N["RollDayCutMax"] = pct(-rp8.day_change_max, 0), pct(-rp8.day_change_min, 0)
r5 = ro.loc[0.5]
N["RollLowTauDaysBetter"] = pct(r5.days_better, 0)
for c, nm in [("city", "City"), ("category", "Cat")]:
    N[f"{nm}CILo"], N[f"{nm}CIHi"] = pct(-w90[f"{c}_ci_hi"]), pct(-w90[f"{c}_ci_lo"])
for key, nm in [(("stress", "mean", "m1"), "Stress"), (("global", "mean", "tobit"), "Tobit"), (("stress", "mean", "tobit"), "TobitStress"),
                (("global", "mean", "uniform"), "Flat"), (("stress", "mean", "uniform"), "FlatStress"), (("global", "mean", "m1"), "RevGlobal"),
                (("global", "safety", "m1"), "Safety"), (("global", "adaptive", "m1"), "Adaptive")]:
    r = r8.loc[key]
    N[f"{nm}Gap"], N[f"{nm}Cut"], N[f"{nm}Bias"] = pct(r.gap_closed, 0), pct(-r.cost_change, 0), spct(r.rec_bias_cens_days)
    N[f"{nm}StockDays"] = pct(r.stockout_day_rate, 0)
g8 = r8.loc[("global", "mean", "m1")]
N["ClusterLo"], N["ClusterHi"] = pct(-g8.cluster_ci_hi), pct(-g8.cluster_ci_lo)
N["SeriesCILo"], N["SeriesCIHi"] = pct(-g8.series_ci_hi), pct(-g8.series_ci_lo)
N["DayCutMin"], N["DayCutMax"] = pct(-g8.day_change_max, 0), pct(-g8.day_change_min, 0)
N["RevDayWorst"] = pct(-max(rt[(rt.tau == 0.8) & (rt.recovery == "m1")].day_change_max.max(), r28[~r28.index.str.startswith("km")].day_change_max.max()), 1)   # smallest daily M1 cut, tau = 0.8
a5 = rt[(rt.tau == 0.5) & (rt.retailer == "adaptive")].iloc[0]
N["AdaptiveLowTauCut"] = pct(-a5.cost_change, 1)
N["FlatGapLowTau"] = pct(rt[(rt.tau == 0.5) & (rt.truth == "global") & (rt.recovery == "uniform")].gap_closed.iloc[0], 0)
N["MOneGapLowTau"] = pct(rt[(rt.tau == 0.5) & (rt.truth == "global") & (rt.recovery == "m1") & (rt.retailer == "mean")].gap_closed.iloc[0], 0)
N["NStores"] = f"{np.unique(dd.store_id.values[::97]).size:,}"

with open(f"{P}/numbers.tex", "w") as f:
    f.write("% generated by tables.py -- do not edit by hand\n")
    for name, v in N.items():
        f.write(f"\\newcommand{{\\{name}}}{{{v}}}\n")
print("\n".join(f"{k} = {v}" for k, v in N.items()))

# ESWA: 3-5 highlights, <= 85 characters each, uploaded as a separate editable file
HL = ["Censoring-aware DSS turns hourly stockout labels into cost-sensitive orders",
      f"Recovery closes {N['GapDonor']}-{N['GapGlobal']}% of the cost gap to a full-visibility oracle",
      f"Expected cost falls {N['CostCut']}% (95% CI {N['CostCutLo']}-{N['CostCutHi']}%) versus the same DSS without recovery",
      "Gains hold across truth models, retailer policies, time windows and shelf lives",
      "Models fit on censored sales understate holiday and promotion uplift"]
assert all(len(h) <= 85 for h in HL), [(len(h), h) for h in HL if len(h) > 85]
open(f"{P}/highlights.txt", "w", encoding="utf-8").write("\n".join(HL) + "\n")
