# Censoring-aware replenishment DSS

Code for the paper *A censoring-aware decision-support system for perishable replenishment: turning latent
demand recovery into service-level-optimal ordering* (submitted to Expert Systems With Applications).

The system recovers latent demand from hourly stockout labels, forecasts its distribution with a global
LightGBM quantile model, converts it into a critical-fractile order, and reports where availability is lost.
It is evaluated in a semi-synthetic world with known demand. Everything runs on a CPU.

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
python run_all.py          # downloads FreshRetailNet-50K, runs every step, writes out/
python tables.py           # writes the paper's numbers and tables as LaTeX into paper/
```

The manuscript source is released together with the published article.

M5 (used only for the transfer experiment) needs the [Kaggle CLI](https://github.com/Kaggle/kaggle-api)
and acceptance of the competition rules; `run_all.py` prints the exact commands if the files are missing.

Measured runtimes on an AMD Ryzen 7 6800HS (8 cores, 16 GB RAM), no GPU: the whole pipeline takes
46 min (p1 49 s, p2 recovery 23 s, p3 forecasts 7.9 min, p6 12 min, p7 8.3 min, p8 4.5 min, p10 2.4 min,
p11 8.2 min, all other steps under 1 min). A clean re-run reproduced every result file bit for bit.

## Pipeline

| Step | Layer / module | Output in `out/` |
|---|---|---|
| `config.py` | Layer 0: all cost, service and shelf-life assumptions | |
| `p1_audit.py` | Layer 1: daily table, censoring mask, data audit | `daily.parquet`, `hourly.npz`, `audit.txt` |
| `p2_recovery.py` | Layer 2 (M1): latent-demand recovery and its validation | `recovered.parquet`, `recovery_validation.csv` |
| `p3_forecast.py` | Layer 3 (M2): LightGBM quantile forecasts, SHAP | `forecasts.parquet`, `shap_*.csv` |
| `p4_simulate.py` | Layers 4-5 (M3 + simulator) on real data | `kpi_all.csv`, `headline.csv` |
| `p5_diagnostics.py` | Layer 6 (M4): lost-demand accounting, stockout patterns | `diagnostics.csv` |
| `p6_semisynthetic.py` | Evaluation with known demand (main result) | `semi_*.csv` |
| `p7_robustness.py` | Truth constructions x retailer stocking regimes | `robustness.csv` |
| `p8_multiperiod.py` | Shelf life 1-3 days, 30-day horizon | `multiperiod_*.csv` |
| `p9_baselines.py` | Dataset authors' SSA forecaster (ported) | `baseline_*.csv` |
| `p10_transfer_m5.py` | Transfer to M5 with a daily censored-likelihood recovery | `m5_*.csv` |
| `p11_confidence.py` | 3 seeds x series bootstrap, global and donor truth | `confidence_*.csv` |
| `figures.py` | Figures 1-4 | `figures/*.pdf` |
| `tables.py` | Every number and table cited in the paper | `paper/numbers.tex`, `paper/tables/*.tex` |

The manuscript never contains a hand-typed result: `tables.py` writes LaTeX macros from the CSVs.
`p12_lt.py` (optional, not in the paper) repeats the evaluation on 97-day windows of FreshRetailNet-LT;
it reads the input and output folders from the `FRN_DATA` and `FRN_OUT` environment variables.

## Data

- FreshRetailNet-50K, CC-BY-4.0: <https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K>
- M5 Forecasting - Accuracy: <https://www.kaggle.com/competitions/m5-forecasting-accuracy>

The SSA baseline in `p9_baselines.py` is ported from the dataset authors' public code
(<https://github.com/Dingdong-Inc/frn-50k-baseline>, Apache-2.0).
