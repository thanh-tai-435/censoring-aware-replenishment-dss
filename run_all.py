"""Reproduce every number, table and figure of the paper, in dependency order.
    python run_all.py            # whole pipeline (46 min measured on an 8-core laptop CPU)
    python run_all.py p6 p9      # only the named steps (their inputs must already exist in out/)
FreshRetailNet-50K is downloaded automatically; M5 needs the Kaggle CLI (see README)."""
import os, subprocess, sys, time, urllib.request

STEPS = ["p1_audit", "p2_recovery", "p3_forecast", "p4_simulate", "p5_diagnostics", "p6_semisynthetic",
         "p7_robustness", "p8_multiperiod", "p9_baselines", "p10_transfer_m5", "p11_confidence", "p13_review",
         "figures", "tables"]
HF = "https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K/resolve/main/data/{}.parquet"


def ensure_data():
    os.makedirs("data", exist_ok=True)
    for split in ["train", "eval"]:
        path = f"data/{split}.parquet"
        if not os.path.exists(path):
            print(f"downloading FreshRetailNet-50K {split} ...", flush=True)
            urllib.request.urlretrieve(HF.format(split), path)
    m5 = ["data/m5/sales_train_evaluation.csv", "data/m5/calendar.csv"]
    if not all(os.path.exists(f) for f in m5):
        sys.exit("M5 data missing. Accept the competition rules on kaggle.com/c/m5-forecasting-accuracy, then:\n"
                 "  kaggle competitions download -c m5-forecasting-accuracy -f sales_train_evaluation.csv -p data/m5\n"
                 "  kaggle competitions download -c m5-forecasting-accuracy -f calendar.csv -p data/m5\n"
                 "and unzip both into data/m5/.")


if __name__ == "__main__":
    steps = [s for s in STEPS if s in sys.argv[1:]] or STEPS
    ensure_data()
    for s in steps:
        t = time.time()
        print(f"== {s}", flush=True)
        subprocess.run([sys.executable, "-W", "ignore", f"{s}.py"], check=True)
        print(f"   {s} done in {time.time() - t:.0f}s", flush=True)
    print("All outputs in out/. LaTeX numbers and tables for the paper are in paper/ (written by tables.py).")
