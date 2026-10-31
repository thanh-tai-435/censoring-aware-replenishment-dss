"""Runs in the separate deep-learning environment (.venv-dl: CPU torch + PyPOTS; uses a GPU when one is visible, e.g. on Kaggle), called by p16_timesnet.py.
Fits PyPOTS TimesNet with the dataset authors' recovery settings (frn-50k-baseline, latent_demand_recovery/exp:
30-day windows of 16 business hours, 6 features = sales + 4 covariates + hour index, 2 layers, top_k 7,
d_model 64, d_ffn 32, 5 kernels, non-stationary norm, batch 128, 5 epochs, Adam lr 1e-3, wd 1e-5, seed 1024)
on the training windows, and imputes both the training windows and the per-day evaluation windows.
usage: python p16_timesnet_worker.py in.npz out.npz [epochs]"""
import sys, random, time
import numpy as np, torch
from pypots.imputation import TimesNet
from pypots.optim import Adam

random.seed(1024); np.random.seed(1024); torch.manual_seed(1024)
z = np.load(sys.argv[1])
model = TimesNet(n_steps=480, n_features=6, n_layers=2, top_k=7, d_model=64, d_ffn=32, n_kernels=5, dropout=0.0,
                 apply_nonstationary_norm=True, batch_size=128, epochs=int(sys.argv[3]) if len(sys.argv) > 3 else 5, patience=None,
                 optimizer=Adam(lr=1e-3, weight_decay=1e-5), device="cuda" if torch.cuda.is_available() else "cpu", saving_path=None, verbose=True)
t0 = time.time()
model.fit({"X": z["train"]})
print(f"fit {time.time() - t0:.0f}s", flush=True)
imp = lambda X: model.predict({"X": X})["imputation"][..., 0]
np.savez(sys.argv[2], train=imp(z["train"]), evalw=imp(z["evalw"]))
print(f"done {time.time() - t0:.0f}s", flush=True)
