#!/usr/bin/env python
"""Phase 2, one command: final ANN + CNN training, evaluation, bootstrap and calibration plot.

    uv run python scripts/phase2_train.py
    uv run python scripts/phase2_train.py --cnn-ch 16,32 --cnn-lr 3e-4 --out results/phase2_script

What it does (same logic as notebooks/02_models_clean.ipynb, sections 3-6):
  1. loads the Phase 1 files (tabular.csv, spatial.npz) and fits the scaler on train rows only
  2. baselines: constant and logistic regression
  3. trains the final ANN and the final CNN on N seeds (data lives on the GPU), saves val/test predictions
  4. writes the results table, a paired match-level bootstrap on test, the calibration plot
  5. writes run_info.json: arguments, library versions, GPU, SHA-256 of the input files

The model configs below are the FINAL configs chosen on validation in the notebook. Set them from the
notebook (ANN grid table in 4.2, CNN search table in 5.3), either here or with the command-line flags.
Test is only scored in step 4, after all training is finished.
"""
import argparse
import copy
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                                   # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]              # project root (this file lives in scripts/)
DATA = ROOT / "data" / "processed"
DEVICE = torch.device("cpu")                            # set in main()

NON_FEATURES = ["row_idx", "shot_id", "match_id", "team", "player", "period", "minute", "split", "goal"]
COLORS = {"logistic": "#2a78d6", "ANN": "#eb6834", "CNN": "#1baf7a"}
MARKERS = {"logistic": "o", "ANN": "s", "CNN": "^"}


def parse_args():
    ints = lambda s: tuple(int(x) for x in s.split(","))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (0..N-1) for ANN and CNN")
    p.add_argument("--out", default="results/phase2_script", help="output folder, relative to the project root")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--ann-hidden", type=ints, default=(32,))
    p.add_argument("--ann-dropout", type=float, default=0.3)
    p.add_argument("--ann-lr", type=float, default=1e-3)
    p.add_argument("--cnn-ch", type=ints, default=(16, 32))
    p.add_argument("--cnn-k", type=int, default=3)
    p.add_argument("--cnn-dropout", type=float, default=0.5)
    p.add_argument("--cnn-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4, help="CNN weight decay")
    p.add_argument("--max-epochs", type=int, default=None, help="cap epochs (smoke tests); default 200 ANN / 120 CNN")
    p.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples (0 = skip)")
    return p.parse_args()


# ------------------------------------------------------------------------------------------ data
def load_data():
    tab = pd.read_csv(DATA / "tabular.csv")
    X_sp = np.load(DATA / "spatial.npz")["X"]                    # (N, 6, 40, 60) float16
    assert (tab["row_idx"].values == np.arange(len(tab))).all() and len(X_sp) == len(tab)
    feat_cols = [c for c in tab.columns if c not in NON_FEATURES]
    idx = {s: tab.index[tab["split"] == s].to_numpy() for s in ["train", "val", "test"]}
    scaler = StandardScaler().fit(tab.loc[idx["train"], feat_cols])         # train rows only
    X_tab = scaler.transform(tab[feat_cols]).astype("float32")
    y = tab["goal"].to_numpy().astype("float32")
    return tab, X_sp, X_tab, y, feat_cols, idx


def evaluate(y_true, p):
    return {"log_loss": log_loss(y_true, p), "auc": roc_auc_score(y_true, p), "brier": brier_score_loss(y_true, p)}


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------------------------------ training
class GpuLoader:
    """Whole split on the device; batches (and optional left-right flip) are made there."""
    def __init__(self, X, y, batch_size=128, shuffle=False, augment=False):
        assert not augment or X.ndim == 4, "flip augmentation is only defined for (N, C, H, W) maps"
        self.X = torch.from_numpy(np.ascontiguousarray(X)).to(DEVICE)
        self.y = torch.from_numpy(np.ascontiguousarray(y)).to(DEVICE)
        self.dataset = self.y
        self.bs, self.shuffle, self.augment = batch_size, shuffle, augment

    def __len__(self):
        return (len(self.y) + self.bs - 1) // self.bs

    def __iter__(self):
        n = len(self.y)
        order = torch.randperm(n, device=DEVICE) if self.shuffle else torch.arange(n, device=DEVICE)
        for i in range(0, n, self.bs):
            ix = order[i:i + self.bs]
            xb = self.X[ix].float()
            if self.augment:
                flip = torch.rand(len(ix), device=DEVICE) < 0.5
                xb = torch.where(flip.view(-1, 1, 1, 1), xb.flip(2), xb)    # mirror the y axis
            yield xb, self.y[ix]


class ANN(nn.Module):
    def __init__(self, n_in, hidden=(32,), dropout=0.3):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]                    # raw logit
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class CNN(nn.Module):
    def __init__(self, ch=(16, 32), k=3, dropout=0.5, hidden=32):
        super().__init__()
        layers, c_in = [], 6
        for c in ch:
            layers += [nn.Conv2d(c_in, c, k, padding=k // 2), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(ch[-1] * 12, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


def predict(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            out.append(torch.sigmoid(model(xb)).cpu())
    return torch.cat(out).numpy()


def train_model(model, train_dl, val_dl, lr, weight_decay=0.0, max_epochs=200, patience=15):
    """Adam + BCE-with-logits, early stopping on val log-loss, best epoch's weights restored."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()                    # no pos_weight: keeps probabilities calibrated
    y_val = val_dl.y.cpu().numpy()
    best = {"ll": np.inf, "state": None, "epoch": 0}
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        val_ll = log_loss(y_val, predict(model, val_dl))
        if val_ll < best["ll"] - 1e-4:
            best = {"ll": val_ll, "state": copy.deepcopy(model.state_dict()), "epoch": epoch}
        elif epoch - best["epoch"] >= patience:
            break
    model.load_state_dict(best["state"])
    return model, best["epoch"]


# ------------------------------------------------------------------------------------------ reporting
def summarize(name, yv, yt, Pv, Pt):
    row = {"model": name}
    for split, P, yy in (("val", Pv, yv), ("test", Pt, yt)):
        ms = [evaluate(yy, p) for p in P]
        for k in ("log_loss", "auc", "brier"):
            v = [m[k] for m in ms]
            row[f"{split}_{k}"] = f"{np.mean(v):.4f}" + (f" ± {np.std(v):.4f}" if len(v) > 1 else "")
    return row


def paired_bootstrap(tab, idx, yt, P, n_boot):
    """Resample whole TEST MATCHES (shots within a match are not independent); same resample for all models."""
    rng = np.random.default_rng(0)
    mid = tab.loc[idx["test"], "match_id"].to_numpy()
    groups = {m: np.where(mid == m)[0] for m in np.unique(mid)}
    keys = list(groups)
    ll = {k: -(yt * np.log(np.clip(p, 1e-15, 1 - 1e-15)) + (1 - yt) * np.log(np.clip(1 - p, 1e-15, 1))) for k, p in P.items()}
    pairs = [("ANN", "logistic"), ("CNN", "ANN"), ("CNN", "logistic")]
    diff = {pr: {"ll": [], "auc": []} for pr in pairs}
    for _ in range(n_boot):
        ix = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        if yt[ix].min() == yt[ix].max():
            continue
        auc = {k: roc_auc_score(yt[ix], p[ix]) for k, p in P.items()}
        for a, b in pairs:
            diff[(a, b)]["ll"].append(ll[a][ix].mean() - ll[b][ix].mean())
            diff[(a, b)]["auc"].append(auc[a] - auc[b])
    rows = []
    for (a, b), d in diff.items():
        for metric in ("ll", "auc"):
            v = np.array(d[metric])
            lo, hi = np.percentile(v, [2.5, 97.5])
            rows.append({"comparison": f"{a} - {b}", "metric": metric, "mean_diff": round(float(v.mean()), 4),
                         "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4),
                         "ci_excludes_0": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def calibration_plot(path, yv, yt, Pv, Pt):
    ink2, muted, grid, surface = "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
                         "axes.edgecolor": grid, "axes.labelcolor": ink2, "xtick.color": muted, "ytick.color": muted,
                         "legend.labelcolor": ink2, "legend.frameon": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.8})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    for ax, (title, yy, P) in zip(axes, [("validation", yv, Pv), ("test", yt, Pt)]):
        hi = 0.0
        for name, p in P.items():
            frac, mean_p = calibration_curve(yy, p, n_bins=6, strategy="quantile")
            ax.plot(mean_p, frac, marker=MARKERS[name], ms=6, lw=1.8, color=COLORS[name], label=name,
                    markeredgecolor=surface, markeredgewidth=1.2)
            hi = max(hi, mean_p.max(), frac.max())
        ax.plot([0, hi * 1.05], [0, hi * 1.05], ls="--", lw=1, color=muted, label="perfect")
        ax.set_title(f"Calibration: {title}", loc="left", fontsize=11)
        ax.set_xlabel("mean predicted goal probability")
    axes[0].set_ylabel("observed goal rate")
    axes[0].legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------ main
def main():
    global DEVICE
    a = parse_args()
    DEVICE = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) or a.device == "cuda" else "cpu")
    torch.backends.cudnn.deterministic = True
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    print(f"device: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))

    tab, X_sp, X_tab, y, feat_cols, idx = load_data()
    yv, yt = y[idx["val"]], y[idx["test"]]
    print(f"{len(feat_cols)} features | {X_sp.shape} | shots per split: { {s: len(i) for s, i in idx.items()} }")

    # baselines
    base_rate = y[idx["train"]].mean()
    logreg = LogisticRegression(max_iter=2000).fit(X_tab[idx["train"]], y[idx["train"]])
    lg_v, lg_t = (logreg.predict_proba(X_tab[idx[s]])[:, 1] for s in ("val", "test"))

    # loaders (data on the device)
    tab_tr = GpuLoader(X_tab[idx["train"]], y[idx["train"]], 256, shuffle=True)
    tab_va = GpuLoader(X_tab[idx["val"]], y[idx["val"]], 256)
    tab_te = GpuLoader(X_tab[idx["test"]], y[idx["test"]], 256)
    sp_tr = GpuLoader(X_sp[idx["train"]], y[idx["train"]], 128, shuffle=True, augment=True)   # flip: train only
    sp_va = GpuLoader(X_sp[idx["val"]], y[idx["val"]], 128)
    sp_te = GpuLoader(X_sp[idx["test"]], y[idx["test"]], 128)

    ann_cap = a.max_epochs or 200
    cnn_cap = a.max_epochs or 120
    ann_v, ann_t, cnn_v, cnn_t = [], [], [], []
    for seed in range(a.seeds):
        t0 = time.time()
        set_seed(seed)
        m, ep = train_model(ANN(len(feat_cols), a.ann_hidden, a.ann_dropout), tab_tr, tab_va,
                            lr=a.ann_lr, max_epochs=ann_cap)
        ann_v.append(predict(m, tab_va)); ann_t.append(predict(m, tab_te))
        print(f"ANN seed {seed} | best epoch {ep:3d} | val log-loss {log_loss(yv, ann_v[-1]):.4f} | {time.time() - t0:.0f}s")
    for seed in range(a.seeds):
        t0 = time.time()
        set_seed(seed)
        m, ep = train_model(CNN(a.cnn_ch, a.cnn_k, a.cnn_dropout), sp_tr, sp_va,
                            lr=a.cnn_lr, weight_decay=a.weight_decay, max_epochs=cnn_cap)
        cnn_v.append(predict(m, sp_va)); cnn_t.append(predict(m, sp_te))
        print(f"CNN seed {seed} | best epoch {ep:3d} | val log-loss {log_loss(yv, cnn_v[-1]):.4f} | {time.time() - t0:.0f}s")
    ann_v, ann_t, cnn_v, cnn_t = map(np.array, (ann_v, ann_t, cnn_v, cnn_t))
    for name, arr in (("ann_val", ann_v), ("ann_test", ann_t), ("cnn_val", cnn_v), ("cnn_test", cnn_t)):
        np.save(out / f"{name}.npy", arr)

    # ---- everything below scores test; no training happens after this point
    results = pd.DataFrame([
        summarize("constant", yv, yt, np.full((1, len(yv)), base_rate), np.full((1, len(yt)), base_rate)),
        summarize("logistic", yv, yt, lg_v[None], lg_t[None]),
        summarize("ANN", yv, yt, ann_v, ann_t),
        summarize("CNN", yv, yt, cnn_v, cnn_t)])
    results.to_csv(out / "phase2_results.csv", index=False)
    print("\n" + results.to_string(index=False))

    Pv = {"logistic": lg_v, "ANN": ann_v.mean(0), "CNN": cnn_v.mean(0)}
    Pt = {"logistic": lg_t, "ANN": ann_t.mean(0), "CNN": cnn_t.mean(0)}
    calibration_plot(out / "calibration.png", yv, yt, Pv, Pt)

    if a.bootstrap > 0:
        boot = paired_bootstrap(tab, idx, yt, Pt, a.bootstrap)
        boot.to_csv(out / "bootstrap_test.csv", index=False)
        print("\nPaired match-level bootstrap on test (log-loss: + = first model worse; AUC: - = first model worse)")
        print(boot.to_string(index=False))

    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": vars(a),
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else None,
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "cudnn_deterministic": True,
        "input_sha256": {f: sha256(DATA / f) for f in ("tabular.csv", "spatial.npz", "splits.csv") if (DATA / f).exists()},
        "shots_per_split": {s: int(len(i)) for s, i in idx.items()},
        "total_seconds": round(time.time() - t_start),
        "results": results.to_dict(orient="records"),
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
