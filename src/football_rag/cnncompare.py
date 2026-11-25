"""ANN vs CNN comparison data for the Evaluation tab, from the saved Phase 2 predictions (nothing is retrained).

- calibration bins (6 quantile bins, as in results/phase2_script/calibration.png) on the TEST split for logistic regression
  (refit here with the same recipe as scripts/phase2_train.py: standard scaling fitted on train rows, LogisticRegression
  max_iter=2000), the 5-seed ANN mean and the 5-seed CNN mean (results/phase2_script/{ann,cnn}_test.npy);
- per-seed test metrics, the CNN architecture search table (results/cnn_search.csv);
- the spatial tensor of one test shot (data/processed/spatial.npz, loaded lazily on first use) with the three models' probabilities.
"""
import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
P2 = ROOT / "results" / "phase2_script"
NON_FEATURES = ["row_idx", "shot_id", "match_id", "team", "player", "period", "minute", "split", "goal"]   # as in phase2_train.py
CHANNELS = ["attackers", "defenders", "opponent goalkeeper", "shooter", "visible area (camera polygon)", "distance to goal / 120"]


@lru_cache(maxsize=1)
def _base():
    tab = pd.read_csv(DATA / "tabular.csv")
    feat = [c for c in tab.columns if c not in NON_FEATURES]
    idx = {s: tab.index[tab["split"] == s].to_numpy() for s in ("train", "val", "test")}
    X = StandardScaler().fit(tab.loc[idx["train"], feat]).transform(tab[feat]).astype("float32")
    y = tab["goal"].to_numpy().astype("float32")
    lg = LogisticRegression(max_iter=2000).fit(X[idx["train"]], y[idx["train"]])
    p_test = {"logistic": lg.predict_proba(X[idx["test"]])[:, 1],
              "ANN": np.load(P2 / "ann_test.npy").mean(0), "CNN": np.load(P2 / "cnn_test.npy").mean(0)}
    return tab, idx, y[idx["test"]], p_test


def _metrics(y, p):
    return {"log_loss": float(log_loss(y, p)), "auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p))}


@lru_cache(maxsize=1)
def load_cnn_comparison() -> dict:
    tab, idx, yt, pt = _base()
    cal = {}
    for name, p in pt.items():
        frac, mean_p = calibration_curve(yt, p, n_bins=6, strategy="quantile")
        edges = np.quantile(p, np.linspace(0, 1, 7))
        counts = [int(((p >= lo) & (p <= hi if i == 5 else p < hi)).sum()) for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]
        cal[name] = [{"mean_pred": float(m), "observed": float(f), "n": c} for m, f, c in zip(mean_p, frac, counts)]
    seeds = {}
    for name, f in (("ANN", "ann_test.npy"), ("CNN", "cnn_test.npy")):
        arr = np.load(P2 / f)
        seeds[name] = [_metrics(yt, a) for a in arr]
    summary = []   # reported in the Phase 2 notes: mean ± sd (ddof 0) of the per-seed metrics, plus the metrics of the 5-seed averaged prediction
    for name in ("logistic", "ANN", "CNN"):
        ens = _metrics(yt, pt[name])
        if name in seeds:
            cells = [f"{np.mean([m[k] for m in seeds[name]]):.4f} ± {np.std([m[k] for m in seeds[name]]):.4f}" for k in ("log_loss", "auc", "brier")]
        else:
            cells = [f"{ens[k]:.4f}" for k in ("log_loss", "auc", "brier")]
        summary.append([name, *cells, *(f"{ens[k]:.4f}" for k in ("log_loss", "auc", "brier"))])
    search = []
    sp = ROOT / "results" / "cnn_search.csv"
    if sp.exists():
        with open(sp, encoding="utf-8", newline="") as f:
            search = [[r["ch"], r["k"], r["lr"], f"{float(r['ll']):.4f} ± {float(r['ll_sd']):.4f}", f"{float(r['auc']):.3f}", r["epoch"]] for r in csv.DictReader(f)]
    test_rows = tab.loc[idx["test"]]
    sample = [{"shot_id": r.shot_id, "label": f"{r.team}, minute {int(r.minute)}, {'goal' if r.goal == 1 else 'no goal'}"}
              for r in test_rows.iloc[::74].head(20).itertuples()]
    return {
        "summary": {"columns": ["model", "log-loss (mean ± sd over seeds)", "AUC", "Brier", "log-loss of 5-seed averaged prediction", "AUC", "Brier"], "rows": summary},
        "calibration": cal, "seeds": seeds, "cnn_search": {"columns": ["conv channels", "kernel", "lr", "val log-loss ± sd", "val AUC", "mean best epoch"], "rows": search},
        "channels": CHANNELS, "sample_shots": sample,
        "facts": [
            "ANN: tabular features (distance, angle, body part, technique, play pattern, pressure, ...); one hidden layer of 32 units, dropout 0.3; 5 seeds averaged.",
            "CNN: the 360 freeze-frame as a 6-channel 40x60 pitch tensor (attackers, defenders, goalkeeper, shooter, visible area, distance to goal); "
            "2 conv blocks (16, 32 channels, kernel 3, batch norm, max pool), small head, dropout 0.5, weight decay 1e-4, left-right flip augmentation on training data; 5 seeds averaged.",
            "The CNN sees positions only: no body part, technique, play pattern or pressure. From positions alone it recovers about 89% of the log-loss improvement over the base rate that the tabular models reach.",
            "Test differences are within sampling noise: paired match-bootstrap CNN minus ANN log-loss +0.0053 (95% CI -0.0028 to +0.0139), AUC -0.0174 (-0.0416 to +0.0053).",
            "CNN v1 (3 blocks, large head, no augmentation) overfitted (validation log-loss about 0.30); the table shows the validation search for the final design (the 16,32 design was kept as the simplest within noise).",
        ],
    }


@lru_cache(maxsize=1)
def _spatial():
    return np.load(DATA / "spatial.npz")["X"]


def shot_tensor(shot_id: str) -> dict | None:
    """Spatial tensor and model probabilities of one TEST-split shot (saved CNN predictions exist only for val/test shots)."""
    tab, idx, _, pt = _base()
    hit = tab.index[tab["shot_id"] == shot_id]
    if len(hit) == 0:
        return None
    i = int(hit[0])
    row = tab.loc[i]
    pos = np.where(idx["test"] == i)[0]
    probs = {n: float(p[pos[0]]) for n, p in pt.items()} if len(pos) else None
    x = _spatial()[i].astype("float32")
    return {"shot_id": shot_id, "team": row["team"], "player": row["player"], "minute": int(row["minute"]), "split": row["split"],
            "goal": int(row["goal"]), "probabilities": probs, "channels": CHANNELS,
            "tensor": np.round(x, 3).tolist(), "shape": list(x.shape)}


def _json_default(o):   # pragma: no cover
    return str(o)
