#!/usr/bin/env python
"""Phase 4, step 1: choose the xG model for the pipeline. VALIDATION ONLY (test is never loaded here).

    uv run python scripts/phase4_select_xg.py

SELECTION RULE (written before any result was computed):
  candidates : ANN (32,) and CNN (16,32); logistic regression is a reference only.
  primary    : validation log-loss of the seed-averaged prediction.
  "clearly better" = log-loss gap > 2 x the larger seed sd  AND  the paired bootstrap over the 60 val
                     MATCHES (2000 resamples) has a 95% CI that excludes 0.
  otherwise  : take the simpler model (ANN before CNN).
  secondary  : calibration on val = ECE with equal-frequency bins (10 bins; 5 bins also shown) and a
               reliability curve. It only decides if log-loss is NOT clearly different, and with
               ~146 val goals (~15 per bin) it is noisy.
  post-hoc calibration (Platt on the logit, isotonic): fitted/evaluated with match-grouped 5-fold CV
               on val only; adopted only if its CV log-loss beats raw by more than the seed-sd margin
               and the bootstrap CI of the difference excludes 0.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
P2 = ROOT / "results" / "phase2_script"
OUT = ROOT / "results" / "phase4"
NON_FEATURES = ["row_idx", "shot_id", "match_id", "team", "player", "period", "minute", "split", "goal"]
EPS = 1e-6


def ece(y, p, bins):
    """Expected calibration error with equal-frequency bins (sorted by p, split into `bins` chunks)."""
    order = np.argsort(p)
    e = 0.0
    for chunk in np.array_split(order, bins):
        e += len(chunk) / len(y) * abs(y[chunk].mean() - p[chunk].mean())
    return e


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def per_shot_ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def match_bootstrap(match_ids, a, b, n=2000, seed=0):
    """Paired bootstrap over matches of mean(a) - mean(b), a and b = per-shot losses."""
    rng = np.random.default_rng(seed)
    keys = np.unique(match_ids)
    groups = {k: np.where(match_ids == k)[0] for k in keys}
    d = []
    for _ in range(n):
        ix = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])
        d.append(a[ix].mean() - b[ix].mean())
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi)


def cv_calibrate(y, p, groups, method, seed=0):
    """Out-of-fold calibrated probabilities (match-grouped 5-fold) on val."""
    out = np.zeros_like(p, dtype=float)
    for tr, te in GroupKFold(n_splits=5).split(p, y, groups):
        if method == "platt":
            m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p[tr])[:, None], y[tr])
            out[te] = m.predict_proba(logit(p[te])[:, None])[:, 1]
        else:
            m = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip").fit(p[tr], y[tr])
            out[te] = m.predict(p[te])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tab = pd.read_csv(DATA / "tabular.csv")
    feat = [c for c in tab.columns if c not in NON_FEATURES]
    tr, va = (tab["split"] == "train").to_numpy(), (tab["split"] == "val").to_numpy()
    y = tab.loc[va, "goal"].to_numpy().astype(float)
    mid = tab.loc[va, "match_id"].to_numpy()

    sc = StandardScaler().fit(tab.loc[tr, feat])
    lg = LogisticRegression(max_iter=2000).fit(sc.transform(tab.loc[tr, feat]), tab.loc[tr, "goal"])
    lg_v = lg.predict_proba(sc.transform(tab.loc[va, feat]))[:, 1]

    ann_seeds, cnn_seeds = np.load(P2 / "ann_val.npy"), np.load(P2 / "cnn_val.npy")   # (5, 1463); VAL files only
    assert ann_seeds.shape[1] == cnn_seeds.shape[1] == va.sum()
    P = {"logistic": lg_v, "ANN": ann_seeds.mean(0), "CNN": cnn_seeds.mean(0)}
    sd = {"ANN": np.std([log_loss(y, s) for s in ann_seeds]), "CNN": np.std([log_loss(y, s) for s in cnn_seeds])}

    rows = []
    for k, p in P.items():
        rows.append({"model": k, "val_log_loss": log_loss(y, p), "seed_sd": sd.get(k, np.nan),
                     "val_auc": roc_auc_score(y, p), "val_brier": brier_score_loss(y, p),
                     "ece_10bin": ece(y, p, 10), "ece_5bin": ece(y, p, 5), "mean_p": p.mean()})
    table = pd.DataFrame(rows)
    print(f"val goals: {int(y.sum())} of {len(y)} | base rate {y.mean():.4f}")
    print(table.round(4).to_string(index=False))

    # ---- rule applied: CNN vs ANN
    ll = {k: per_shot_ll(y, p) for k, p in P.items()}
    gap = log_loss(y, P["CNN"]) - log_loss(y, P["ANN"])             # + = CNN worse
    thr = 2 * max(sd.values())
    bm, lo, hi = match_bootstrap(mid, ll["CNN"], ll["ANN"])
    cnn_clearly_better = (-gap > thr) and (hi < 0)
    ann_clearly_better = (gap > thr) and (lo > 0)
    print(f"\nCNN - ANN val log-loss gap {gap:+.4f} (2 x seed sd = {thr:.4f}); bootstrap {bm:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    if cnn_clearly_better:
        chosen, why = "CNN", "CNN clearly better on log-loss"
    elif ann_clearly_better:
        chosen, why = "ANN", "ANN clearly better on log-loss"
    else:
        # not clearly different on log-loss: simpler model (ANN) unless calibration says otherwise (reported, not used here)
        chosen, why = "ANN", "no clear log-loss difference -> simpler model"
    print(f"CHOSEN: {chosen}  ({why})")

    # ---- post-hoc calibration, CV on val, for the chosen model
    p = P[chosen]
    cal = {"raw": p, "platt": cv_calibrate(y, p, mid, "platt"), "isotonic": cv_calibrate(y, p, mid, "isotonic")}
    cal_rows, adopt = [], "raw"
    for k, q in cal.items():
        m, l, h = match_bootstrap(mid, per_shot_ll(y, q), ll[chosen]) if k != "raw" else (0.0, 0.0, 0.0)
        cal_rows.append({"variant": k, "cv_log_loss": log_loss(y, q), "ece_10bin": ece(y, q, 10),
                         "ll_diff_vs_raw": m, "ci_low": l, "ci_high": h})
    cal_tab = pd.DataFrame(cal_rows)
    print("\nPost-hoc calibration, match-grouped 5-fold CV on val (diff < 0 = better than raw)")
    print(cal_tab.round(4).to_string(index=False))
    for r in cal_rows[1:]:
        if -r["ll_diff_vs_raw"] > thr and r["ci_high"] < 0:
            adopt = r["variant"]
    print(f"post-hoc calibration adopted: {adopt}")

    # ---- outputs
    table.to_csv(OUT / "selection_table.csv", index=False)
    cal_tab.to_csv(OUT / "posthoc_calibration_cv.csv", index=False)
    (OUT / "selection.json").write_text(json.dumps({
        "chosen": chosen, "reason": why, "posthoc_calibration": adopt,
        "cnn_minus_ann_logloss": gap, "two_x_seed_sd": thr,
        "bootstrap_cnn_minus_ann": {"mean": bm, "ci_low": lo, "ci_high": hi},
        "val_goals": int(y.sum()), "val_shots": int(len(y))}, indent=2), encoding="utf-8")

    ink, muted, grid, surface = "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
    colors = {"logistic": "#2a78d6", "ANN": "#eb6834", "CNN": "#1baf7a"}
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
                         "axes.edgecolor": grid, "axes.labelcolor": ink, "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": grid})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, bins in zip(axes, (10, 5)):
        hi_ = 0.0
        for k, q in P.items():
            chunks = np.array_split(np.argsort(q), bins)
            mp = [q[c].mean() for c in chunks]; fr = [y[c].mean() for c in chunks]
            se = [np.sqrt(f * (1 - f) / len(c)) for f, c in zip(fr, chunks)]
            ax.errorbar(mp, fr, yerr=1.96 * np.array(se), marker="o", ms=5, lw=1.5, capsize=2, color=colors[k],
                        label=f"{k} (ECE {ece(y, q, bins):.3f})", alpha=0.9)
            hi_ = max(hi_, max(mp), max(fr))
        ax.plot([0, hi_ * 1.05], [0, hi_ * 1.05], ls="--", color=muted, lw=1, label="perfect")
        ax.set_title(f"Validation reliability, {bins} equal-frequency bins", loc="left", fontsize=11)
        ax.set_xlabel("mean predicted goal probability")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("observed goal rate (95% binomial bars)")
    plt.tight_layout()
    plt.savefig(OUT / "reliability_val.png", dpi=150)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
