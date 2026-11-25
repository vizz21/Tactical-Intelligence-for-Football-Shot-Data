#!/usr/bin/env python
"""Phase 4, step 2: retrain the selected xG model (ANN), save it, verify vs Phase 2, score all shots.

    uv run python scripts/phase4_build_xg.py            # CPU

Order matters: (1) train 5 seeds with the Phase 2 code, (2) save weights + train-only scaler,
(3) VERIFY on VAL against results/phase2_script/ann_val.npy, (4) write data/processed/xg_scores.csv,
(5) only then report the test log-loss/AUC once. Nothing is selected on test.
"""
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import phase2_train as p2                                   # noqa: E402  reuse ANN, GpuLoader, train_model
from football_rag import config as C                         # noqa: E402
from football_rag.xg import load_xg                          # noqa: E402

CFG = {"hidden": [32], "dropout": 0.3, "lr": 1e-3, "n_seeds": 5, "max_epochs": 200}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def main():
    t0 = time.time()
    p2.DEVICE = torch.device("cpu")
    tab = pd.read_csv(C.TABULAR)
    feat = [c for c in tab.columns if c not in C.NON_FEATURES]
    idx = {s: tab.index[tab["split"] == s].to_numpy() for s in ("train", "val", "test")}
    scaler = StandardScaler().fit(tab.loc[idx["train"], feat])          # train rows only
    X = scaler.transform(tab[feat]).astype("float32")
    y = tab["goal"].to_numpy().astype("float32")

    dl_tr = p2.GpuLoader(X[idx["train"]], y[idx["train"]], 256, shuffle=True)
    dl_va = p2.GpuLoader(X[idx["val"]], y[idx["val"]], 256)
    out = C.XG_MODEL_DIR
    out.mkdir(parents=True, exist_ok=True)
    epochs = []
    for seed in range(CFG["n_seeds"]):
        p2.set_seed(seed)
        m, ep = p2.train_model(p2.ANN(len(feat), tuple(CFG["hidden"]), CFG["dropout"]), dl_tr, dl_va,
                               lr=CFG["lr"], max_epochs=CFG["max_epochs"])
        torch.save(m.state_dict(), out / f"seed{seed}.pt")
        epochs.append(ep)
        print(f"seed {seed} | best epoch {ep} | val log-loss {log_loss(y[idx['val']], p2.predict(m, dl_va)):.4f}")
    np.savez(out / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    (out / "config.json").write_text(json.dumps({
        **CFG, "feat_cols": feat, "best_epochs": epochs, "scaler_fit": "train rows only", "device_trained": "cpu",
        "torch": torch.__version__, "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "input_sha256": {"tabular.csv": sha(C.TABULAR)}}, indent=2), encoding="utf-8")

    # ---- (3) verify on VAL against the Phase 2 saved predictions (loader round-trip included)
    model = load_xg()
    seeds_all = model.predict_seeds(tab)                                  # (5, N)
    ref = np.load(ROOT / "results" / "phase2_script" / "ann_val.npy")
    yv = y[idx["val"]]
    mine = seeds_all[:, idx["val"]]
    print("\nVAL check vs Phase 2 (GPU) predictions")
    print(f"  Phase 2 : per-seed val log-loss {np.mean([log_loss(yv, s) for s in ref]):.4f}, ensemble {log_loss(yv, ref.mean(0)):.4f}")
    print(f"  Phase 4 : per-seed val log-loss {np.mean([log_loss(yv, s) for s in mine]):.4f}, ensemble {log_loss(yv, mine.mean(0)):.4f}")
    print(f"  seed-wise max|diff| {np.abs(mine - ref).max():.4f} | mean|diff| {np.abs(mine - ref).mean():.5f} "
          f"| corr of ensemble {np.corrcoef(mine.mean(0), ref.mean(0))[0, 1]:.5f}")

    gap = abs(log_loss(yv, mine.mean(0)) - log_loss(yv, ref.mean(0)))
    if gap > 0.003:                                                       # val gate: stop before touching test
        raise SystemExit(f"VAL check failed (ensemble log-loss differs by {gap:.4f}); not scoring or reporting test")
    print(f"  val check passed (ensemble log-loss differs by {gap:.4f})")

    # ---- (4) score all shots
    xg = seeds_all.mean(0)
    scores = pd.DataFrame({"row_idx": tab["row_idx"], "shot_id": tab["shot_id"], "xg": xg,
                           "split": tab["split"], "in_sample": tab["split"].eq("train")})
    scores.to_csv(C.XG_SCORES, index=False)
    assert len(scores) == 10233 and scores["xg"].between(0, 1).all()
    print(f"wrote {C.XG_SCORES} ({len(scores)} rows, xg in [{xg.min():.4f}, {xg.max():.4f}], mean {xg.mean():.4f})")

    # ---- (5) final one-time test report
    yt, pt = y[idx["test"]], xg[idx["test"]]
    pts = seeds_all[:, idx["test"]]
    print(f"\nTEST (reported once): ensemble log-loss {log_loss(yt, pt):.4f}, AUC {roc_auc_score(yt, pt):.4f} | "
          f"per-seed log-loss {np.mean([log_loss(yt, s) for s in pts]):.4f} +- {np.std([log_loss(yt, s) for s in pts]):.4f}, "
          f"AUC {np.mean([roc_auc_score(yt, s) for s in pts]):.4f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
