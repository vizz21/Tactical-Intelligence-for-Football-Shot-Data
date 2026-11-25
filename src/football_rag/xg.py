"""xG model loading and scoring (tabular ANN from Phase 2, saved by scripts/phase4_build_xg.py)."""
import json
import sys

import numpy as np
import pandas as pd
import torch

from .config import SCRIPTS, XG_MODEL_DIR

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from phase2_train import ANN  # noqa: E402  reuse the Phase 2 model class, do not rewrite it


class XGModel:
    """Seed ensemble of ANNs plus the train-only scaler. predict() returns the seed-averaged probability."""

    def __init__(self, models, mean, scale, feat_cols):
        self.models, self.mean, self.scale, self.feat_cols = models, mean, scale, feat_cols

    def predict_seeds(self, df: pd.DataFrame) -> np.ndarray:
        """(n_seeds, n_rows) probabilities for a dataframe that has the feature columns."""
        x = ((df[self.feat_cols].to_numpy(dtype="float64") - self.mean) / self.scale).astype("float32")
        xt = torch.from_numpy(x)
        with torch.no_grad():
            return np.stack([torch.sigmoid(m(xt)).numpy() for m in self.models])

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_seeds(df).mean(0)


def load_xg(model_dir=XG_MODEL_DIR) -> XGModel:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    sc = np.load(model_dir / "scaler.npz")
    models = []
    for s in range(cfg["n_seeds"]):
        m = ANN(len(cfg["feat_cols"]), tuple(cfg["hidden"]), cfg["dropout"])
        m.load_state_dict(torch.load(model_dir / f"seed{s}.pt", map_location="cpu", weights_only=True))
        models.append(m.eval())
    return XGModel(models, sc["mean"], sc["scale"], cfg["feat_cols"])
