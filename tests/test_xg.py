import numpy as np
import pandas as pd

from football_rag import config as C
from football_rag.xg import load_xg


def test_scores_file_shape_and_range():
    s = pd.read_csv(C.XG_SCORES)
    assert list(s.columns) == ["row_idx", "shot_id", "xg", "split", "in_sample"]
    assert len(s) == 10233 and s["shot_id"].is_unique
    assert s["xg"].between(0, 1).all()
    assert (s["in_sample"] == (s["split"] == "train")).all()


def test_loaded_model_reproduces_saved_scores():
    tab = pd.read_csv(C.TABULAR)
    s = pd.read_csv(C.XG_SCORES)
    p = load_xg().predict(tab)
    assert p.shape == (10233,) and ((p >= 0) & (p <= 1)).all()
    assert np.allclose(p, s["xg"].to_numpy(), atol=1e-5)
    assert (tab["shot_id"] == s["shot_id"]).all()


def test_mean_xg_close_to_goal_rate():
    s = pd.read_csv(C.XG_SCORES)
    tab = pd.read_csv(C.TABULAR)
    val = s["split"] == "val"
    assert abs(s.loc[val, "xg"].mean() - tab.loc[val, "goal"].mean()) < 0.01
