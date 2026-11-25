"""Phase 6: figures for the final report that show the CNN's input (no training, no new results).

Writes final_report/figures/cnn_tensor.png: the 6-channel 40x60 spatial tensor of one TEST-split shot (data/processed/spatial.npz),
as a composite pitch view plus the six channels. The shot is chosen by a fixed rule (first test goal whose frame shows the goalkeeper and at
least 8 players), not by looking at model output.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "final_report" / "figures"
NAMES = ["Attackers", "Defenders", "Opponent goalkeeper", "Shooter", "Visible area", "Distance to goal"]


def pick(tab, X):
    for i in tab.index[(tab["split"] == "test") & (tab["goal"] == 1)]:
        x = X[i].astype("float32")
        if x[2].sum() >= 1 and x[0].sum() + x[1].sum() >= 8 and x[3].sum() == 1:
            return int(i)
    raise RuntimeError("no shot met the rule")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tab = pd.read_csv(ROOT / "data" / "processed" / "tabular.csv")
    X = np.load(ROOT / "data" / "processed" / "spatial.npz")["X"]
    i = pick(tab, X)
    T = X[i].astype("float32")
    row = tab.loc[i]
    comp = np.zeros((40, 60, 3))
    comp[:] = (0.33, 0.33, 0.33)                                   # outside the camera view: unknown
    comp[T[4] > 0.5] = (0.91, 0.945, 0.92)                         # visible
    for ch, col in ((1, (0.85, 0.33, 0.31)), (0, (0.16, 0.44, 0.84)), (2, (0.94, 0.75, 0.13)), (3, (0, 0, 0))):
        comp[T[ch] > 0.5] = col

    fig = plt.figure(figsize=(10.2, 3.9))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.55, 1.55, 1, 1, 1], wspace=0.12, hspace=0.18)
    ax = fig.add_subplot(gs[:, 0:2])
    ax.imshow(comp, origin="lower", interpolation="nearest", aspect="equal")
    ax.set_title("All channels, one shot (pitch seen from above, attacking right)", fontsize=8.5, loc="left")
    ax.axvline(59.5, ymin=15 / 40, ymax=25 / 40, color="#168c83", lw=4)
    ax.set_xticks([]); ax.set_yticks([])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in ((comp[0, 0] if False else (0.16, 0.44, 0.84), "attacker"), ((0.85, 0.33, 0.31), "defender"),
                                                             ((0.94, 0.75, 0.13), "goalkeeper"), ((0, 0, 0), "shooter"), ((0.33, 0.33, 0.33), "outside camera view"))],
              fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    for k in range(6):
        a = fig.add_subplot(gs[k // 3, 2 + k % 3])
        a.imshow(T[k], origin="lower", cmap="Greys", vmin=0, vmax=1, interpolation="nearest", aspect="equal")
        a.set_title(f"{k}: {NAMES[k]}", fontsize=7.2, loc="left")
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"CNN input for one test-split shot ({row['team']}, minute {int(row['minute'])}, goal): a 6 x 40 x 60 tensor, 2-yard cells", fontsize=9.5, y=0.99)
    fig.savefig(OUT / "cnn_tensor.png", dpi=170, bbox_inches="tight")
    print("shot row", i, row["team"], int(row["minute"]), "->", OUT / "cnn_tensor.png")


if __name__ == "__main__":
    main()
