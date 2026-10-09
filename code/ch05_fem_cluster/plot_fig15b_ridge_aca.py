"""Fig. 15(b) ridge plot — original manuscript style, corrected data.

Style replicated from the original panel (Clipboard_Screenshot):
- pastel translucent fills (alpha ~0.75), ridges OVERLAP vertically
- thin full-width separator lines at each ridge base
- median line up to the curve, label just right of the line top ("37.2°C")
- legend: 2 rows x 3 cols, frameless, top-left
  row1: Chongqing(R1)  NE. Yunnan(R5)  W. Yunnan(R4)
  row2: Guangxi(R3)    Guizhou(R2)
- Times New Roman, near-square aspect
Only change vs. original: corrected CEBM data (28-43 degC) and region names
instead of R1-R5 codes.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
import matplotlib.patheffects as pe

STROKE = [pe.withStroke(linewidth=2.2, foreground="white")]

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "cluster_results" / "bridge_indicators.csv"
OUT = ROOT / "figures" / "ch05_fem_cluster"

LOBO_RMSE = {"M01": 1.48, "M03": 1.61, "M04": 2.16, "M07": 2.14}
SIGMA_DEFAULT = float(np.mean(list(LOBO_RMSE.values())))
S_TOP, N_MC, SEED = 0.90, 200, 42

REGION_ORDER = ["R1", "R2", "R3", "R4", "R5"]          # bottom -> top
FILL = {"R1": "#F2A35C", "R2": "#6AA5C8", "R3": "#3FAE8A",
        "R4": "#F0C75E", "R5": "#E194B8"}
EDGE = {"R1": "#D0823A", "R2": "#4A7FA0", "R3": "#2A8266",
        "R4": "#C9A13B", "R5": "#B96A90"}
LABELS = {"R1": "Fengjie", "R2": "Qiannan", "R3": "Laibin",
          "R4": "Baoshan", "R5": "Zhaotong"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.default": "it",
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "legend.fontsize": 9.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 400,
    "savefig.dpi": 400,
})

# ---------------- data ----------------
df = pd.read_csv(DATA)
df = df[df["status"] == "ok"].reset_index(drop=True)
sigma = np.where(df["bridge_id"].isin(LOBO_RMSE),
                 df["bridge_id"].map(LOBO_RMSE).fillna(SIGMA_DEFAULT),
                 SIGMA_DEFAULT).astype(float)
rng = np.random.RandomState(SEED)
t_top_mc = (df["T_top_peak"].values.astype(float)[:, None]
            + S_TOP * sigma[:, None] * rng.normal(0.0, 1.0, size=(len(df), N_MC)))

# ---------------- figure ----------------
fig, ax = plt.subplots(figsize=(4.3, 4.0))

xgrid = np.linspace(24, 50, 800)
BW = 0.12                  # narrow bandwidth: resolve per-bridge mixture structure
ymax = max(gaussian_kde(t_top_mc[df["region"].values == r].ravel(),
                        bw_method=BW)(xgrid).max() for r in REGION_ORDER)

SPAN = 1.0                 # band spacing
H = 0.92                   # ridge peak stays below the separator line above

for j, r in enumerate(REGION_ORDER):
    v = t_top_mc[df["region"].values == r].ravel()
    y = gaussian_kde(v, bw_method=BW)(xgrid) / ymax * H
    base = j * SPAN
    ax.axhline(base, color="0.45", lw=0.6, zorder=1)
    ax.fill_between(xgrid, base, base + y, color=FILL[r], alpha=0.78,
                    lw=0, zorder=2)
    ax.plot(xgrid, base + y, color=EDGE[r], lw=0.9, zorder=3)
    med = np.median(v)
    ym = np.interp(med, xgrid, y)
    ax.plot([med, med], [base, base + ym], color="0.12", lw=1.2, zorder=4)
    if j in (0, 4):
        # Fengjie (bottom) and Zhaotong (top): label at the line top (approved)
        ax.annotate(f"{med:.1f}°C", xy=(med, base + ym), xytext=(4, 1),
                    textcoords="offset points", fontsize=9.5, color="0.08",
                    ha="left", va="bottom", zorder=6, path_effects=STROKE)
    else:
        # bands 2-4: label on the right SHOULDER of the peak (close to the
        # median line), floating just above the curve on white background
        below = np.where((xgrid > med) & (y < 0.55))[0]
        x_lab = xgrid[below[0]] + 0.2 if len(below) else med + 2.0
        y_lab = base + np.interp(x_lab, xgrid, y) + 0.12
        ax.text(x_lab, y_lab, f"{med:.1f}°C", fontsize=9.5, color="0.08",
                ha="left", va="bottom", zorder=6, path_effects=STROKE)
    # direct region label at the right of each ridge (no legend needed)
    ax.text(49.7, base + 0.22, LABELS[r], fontsize=10, color=EDGE[r],
            fontweight="bold", ha="right", va="center", zorder=6,
            path_effects=STROKE)

ax.set_xlim(25, 50)
ax.set_xticks(np.arange(25, 51, 5))
ax.set_ylim(-0.12, 4 * SPAN + H + 0.12)
ax.set_yticks([])
ax.set_xlabel("Peak top-slab temperature (°C)")
ax.set_ylabel("Relative density")
ax.tick_params(length=3, width=0.8)

fig.tight_layout(pad=0.6)
fig.savefig(OUT / "fig15b_peak_ttop_ridge.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fig15b_peak_ttop_ridge.pdf", bbox_inches="tight", facecolor="white")
print("Saved:", OUT / "fig15b_peak_ttop_ridge.png")
