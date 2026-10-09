"""Redraw manuscript Fig. 15 (repo Figure 22): 200-member Monte Carlo two-scale
uncertainty propagation.

Fix: panel (b) previously used the outdated equivalent-temperature proxy
(T_top ~ T_air + solar_delta, giving 44-61 degC), which contradicts the CEBM
cluster results (T_top_peak = 28.3-43.2 degC, Fig. 14 / Section 5.2).
This script rebuilds panel (b) from the CEBM deterministic peak top-slab
temperatures (bridge_indicators.csv, T_top_peak) with Gaussian air-temperature
perturbations propagated through the top-slab sensitivity S=0.9:
    T_top_mc(i, mc) = T_top_peak(i) + S * eps,  eps ~ N(0, sigma_i)
where sigma_i is the LOBO-calibrated air-temperature uncertainty (Table 3
values for monitored bridges; distance-model mean 1.85 degC otherwise).

Panel (a) is regenerated unchanged in concept: M04 48 h ensemble cloud
(10-11 Aug 2024) with 68%/95% prediction intervals, sigma_M04 = 2.16 degC
(manuscript Table 3).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]          # repo root
PROJ = ROOT.parent                                   # 11：气象桥梁温度场
DATA = ROOT / "data" / "cluster_results" / "bridge_indicators.csv"
BND = PROJ / "07_GNN模型与跨桥泛化" / "stage1_deliverables" / "boundary_2024_2026.npz"
OUT = ROOT / "figures" / "ch05_fem_cluster"

# LOBO RMSE from manuscript Table 3 (QC-passed monitored bridges)
LOBO_RMSE = {"M01": 1.48, "M03": 1.61, "M04": 2.16, "M07": 2.14}
SIGMA_DEFAULT = float(np.mean(list(LOBO_RMSE.values())))   # 1.85 degC
S_TOP = 0.90          # air -> top-slab sensitivity (linear FEM regime)
N_MC = 200
SEED = 42

REGION_ORDER = ["R1", "R2", "R3", "R4", "R5"]
REGION_COLORS = {
    "R1": "#E67E22",   # orange
    "R2": "#2E86C1",   # blue
    "R3": "#27AE60",   # green
    "R4": "#F4D03F",   # yellow
    "R5": "#C77DA1",   # mauve
}
# Region names consistent with Fig. 14 legend (R1-R5 codes dropped per author)
REGION_LABELS = {
    "R1": "Chongqing",
    "R2": "Guizhou",
    "R3": "Guangxi",
    "R4": "W. Yunnan",
    "R5": "NE. Yunnan",
}
RIDGE_TOP_TO_BOTTOM = ["R5", "R4", "R3", "R2", "R1"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.unicode_minus": False,
})

# ---------------- data ----------------
df = pd.read_csv(DATA)
df = df[df["status"] == "ok"].reset_index(drop=True)
assert df["T_top_peak"].between(20, 50).all()

d = np.load(BND, allow_pickle=True)
t2m = d["t2m"]
bridge_ids = list(d["bridge_ids"])
time_index = pd.DatetimeIndex(d["time_index"])

rng = np.random.RandomState(SEED)

# per-bridge air-temperature uncertainty (degC)
sigma = np.where(df["bridge_id"].isin(LOBO_RMSE),
                 df["bridge_id"].map(LOBO_RMSE).fillna(SIGMA_DEFAULT),
                 SIGMA_DEFAULT).astype(float)

# Monte Carlo peak top-slab temperature: (207, 200)
t_top = df["T_top_peak"].values.astype(float)[:, None]
eps = rng.normal(0.0, 1.0, size=(len(df), N_MC))
t_top_mc = t_top + S_TOP * sigma[:, None] * eps

print("Pooled MC range: %.2f - %.2f degC" % (t_top_mc.min(), t_top_mc.max()))
for r in REGION_ORDER:
    v = t_top_mc[df["region"].values == r]
    print(f"  {r}: median {np.median(v):.1f} degC, "
          f"95%CI [{np.percentile(v, 2.5):.1f}, {np.percentile(v, 97.5):.1f}]")

# ---------------- figure ----------------
fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 3.3))

# (a) M04 ensemble cloud, 10-11 Aug 2024 (48 h)
m04 = bridge_ids.index("M04")
hours = np.where((time_index >= "2024-08-10") & (time_index < "2024-08-12"))[0][:48]
tt = time_index[hours]
det = t2m[m04, hours].astype(float)
s4 = LOBO_RMSE["M04"]
ens = det[None, :] + rng.normal(0.0, s4, size=(N_MC, len(hours)))

x = np.arange(len(hours))
axa.fill_between(x, det - 1.96 * s4, det + 1.96 * s4,
                 color="#A6C8E0", alpha=0.55, lw=0, label="95% prediction interval")
axa.fill_between(x, det - s4, det + s4,
                 color="#5B9BD5", alpha=0.55, lw=0, label="68% prediction interval")
axa.plot(x, np.median(ens, axis=0), color="black", lw=1.6, label="Ensemble median")
axa.plot(x, det, color="#E8912D", lw=1.2, ls="--", label="Deterministic $T_{air}$")
axa.set_ylim(22, 40)
axa.set_ylabel("Air temperature (°C)")
axa.set_xlabel("Date")
step = 3
axa.set_xticks(x[::step])
axa.set_xticklabels([tt[i].strftime("%m/%d %H:%M") for i in x[::step]],
                    rotation=30, ha="right", fontsize=7)
axa.legend(loc="upper left", fontsize=7, ncol=2, frameon=True, edgecolor="0.7")
axa.set_title("(a) 200-member $T_{air}$ ensemble at M04", fontsize=10)

# (b) ridge density by region (bottom -> top: R1, R2, R3, R4, R5)
xgrid = np.linspace(24, 50, 700)
ymax = 0.0
dens = {}
for r in REGION_ORDER:
    v = t_top_mc[df["region"].values == r].ravel()
    kde = gaussian_kde(v, bw_method=0.35)
    dens[r] = kde(xgrid)
    ymax = max(ymax, dens[r].max())

for j, r in enumerate(REGION_ORDER):                 # j=0 -> bottom (R1)
    base = j * 1.15
    y = dens[r] / ymax * 0.95
    axb.axhline(base, color="0.6", lw=0.5, zorder=1)
    axb.fill_between(xgrid, base, base + y, color=REGION_COLORS[r], alpha=0.75, lw=0)
    axb.plot(xgrid, base + y, color="0.25", lw=0.7)
    med = np.median(t_top_mc[df["region"].values == r])
    ym = np.interp(med, xgrid, y)
    axb.plot([med, med], [base, base + ym], color="0.15", lw=1.2)
    axb.annotate(f"{med:.1f}°C", xy=(med, base + ym), xytext=(6, 2),
                 textcoords="offset points", fontsize=8, color="0.1")

axb.set_ylim(-0.15, 4 * 1.15 + 2.6)   # extra headroom for the legend
axb.set_yticks([])
axb.set_ylabel("Relative density")
axb.set_xlabel("Peak top-slab temperature (°C)")
axb.set_xlim(27, 48)
# column-major legend fill -> displayed as row1: R1 R5 R4, row2: R3 R2
handles = [Patch(facecolor=REGION_COLORS[r], edgecolor="none", label=REGION_LABELS[r])
           for r in ["R1", "R3", "R5", "R2", "R4"]]
axb.legend(handles=handles, loc="upper left", fontsize=6.5, ncol=3,
           frameon=True, edgecolor="0.7", columnspacing=0.7, handlelength=1.1)
axb.set_title("(b) 200-member peak $T_{top}$ distributions by region", fontsize=10)

for ax in (axa, axb):
    ax.tick_params(length=2.5, width=0.8)

fig.tight_layout(pad=1.2)
fig.savefig(OUT / "monte_carlo_uncertainty.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "monte_carlo_uncertainty.pdf", bbox_inches="tight", facecolor="white")
print("Saved:", OUT / "monte_carlo_uncertainty.png")

# ---------------- standalone square panel (b), original aspect ~1:1 ----------------
figb, axc = plt.subplots(figsize=(4.4, 4.2))

xgrid = np.linspace(24, 50, 700)
for j, r in enumerate(REGION_ORDER):                 # bottom -> top: R1..R5
    base = j * 1.0
    y = dens[r] / ymax * 1.25
    axc.axhline(base, color="0.55", lw=0.6, zorder=1)
    axc.fill_between(xgrid, base, base + y, color=REGION_COLORS[r], alpha=0.85, lw=0)
    axc.plot(xgrid, base + y, color="0.3", lw=0.7)
    med = np.median(t_top_mc[df["region"].values == r])
    ym = np.interp(med, xgrid, y)
    axc.plot([med, med], [base, base + ym], color="0.1", lw=1.3)
    axc.annotate(f"{med:.1f}°C", xy=(med, base + 0.62 * ym), xytext=(8, 0),
                 textcoords="offset points", fontsize=9, color="0.05",
                 ha="left", va="center")

axc.set_ylim(-0.1, 4 * 1.0 + 2.9)                    # headroom for legend
axc.set_yticks([])
axc.set_ylabel("Relative density")
axc.set_xlabel("Peak top-slab temperature (°C)")
axc.set_xlim(25, 50)
axc.set_xticks(np.arange(25, 51, 5))
# column-major legend fill -> displayed as row1: R1 R5 R4, row2: R3 R2
handles = [Patch(facecolor=REGION_COLORS[r], edgecolor="none", label=REGION_LABELS[r])
           for r in ["R1", "R3", "R5", "R2", "R4"]]
axc.legend(handles=handles, loc="upper left", fontsize=8, ncol=2,
           frameon=False, columnspacing=1.0, handlelength=1.3,
           bbox_to_anchor=(0.0, 1.02))
axc.tick_params(length=3, width=0.8)

figb.tight_layout(pad=0.8)
figb.savefig(OUT / "fig15b_peak_ttop_ridge.png", dpi=400, bbox_inches="tight", facecolor="white")
figb.savefig(OUT / "fig15b_peak_ttop_ridge.pdf", bbox_inches="tight", facecolor="white")
print("Saved:", OUT / "fig15b_peak_ttop_ridge.png")
