# -*- coding: utf-8 -*-
"""s1_plot_m03_full_year_en.py

Publication-quality English version of M03 full-year temperature downscaling figure.
Follows academic visualization standards: Times New Roman, 300 dpi, colorblind-safe palette.
Trains LOBO (M01/M04/M07 train, M03 test) and plots observed vs downscaled only.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import rcParams
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Academic visualization standards: Times New Roman for all English text
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["mathtext.fontset"] = "stix"
rcParams["axes.unicode_minus"] = True

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "s1_plots")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
from s1_xgb_screen_v6 import load_data, build_features

PARAMS = dict(max_depth=6, learning_rate=0.03, n_estimators=500,
              subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
              eval_metric="rmse", n_jobs=-1, random_state=0, reg_lambda=1.0,
              min_child_weight=3)

data = load_data(["M05", "M06", "M02"], clean=True)
L = data["L"]
bids = data["mon_bids"]
h = bids.index("M03")
tr = [i for i in range(L) if i != h]

Xtr, ytr = [], []
for i in tr:
    Mi = data["M"][i]
    Xi = build_features(data, i)
    Ri = data["land_t2m"][i] - data["Y"][i]
    Xtr.append(Xi[Mi])
    ytr.append(Ri[Mi])
Xtr = np.concatenate(Xtr)
ytr = np.concatenate(ytr)
print(f"Training samples: {len(ytr)} from {[bids[i] for i in tr]}")

Mh = data["M"][h]
Xte = build_features(data, h)[Mh]
land = data["land_t2m"][h, Mh]
obs = data["Y"][h, Mh]
t = data["times"][Mh]

model = xgb.XGBRegressor(**PARAMS)
model.fit(Xtr, ytr)
Rpred = model.predict(Xte).astype(np.float32)
pred = land - Rpred

rmse = float(np.sqrt(mean_squared_error(obs, pred)))
corr = float(np.corrcoef(obs, pred)[0, 1])
print(f"M03: RMSE={rmse:.3f} degC, corr={corr:.3f}")

# Sort by time
df = pd.DataFrame({"time": pd.DatetimeIndex(t), "obs": obs, "pred": pred})
df = df.sort_values("time").reset_index(drop=True)

# Zoom window: 2025-02-05 ~ 2025-02-25
zoom_start = pd.Timestamp("2025-02-05")
zoom_end = pd.Timestamp("2025-02-25")
dfz = df[(df["time"] >= zoom_start) & (df["time"] <= zoom_end)].copy()
if len(dfz) > 0:
    zrmse = float(np.sqrt(mean_squared_error(dfz["obs"], dfz["pred"])))
    zcorr = float(np.corrcoef(dfz["obs"], dfz["pred"])[0, 1])
    print(f"Zoom {zoom_start.date()} ~ {zoom_end.date()}: RMSE={zrmse:.3f} degC, corr={zcorr:.3f}")
else:
    zrmse = None
    zcorr = None

# ---------------------------------------------------------------------------
# Plot: single panel (residual subplot removed per user request)
# ---------------------------------------------------------------------------
fig_width = 6.9
fig_height = 3.1
fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

# Main panel
ax.plot(df["time"], df["obs"], "k-", lw=0.7, label="Observed", zorder=3)
ax.plot(df["time"], df["pred"], "#CC3311", lw=0.7, alpha=0.85, label="Downscaled", zorder=3)
ax.set_ylabel("Air temperature (\u00b0C)", fontsize=13)
ax.set_xlabel("Time", fontsize=13)
ax.set_title(
    "Hourly observed and downscaled air temperature at M03 in 2025",
    fontsize=14, fontweight="bold", pad=8
)
ax.legend(loc="upper left", fontsize=12, framealpha=0.9, edgecolor="gray")
ax.tick_params(axis="both", which="major", labelsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.grid(True, alpha=0.3, lw=0.5)

# Inset: middle-lower blank area (verified 0% overlap with main curves)
if len(dfz) > 0:
    # Inset axes position [left, bottom, width, height] in axes coordinates
    ax_inset = ax.inset_axes([0.45, 0.15, 0.33, 0.25])
    ax_inset.plot(dfz["time"], dfz["obs"], "k-", lw=1.0, label="Observed", zorder=3)
    ax_inset.plot(dfz["time"], dfz["pred"], "#CC3311", lw=1.0, alpha=0.9, label="Downscaled", zorder=3)
    ax_inset.set_xlim(zoom_start, zoom_end)
    y_min = dfz[["obs", "pred"]].min().min() - 1.5
    y_max = dfz[["obs", "pred"]].max().max() + 1.5
    ax_inset.set_ylim(y_min, y_max)
    # Move y-ticks to the right so they do not overlap the main axis labels
    ax_inset.yaxis.tick_right()
    ax_inset.yaxis.set_label_position("right")
    ax_inset.tick_params(axis="both", which="major", labelsize=8, rotation=15)
    ax_inset.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_inset.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax_inset.grid(True, alpha=0.3, lw=0.5)

    # ---- Standard zoom indicator: rectangle on main axes + TWO connecting lines ----
    from matplotlib.patches import Rectangle, ConnectionPatch

    x0 = mdates.date2num(zoom_start)
    x1 = mdates.date2num(zoom_end)
    y0 = dfz[["obs", "pred"]].min().min() - 1.0
    y1 = dfz[["obs", "pred"]].max().max() + 1.0

    # Rectangle highlighting the zoomed interval (over the Feb curve)
    rect = Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        linewidth=1.2, edgecolor="#CC3311", facecolor="#CC3311",
        alpha=0.12, zorder=2
    )
    ax.add_patch(rect)
    ax.plot(
        [x0, x1, x1, x0, x0],
        [y0, y0, y1, y1, y0],
        color="#CC3311", lw=1.1, alpha=0.6, zorder=3
    )

    # Two connecting lines from the right side of the rectangle to the left side of the inset.
    con1 = ConnectionPatch(
        xyA=(x1, y1), coordsA=ax.transData,
        xyB=(0, 1), coordsB=ax_inset.transAxes,
        color="#CC3311", lw=0.9, alpha=0.40, linestyle="-"
    )
    con2 = ConnectionPatch(
        xyA=(x1, y0), coordsA=ax.transData,
        xyB=(0, 0), coordsB=ax_inset.transAxes,
        color="#CC3311", lw=0.9, alpha=0.40, linestyle="-"
    )
    fig.add_artist(con1)
    fig.add_artist(con2)

# Tight layout and save
plt.tight_layout()
fig_path = os.path.join(OUT, "figure_10_m03_full_year.png")
fig.savefig(fig_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print(f"Saved: {fig_path}")
