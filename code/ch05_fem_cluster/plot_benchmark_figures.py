# -*- coding: utf-8 -*-
"""plot_benchmark_figures.py — Regenerate the manuscript Section 5.1 benchmark
figures from the committed slim result file ``data/fem_benchmark_annual.npz``.

Outputs (PNG, 300 dpi) to ``figures/ch05_fem_cluster/``:
  benchmark_sunny_day_fields.png   6 time-point temperature fields, 2024-08-12
  fem_vs_observed_sensors.png      4 calibrated channels, annual comparison
                                   with 2024-08-12..17 zoom inset
  depth_temperature_profiles.png   depth profiles at x = 600 / 440 mm

These reproductions follow the manuscript layout; the committed PNGs are the
final styled versions used in the manuscript.
"""
import json
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")
FIGDIR = os.path.join(REPO_ROOT, "figures", "ch05_fem_cluster")

VAL0 = pd.Timestamp("2024-02-26")
VAL1 = pd.Timestamp("2024-12-31 23:00")
ZOOM0 = pd.Timestamp("2024-08-12")
ZOOM1 = pd.Timestamp("2024-08-17 23:00")
PICK4 = ['N101', 'N109', 'N102', 'N209']
SHIFT = np.array([600.0, 300.0])
HOURS6 = [6, 9, 12, 15, 18, 21]

plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "axes.unicode_minus": False})


def load():
    z = np.load(os.path.join(DATA, "fem_benchmark_annual.npz"), allow_pickle=True)
    geo = json.load(open(os.path.join(DATA, "dwg_geometry.json")))
    times = pd.DatetimeIndex(z["times"])
    ht = pd.read_csv(os.path.join(DATA, "measured_temperatures_2024.csv"))
    ht["DateTime"] = pd.to_datetime(ht["DateTime"])
    ht = ht.set_index("DateTime").sort_index()
    return z, geo, times, ht


def section_outlines(geo):
    outer = np.array(geo["outer"], float) + SHIFT
    holes = [np.array(geo[k], float) + SHIFT
             for k in ("hole_left", "hole_center", "hole_right")]
    return outer, holes


def draw_section(ax, outer, holes):
    ax.plot(*np.vstack([outer, outer[:1]]).T, 'k-', lw=1.4)
    for h in holes:
        ax.plot(*np.vstack([h, h[:1]]).T, color='0.45', lw=0.9)
    ax.set_aspect('equal')


def fig_sunny_day_fields(z, geo):
    T_day = z["T_day"]                     # (24, N) end-of-hour states
    coords = z["coords"]
    outer, holes = section_outlines(geo)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 5.8), constrained_layout=True)
    vmin, vmax = float(T_day.min()), float(T_day.max())
    sc = None
    for ax, hh in zip(axes.ravel(), HOURS6):
        ax.set_facecolor('0.92')
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=T_day[hh], cmap='jet',
                        s=2.2, vmin=vmin, vmax=vmax)
        draw_section(ax, outer, holes)
        ax.set_xlim(-20, 1220); ax.set_ylim(-20, 620)
        ax.set_title("%02d:00  (%.1f–%.1f °C)" % (hh, T_day[hh].min(), T_day[hh].max()),
                     fontsize=10)
        ax.set_xlabel("x (mm)", fontsize=9); ax.set_ylabel("y (mm)", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.92, pad=0.012,
                 label="Temperature (°C)")
    fig.suptitle("Hourly temperature fields of the benchmark cross-section, "
                 "2024-08-12", fontsize=13)
    out = os.path.join(FIGDIR, "benchmark_sunny_day_fields.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print("saved", out)


def fig_annual_comparison(z, times, ht):
    names = list(z["node_names"])
    Tn = z["T_year_nodes"].astype(float)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    fs, os_ = [], []
    for k, (ax, c) in enumerate(zip(axes.ravel(), PICK4)):
        i = names.index(c)
        fv = Tn[:, i]
        obs = ht[c].reindex(times).values
        m = np.isfinite(obs) & np.isfinite(fv) & (times >= VAL0) & (times <= VAL1)
        f, o = fv[m], obs[m]
        fs.append(f); os_.append(o)
        ax.plot(times[m], obs[m], color='#1f77b4', lw=0.6, ls='-',
                marker='o', ms=1.8, markevery=6, label='Measured', zorder=2)
        ax.plot(times[m], fv[m], color='#d62728', lw=0.8, ls='--',
                label='Calculated', zorder=3)
        rmse = float(np.sqrt(np.mean((f - o) ** 2)))
        r = float(np.corrcoef(f, o)[0, 1])
        ax.set_title("(%s) %s  (RMSE = %.2f °C, r = %.2f)"
                     % ("abcd"[k], c, rmse, r), fontsize=11)
        ax.set_ylabel("Temperature (°C)", fontsize=10)
        ax.grid(True, ls=':', alpha=0.35)
        ax.legend(loc='upper right', fontsize=9, ncol=1, framealpha=0.9)
        ax.set_xlim(VAL0, VAL1)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        for lb in ax.get_xticklabels():
            lb.set_rotation(25); lb.set_ha('right')

        # zoom inset 2024-08-12..17
        zm = m & (times >= ZOOM0) & (times <= ZOOM1)
        x0 = mdates.date2num(ZOOM0); x1 = mdates.date2num(ZOOM1)
        ymin = float(np.min(np.r_[fv[zm], obs[zm]])) - 0.4
        ymax = float(np.max(np.r_[fv[zm], obs[zm]])) + 0.4
        ax.add_patch(Rectangle((x0, ymin), x1 - x0, ymax - ymin,
                               fill=False, edgecolor='0.25', lw=0.9, zorder=3.5))
        IX, IY, IW, IH = 0.41, 0.17, 0.30, 0.31
        axins = ax.inset_axes([IX, IY, IW, IH])
        axins.plot(times[zm], obs[zm], color='#1f77b4', lw=0.7, ls='-', zorder=2)
        axins.plot(times[zm], fv[zm], color='#d62728', lw=0.9, ls='--', zorder=3)
        axins.set_title("%s–%s" % (ZOOM0.strftime('%m-%d'), ZOOM1.strftime('%m-%d')),
                        fontsize=9)
        axins.set_xlim(ZOOM0, ZOOM1)
        axins.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        axins.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        axins.tick_params(labelsize=7)
        axins.patch.set_facecolor('white'); axins.patch.set_alpha(0.95)
        tr = ax.transAxes.inverted()
        ra = tr.transform(ax.transData.transform((x0, ymin)))
        rb = tr.transform(ax.transData.transform((x0, ymax)))
        ax.plot([ra[0], IX], [ra[1], IY + IH], color='0.4', lw=0.7,
                transform=ax.transAxes, zorder=4, clip_on=False)
        ax.plot([rb[0], IX + IW], [rb[1], IY + IH], color='0.4', lw=0.7,
                transform=ax.transAxes, zorder=4, clip_on=False)

    f = np.concatenate(fs); o = np.concatenate(os_)
    rmse = float(np.sqrt(np.mean((f - o) ** 2)))
    mae = float(np.mean(np.abs(f - o)))
    r = float(np.corrcoef(f, o)[0, 1])
    fig.subplots_adjust(bottom=0.21, hspace=0.55, wspace=0.30, left=0.075,
                        right=0.985)
    fig.text(0.5, 0.02,
             "Calculated and measured temperatures at the calibration points, "
             "2024-02-26 to 2024-12-31", ha='center', va='bottom',
             fontsize=12, weight='bold')
    fig.text(0.5, -0.02,
             "Note: overall RMSE = %.3f °C, MAE = %.3f °C and r = %.3f for the "
             "four calibration channels (N = %d)." % (rmse, mae, r, len(f)),
             ha='center', va='bottom', fontsize=10)
    out = os.path.join(FIGDIR, "fem_vs_observed_sensors.png")
    fig.savefig(out, dpi=300, bbox_inches='tight'); plt.close(fig)
    print("saved", out, " overall RMSE=%.3f MAE=%.3f r=%.3f N=%d"
          % (rmse, mae, r, len(f)))


def fig_depth_profiles(z, geo):
    T_day = z["T_day"]
    coords = z["coords"]
    outer, holes = section_outlines(geo)
    colors = {6: '#0b5394', 9: '#38761d', 12: '#b45f06',
              15: '#cc0000', 18: '#674ea7', 21: '#444444'}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)

    def line(x0):
        sel = np.where(np.abs(coords[:, 0] - x0) < 5.0)[0]
        order = np.argsort(coords[sel, 1])
        return coords[sel, 1][order], T_day[:, sel][:, order]   # (24, n)

    for ax, x0, title in zip(axes, (600.0, 440.0),
                             ("(a) Centreline x = 600 mm (dotted segments "
                              "bridge the voids)",
                              "(b) Mid-line of the 2nd rib, x = 440 mm")):
        y, Tl = line(x0)
        for hh in HOURS6:
            ax.plot(Tl[hh], y, color=colors[hh], lw=1.4,
                    ls='-' if x0 == 440.0 else '-',
                    marker='o', ms=2.5, label="%02d:00" % hh)
            if x0 == 600.0:
                # dotted connector across void gaps
                gap = np.where(np.diff(y) > 15)[0]
                for gi in gap:
                    ax.plot([Tl[hh, gi], Tl[hh, gi + 1]],
                            [y[gi], y[gi + 1]], color=colors[hh], lw=1.0,
                            ls=':')
        ax.set_xlabel("Temperature (°C)", fontsize=11)
        ax.set_ylabel("$y$ (mm)", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.grid(True, ls=':', alpha=0.35)
        ax.legend(fontsize=9, loc='lower right')
        # section inset with profile line
        iax = ax.inset_axes([0.02, 0.60, 0.26, 0.36])
        iax.set_facecolor('white')
        draw_section(iax, outer, holes)
        iax.plot([x0, x0], [0, 600], 'r--', lw=1.2)
        iax.set_xlim(-30, 1230); iaq = iax.set_ylim(-30, 630)
        iax.set_xticks([]); iax.set_yticks([])
    fig.suptitle("Temperature profiles along the depth direction on a clear day "
                 "(2024-08-12)", fontsize=13)
    out = os.path.join(FIGDIR, "depth_temperature_profiles.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print("saved", out)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    z, geo, times, ht = load()
    fig_sunny_day_fields(z, geo)
    fig_annual_comparison(z, times, ht)
    fig_depth_profiles(z, geo)


if __name__ == "__main__":
    main()
