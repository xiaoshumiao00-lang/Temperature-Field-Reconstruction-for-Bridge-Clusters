#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate the multi-panel Figure 21 for Section 5.2 of the manuscript.

All 207 bridges are shown with a regional basemap (ETOPO1 DEM + province boundaries,
same style as Fig. 4). Each bridge is represented by its actual typical cross-section
used in the FEM batch run; statistics are computed over the full 207-bridge cluster.

Panel layout
------------
(a) Spatial map: peak top temperature (regional basemap)
(b) Spatial map: maximum vertical gradient (regional basemap)
(c) Violin: peak top temperature by elevation band
(d) Scatter: elevation vs. peak top temperature
(e) Scatter: elevation vs. maximum vertical gradient

Outputs
-------
    ./figures/Fig21_cluster_multi_panel.png   (300 dpi, English labels)
    ./figures/Fig21_cluster_multi_panel.pdf
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import netcdf_file

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "cluster_results" / "bridge_indicators.parquet"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# Basemap assets (same as Fig.4)
ETOPO1_PATH = ROOT.parent / "paper" / "manuscript" / "figures" / "map_assets" / "ETOPO1_Bed_g_gmt4.grd"
PROVINCE_JSON = ROOT.parent / "paper" / "manuscript" / "figures" / "map_assets" / "china_province.geojson"

# Study area (same as Fig.4)
LON_LO, LON_HI = 97.5, 111.5
LAT_LO, LAT_HI = 22.5, 32.5


def load() -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError(f"Run batch_cluster.py first. Missing {DATA}")
    return pd.read_parquet(DATA)


def load_etopo1_subset():
    """Read ETOPO1 subset for the study area."""
    f = netcdf_file(ETOPO1_PATH, 'r', mmap=False)
    x = f.variables['x'].data.copy()
    y = f.variables['y'].data.copy()
    z = f.variables['z'].data.copy()
    f.close()
    ix = np.where((x >= LON_LO) & (x <= LON_HI))[0]
    iy = np.where((y >= LAT_LO) & (y <= LAT_HI))[0]
    lon = x[ix]
    lat = y[iy]
    elev = z[np.ix_(iy, ix)].astype(np.float32)
    return lon, lat, elev


def load_province_boundaries():
    """Return list of (lon_arr, lat_arr) boundary segments."""
    if not PROVINCE_JSON.exists():
        return []
    with open(PROVINCE_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    segs = []
    for feat in data.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") == "Polygon":
            polys = [geom["coordinates"]]
        elif geom.get("type") == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue
        for poly in polys:
            for ring in poly:
                arr = np.asarray(ring)
                if arr.ndim == 2 and arr.shape[1] >= 2 and len(arr) >= 3:
                    segs.append((arr[:, 0], arr[:, 1]))
    return segs


def draw_basemap(ax, with_grid=True):
    """Draw regional ETOPO1 basemap + province boundaries, same style as Fig.4."""
    # ETOPO1 terrain background
    if ETOPO1_PATH.exists():
        lon, lat, elev = load_etopo1_subset()
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
        display = np.clip(elev, 0, 3000)
        ax.imshow(display, extent=extent, origin='lower', cmap='terrain',
                  aspect='equal', interpolation='bilinear', zorder=1,
                  vmin=0, vmax=3000, alpha=0.5)

    # Province boundaries
    for lon_arr, lat_arr in load_province_boundaries():
        ax.plot(lon_arr, lat_arr, color='#4a4a4a', linewidth=0.7, alpha=0.75, zorder=3)

    ax.set_xlim(LON_LO, LON_HI)
    ax.set_ylim(LAT_LO, LAT_HI)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude (°E)", fontsize=11)
    ax.set_ylabel("Latitude (°N)", fontsize=11)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    if with_grid:
        ax.grid(True, linestyle=':', color='gray', alpha=0.5, linewidth=0.5, zorder=2)


def main():
    df = load()
    df = df[df["status"] == "ok"].copy()
    n_total = len(df)

    # Region renaming for legend (all 5 regions)
    region_map = {
        "R1": "Chongqing",
        "R2": "Guizhou",
        "R3": "Guangxi",
        "R4": "W. Yunnan",
        "R5": "NE. Yunnan",
    }
    df["region_en"] = df["region"].map(region_map).fillna(df["region"])

    # Colour palette for region legends (5 distinct colours)
    region_colours = {
        "Chongqing": "#1f77b4",
        "Guizhou": "#ff7f0e",
        "Guangxi": "#2ca02c",
        "W. Yunnan": "#d62728",
        "NE. Yunnan": "#9467bd",
    }

    # Elevation bands for all 207 bridges
    elev_bins = [0, 300, 600, 900, 1200, 2500]
    elev_labels = ["<300", "300–600", "600–900", "900–1200", ">1200"]
    df["elev_band"] = pd.cut(df["elev_m"], bins=elev_bins, labels=elev_labels)

    fig = plt.figure(figsize=(14, 10))

    # (a) Map: peak top temperature for all 207 bridges
    ax1 = fig.add_axes([0.06, 0.55, 0.42, 0.40])
    draw_basemap(ax1)
    sc = ax1.scatter(df["lon"], df["lat"], c=df["T_top_peak"], cmap="hot",
                     s=40, edgecolors="k", linewidths=0.3, vmin=25, vmax=55, zorder=5)
    ax1.set_title(f"(a) Peak top-surface temperature (°C), N = {n_total}", fontsize=12)
    cbar1 = fig.colorbar(sc, ax=ax1, shrink=0.7, pad=0.02)
    cbar1.set_label(r"$T_{\mathrm{peak}}$ (°C)", fontsize=10)

    # (b) Map: max vertical gradient for all 207 bridges
    ax2 = fig.add_axes([0.52, 0.55, 0.40, 0.40])
    draw_basemap(ax2)
    sc2 = ax2.scatter(df["lon"], df["lat"], c=df["max_vertical_gradient"], cmap="YlOrRd",
                      s=40, edgecolors="k", linewidths=0.3, vmin=0, vmax=10, zorder=5)
    ax2.set_title(f"(b) Maximum vertical gradient (°C), N = {n_total}", fontsize=12)
    cbar2 = fig.colorbar(sc2, ax=ax2, shrink=0.7, pad=0.02)
    cbar2.set_label(r"$\Delta T_{\max}$ (°C)", fontsize=10)

    # (c) Violin by elevation band (all 207 bridges)
    ax3 = fig.add_axes([0.06, 0.08, 0.25, 0.40])
    data = [df[df["elev_band"] == lab]["T_top_peak"].dropna().values for lab in elev_labels]
    parts = ax3.violinplot(data, positions=range(len(elev_labels)), showmeans=True, showmedians=True)
    for pc, colour in zip(parts["bodies"], plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(elev_labels)))):
        pc.set_facecolor(colour)
        pc.set_alpha(0.8)
    ax3.set_xticks(range(len(elev_labels)))
    ax3.set_xticklabels(elev_labels, fontsize=9)
    ax3.set_xlabel("Elevation band (m)", fontsize=11)
    ax3.set_ylabel("Peak top temperature (°C)", fontsize=11)
    ax3.set_title(f"(c) Distribution by elevation, N = {n_total}", fontsize=12)
    ax3.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax3.grid(True, axis="y", alpha=0.3)

    # (d) Elevation vs peak top temperature (all 207 bridges, region-coloured)
    ax4 = fig.add_axes([0.35, 0.08, 0.25, 0.40])
    for region, colour in region_colours.items():
        sub = df[df["region_en"] == region]
        if len(sub) == 0:
            continue
        ax4.scatter(sub["elev_m"], sub["T_top_peak"], c=colour, s=35,
                    edgecolors="k", linewidths=0.3, alpha=0.7, label=region)
    z = np.polyfit(df["elev_m"], df["T_top_peak"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df["elev_m"].min(), df["elev_m"].max(), 100)
    ax4.plot(x_line, p(x_line), "k--", linewidth=1.0, label=f"Trend: {z[0]:.4f} °C/m")
    ax4.set_xlabel("Elevation (m)", fontsize=11)
    ax4.set_ylabel("Peak top temperature (°C)", fontsize=11)
    ax4.set_title(f"(d) Elevation vs. peak temperature, N = {n_total}", fontsize=12)
    ax4.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax4.legend(loc="upper right", fontsize=7, ncol=2)
    ax4.grid(True, alpha=0.3)

    # (e) Elevation vs vertical gradient (all 207 bridges, region-coloured)
    ax5 = fig.add_axes([0.66, 0.08, 0.28, 0.40])
    for region, colour in region_colours.items():
        sub = df[df["region_en"] == region]
        if len(sub) == 0:
            continue
        ax5.scatter(sub["elev_m"], sub["max_vertical_gradient"], c=colour, s=35,
                    edgecolors="k", linewidths=0.3, alpha=0.7, label=region)
    z2 = np.polyfit(df["elev_m"], df["max_vertical_gradient"], 1)
    p2 = np.poly1d(z2)
    ax5.plot(x_line, p2(x_line), "k--", linewidth=1.0, label=f"Trend: {z2[0]:.5f} °C/m")
    ax5.set_xlabel("Elevation (m)", fontsize=11)
    ax5.set_ylabel("Max vertical gradient (°C)", fontsize=11)
    ax5.set_title(f"(e) Elevation vs. vertical gradient, N = {n_total}", fontsize=12)
    ax5.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    ax5.legend(loc="upper right", fontsize=7, ncol=2)
    ax5.grid(True, alpha=0.3)

    fig.text(0.5, 0.01,
             "Figure 21 Cluster-scale temperature indicators for 207 concrete box-girder bridges "
             "(FEM driven by downscaled ERA5-Land boundary conditions, actual typical cross-sections).",
             ha="center", fontsize=12)

    fig.savefig(OUT / "Fig21_cluster_multi_panel.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "Fig21_cluster_multi_panel.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT / 'Fig21_cluster_multi_panel.png'}")


if __name__ == "__main__":
    main()
