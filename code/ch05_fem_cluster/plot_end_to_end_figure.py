#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 19 / Section 5.2 — End-to-end pipeline diagram.

Sub-panels:
  (a) Workflow from satellite/weather data to bridge temperature field,
      including the web-app deployment target.
  (b) Screenshot of the Bridge Temperature Field Calculator web interface.

Output
------
    ./figures/Fig19_end_to_end_pipeline.png  (300 dpi, English labels)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "stix"

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def draw_box(ax, x, y, w, h, text, colour="white", fontsize=9):
    rect = mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                     boxstyle="round,pad=0.02,rounding_size=0.05",
                                     facecolor=colour, edgecolor="black", linewidth=1.0)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, wrap=True)


def draw_arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.04, label, ha="center", va="bottom",
                fontsize=8, style="italic")


def main():
    fig = plt.figure(figsize=(12, 6))

    # -------------------------------------------------------------------------
    # Panel (a): workflow
    # -------------------------------------------------------------------------
    ax1 = fig.add_axes([0.06, 0.10, 0.42, 0.85])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis("off")
    ax1.set_title("(a) End-to-end pipeline", fontsize=12, pad=10)

    # Data sources (top row)
    draw_box(ax1, 1.5, 8.5, 2.2, 0.9, "ERA5-Land\n0.1° reanalysis", "#e8f4f8")
    draw_box(ax1, 4.0, 8.5, 2.0, 0.9, "MODIS LST\n1 km", "#e8f4f8")
    draw_box(ax1, 6.4, 8.5, 2.0, 0.9, "SRTM DEM\n30 m", "#e8f4f8")

    # Stage 1: downscaling
    draw_box(ax1, 3.5, 6.4, 3.6, 1.0, "Stage 1\nXGBoost air-temperature downscaling", "#fff2cc")
    draw_arrow(ax1, 1.5, 8.05, 2.5, 6.9, "")
    draw_arrow(ax1, 4.0, 8.05, 3.7, 6.9, "")
    draw_arrow(ax1, 6.4, 8.05, 4.4, 6.9, "")

    # Hourly boundary conditions
    draw_box(ax1, 3.5, 4.4, 3.0, 1.0, "Hourly bridge-level\nboundary conditions", "#e8f4f8")
    draw_arrow(ax1, 3.5, 5.9, 3.5, 4.9, "")

    # Stage 2: FEM
    draw_box(ax1, 3.5, 2.4, 3.0, 1.0, "Stage 2\n2-D transient thermal FEM", "#d9ead3")
    draw_arrow(ax1, 3.5, 3.9, 3.5, 2.9, "")

    # User geometry input
    draw_box(ax1, 7.5, 2.4, 2.2, 1.0, "DXF / parametric\nbox section", "#f3f3f3")
    draw_arrow(ax1, 6.5, 2.4, 6.4, 2.4, "")

    # Outputs
    draw_box(ax1, 1.8, 0.6, 2.0, 0.8, "Cloud maps\n& time series", "#f4cccc")
    draw_box(ax1, 5.2, 0.6, 2.0, 0.8, "Cluster-scale\nindicators", "#f4cccc")
    draw_arrow(ax1, 3.0, 1.9, 2.4, 1.0, "")
    draw_arrow(ax1, 4.0, 1.9, 4.6, 1.0, "")

    # Web app highlight
    draw_box(ax1, 8.5, 0.6, 2.6, 0.8, "Web app\n(open-source)", "#cfe2f3")
    draw_arrow(ax1, 6.2, 0.6, 7.2, 0.6, "")

    # -------------------------------------------------------------------------
    # Panel (b): screenshot
    # -------------------------------------------------------------------------
    ax2 = fig.add_axes([0.54, 0.10, 0.42, 0.85])
    img_path = ROOT / "webapp_inputs.png"
    if img_path.exists():
        img = plt.imread(str(img_path))
        ax2.imshow(img)
    else:
        ax2.text(0.5, 0.5, "Screenshot not available",
                 transform=ax2.transAxes, ha="center", va="center")
    ax2.axis("off")
    ax2.set_title("(b) Bridge Temperature Field Calculator web interface", fontsize=12, pad=10)

    fig.savefig(OUT / "Fig19_end_to_end_pipeline.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "Fig19_end_to_end_pipeline.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT / 'Fig19_end_to_end_pipeline.png'}")


if __name__ == "__main__":
    main()
