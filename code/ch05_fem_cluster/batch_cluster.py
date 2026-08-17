#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch compute simplified FEM temperature indicators for the 207-bridge corridor.

Design choices for speed
------------------------
* Parametric 3-cell box section for all bridges (span does not change thermal
  behaviour in a 2-D section model).
* 50 mm grid (coarser than the 10 mm research mesh but sufficient for cluster-
  scale relative comparisons).
* Auto stable explicit time step (~minutes per bridge).
* One representative summer week (2024-08-12 to 08-18) extracted from the
  downscaled boundary conditions.
* Multiprocessing over bridges.

Inputs
------
    ../../03_桥梁清单与实测数据/bridge_inventory_core.csv   (or 01_...)
    ../../07_GNN模型与跨桥泛化/stage1_deliverables/boundary_2024_2026.npz

Outputs
-------
    ./cluster_results/bridge_indicators.parquet
    ./cluster_results/T_timeseries/<bridge_id>.npz   (optional)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fem_box_core import FEMConfig, BoxSection, build_cartesian_mesh, compute_cos_incidence, extract_section_indicators, run_fem

# Paths
INV_CANDIDATES = [
    ROOT / ".." / "03_桥梁清单与实测数据" / "bridge_inventory_core.csv",
    ROOT / ".." / "数据包_气温降尺度" / "01_桥梁清单与标签" / "bridge_inventory_core.csv",
]
BND_CANDIDATES = [
    ROOT / ".." / "07_GNN模型与跨桥泛化" / "stage1_deliverables" / "boundary_2024_2026.npz",
]

OUT_DIR = ROOT / "cluster_results"
OUT_DIR.mkdir(exist_ok=True)


def find_first(candidates):
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of {candidates} exists.")


def load_inputs():
    inv_path = find_first(INV_CANDIDATES)
    bnd_path = find_first(BND_CANDIDATES)

    df = pd.read_csv(inv_path, encoding="utf-8-sig")
    bnd = np.load(bnd_path, allow_pickle=True)
    bnd_ids = bnd["bridge_ids"]
    t2m = bnd["t2m"]      # (n_bridges, n_hours)
    ws = bnd["ws"]
    ssrd = bnd["ssrd"]
    time_index = pd.to_datetime(bnd["time_index"])
    return df, bnd_ids, time_index, t2m, ws, ssrd


def build_weather_for_bridge(df_row, time_index, t2m, ws, ssrd, bidx, start, end):
    """Return a DataFrame with t2m_C, ws_m_s, ssrd_W_m2 for one bridge."""
    mask = (time_index >= start) & (time_index <= end)
    hours = time_index[mask]
    return pd.DataFrame({
        "datetime": hours,
        "t2m_C": t2m[bidx, mask],
        "ws_m_s": ws[bidx, mask],
        "ssrd_W_m2": ssrd[bidx, mask],
    }).set_index("datetime")


def make_section_from_row(row: pd.Series) -> BoxSection:
    """Build a plausible box section based on bridge kind / dimensions."""
    kind = str(row.get("kind", "highway")).lower()
    span = float(row.get("span_m", 600.0)) if pd.notna(row.get("span_m")) else 600.0
    n_ways = float(row.get("n_ways", 2.0)) if pd.notna(row.get("n_ways")) else 2.0

    # Scale width roughly with lane count / track count
    if kind == "railway":
        width = 13.0 if n_ways <= 1 else 14.5
        height = 3.2
    else:
        width = 10.5 + 2.5 * n_ways
        height = 2.8

    # Clamp to sensible range
    width = float(np.clip(width, 8.0, 20.0))
    height = float(np.clip(height, 2.0, 5.0))

    return BoxSection(
        width=width,
        height=height,
        top_slab=0.35,
        bottom_slab=0.30,
        web_thick=0.45,
        n_cells=3,
        overhang=1.2,
    )


def run_one_bridge(args):
    idx, row, bidx, time_index, t2m, ws, ssrd, start, end, cfg_dict = args
    bridge_id = str(row["bridge_id"])
    lat = float(row["lat"])
    bearing = float(row.get("bearing_deg", 0.0)) if pd.notna(row.get("bearing_deg")) else 0.0
    # Face normal perpendicular to bridge axis; assume one side of the girder.
    face_normal = (bearing + 90.0) % 360.0

    try:
        weather = build_weather_for_bridge(row, time_index, t2m, ws, ssrd, bidx, start, end)
        if len(weather) < 24:
            return {"bridge_id": bridge_id, "status": "too_few_hours"}

        section = make_section_from_row(row)
        cfg = FEMConfig(**cfg_dict, face_normal_deg=face_normal)
        mesh = build_cartesian_mesh(section, cfg.dx)
        cos_inc = compute_cos_incidence(mesh, weather.index, lat, face_normal, use_raytracing=False)
        T_all = run_fem(mesh, weather, weather.index, cos_inc, cfg, verbose=False)
        indicators = extract_section_indicators(T_all, mesh)

        # Save light-weight time series for top surface
        top_idx = np.where(mesh["is_top"])[0]
        if len(top_idx) == 0:
            top_idx = np.where(mesh["is_surface"] & (mesh["coords"][:, 1] > 0))[0]
        T_top_mean = np.mean(T_all[1:, top_idx], axis=1)

        ts_path = OUT_DIR / "T_timeseries" / f"{bridge_id}.npz"
        ts_path.parent.mkdir(exist_ok=True)
        np.savez(ts_path, times=weather.index.astype(str).values, T_top_mean=T_top_mean)

        return {
            "bridge_id": bridge_id,
            "status": "ok",
            "lat": lat,
            "lon": float(row["lon"]),
            "elev_m": float(row["elev_m"]),
            "region": row.get("region", ""),
            "bearing_deg": bearing,
            "face_normal_deg": face_normal,
            **indicators,
        }
    except Exception as e:
        return {"bridge_id": bridge_id, "status": f"error: {e}"}


def main():
    df, bnd_ids, time_index, t2m, ws, ssrd = load_inputs()
    id_to_bidx = {str(bid): i for i, bid in enumerate(bnd_ids)}
    df["bidx"] = df["bridge_id"].astype(str).map(id_to_bidx)
    df = df[df["bidx"] >= 0].copy()

    # Representative summer week
    start = pd.Timestamp("2024-08-12 00:00:00")
    end = pd.Timestamp("2024-08-18 23:00:00")

    cfg_dict = dict(
        rho=2400.0, cp=920.0, k=1.80,
        alpha_solar=0.45, emissivity=0.90,
        h_forced=15.0, h_natural=10.0, h_hole=5.0,
        ws_threshold=1.0,
        dx=0.05, dt=None, longwave_mode="none",
    )

    payloads = []
    for _, row in df.iterrows():
        payloads.append((_, row, int(row["bidx"]), time_index, t2m, ws, ssrd, start, end, cfg_dict))

    n_workers = min(6, cpu_count() or 1)
    print(f"Computing {len(payloads)} bridges with {n_workers} workers...")
    t0 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(run_one_bridge, payloads)
    print(f"Finished in {time.time()-t0:.1f}s")

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_parquet(OUT_DIR / "bridge_indicators.parquet", index=False)
    results_df.to_csv(OUT_DIR / "bridge_indicators.csv", index=False, encoding="utf-8-sig")

    # Metadata
    meta = {
        "created": datetime.now().isoformat(),
        "period": {"start": str(start), "end": str(end)},
        "config": cfg_dict,
        "n_bridges": len(results_df),
        "n_success": int((results_df["status"] == "ok").sum()),
    }
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {OUT_DIR}")
    print(results_df["status"].value_counts())


if __name__ == "__main__":
    main()
