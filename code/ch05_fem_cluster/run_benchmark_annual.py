# -*- coding: utf-8 -*-
"""run_benchmark_annual.py — Reproduce the annual 2-D transient CEBM temperature
field of the benchmark cross-section (manuscript Section 5.1).

Benchmark case: M04, a 340 m deck-type RC arch bridge (single box with three
cells, 1200 mm x 600 mm mid-span arch-ring cross-section). The 10 mm Cartesian
embedded mesh (3526 nodes) is loaded from ``data/benchmark_mesh.npz``; hourly
boundary conditions (ERA5-Land/BNUSSR-derived, bridge-site corrected) from
``data/boundary_2024_benchmark.csv``; in-hole air temperature from the N212
channel of ``data/measured_temperatures_2024.csv``.

Solver: explicit FVM, DT = 30 s (vectorised, numerically equivalent to the
original loop implementation — see fsa_fast.py). Configuration below reproduces
the calibrated parameter set of manuscript Table 5:
  K=1.80, rho=2400, cp=920, alpha_solar=0.45, h_forced=15, h_natural=10,
  h_hole=5, ws_scale=0.39, explicit longwave SVF top/side/bottom = 0.8/0.5/0,
  T_sky = T_a - 5*(1 - ssrd/1000), September t2m correction +2.5 degC.

Outputs (default: repository ``data/`` folder):
  fem_benchmark_annual_full.npz  T (8777, 3526) float32 + times + coords
                                 (~1e5 s on a modern CPU; needs ~2 GB RAM)
  The committed ``fem_benchmark_annual.npz`` is the down-sampled derivative
  produced by ``make_slim_derivative()`` below (full field for 2024-08-12 +
  annual series at the 15 sensor nodes and 7 representative nodes).

Usage:
  python run_benchmark_annual.py            # full annual re-run
  python run_benchmark_annual.py --tables   # full re-run + validation tables
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")
TABLES = os.path.join(REPO_ROOT, "tables")
sys.path.insert(0, HERE)

import fsa_fast as ff  # noqa: E402

# ----------------------------------------------------------------- parameters
LAT = 31.06                 # bridge latitude (deg N)
FACE_NORMAL_DEG = 140.6     # section outward normal azimuth (deg)
SHIFT = np.array([600.0, 300.0])   # drawing-centred coords -> FEM coords (mm)

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2024-12-31")
CLEAR0 = pd.Timestamp("2024-08-12")
CLEAR1 = pd.Timestamp("2024-08-17 23:00")
VAL0 = pd.Timestamp("2024-02-26")          # long validation window
VAL1 = pd.Timestamp("2024-12-31 23:00")
SEP_T2M_FIX = 2.5                          # September seasonal correction (degC)

CONFIG = dict(K=1.80, RHO=2400.0, CP=920.0, ALPHA_SOLAR=0.45,
              H_FORCED=15.0, H_NATURAL=10.0, H_HOLE=5.0, ws_scale=0.39,
              longwave_mode='explicit', longwave_svf_top=0.8, lw_sky_drop=5.0)

# Recorded sensor positions (section-centred, mm) — drawing archive coordinates
SENSOR_MM = {
    'N101': (-350, 290), 'N102': (-350, 270), 'N103': (-350, 250),
    'N107': (350, 290), 'N108': (350, 270), 'N109': (350, 250),
    'N204': (-350, -250), 'N206': (0, -250),
    'N205': (-160, -100), 'N210': (350, -260), 'N211': (160, -100),
    'N212': (-350, -100),
    'N207': (565, 0), 'N208': (505, 0), 'N209': (445, 0),
}
# Calibrated positions for the 4 validation channels (inverse search on daily
# max/min MAE, see manuscript Section 5.1). N209's archived coordinate was
# found to be a recording error; it is a sunlit top-slab point.
SENSOR_FIX = {'N101': (-220, 220), 'N109': (200, 190),
              'N102': (-220, 200), 'N209': (210, 190)}
PICK4 = ['N101', 'N109', 'N102', 'N209']
STRUCT14 = ['N101', 'N102', 'N103', 'N107', 'N108', 'N109',
            'N204', 'N205', 'N206', 'N207', 'N208', 'N209', 'N210', 'N211']


# ------------------------------------------------------------------ data I/O
def load_mesh():
    d = np.load(os.path.join(DATA, "benchmark_mesh.npz"), allow_pickle=True)
    coords = d["valid_coords"]
    is_surface = d["is_surface"]
    is_hole_edge = d["is_hole_edge"]
    mesh = dict(coords=coords, edge_index=d["edge_index"], N=coords.shape[0],
                is_top=d["is_top"], is_bot=d["is_bot"],
                is_right=d["is_right"], is_left=d["is_left"],
                is_hole_edge=is_hole_edge, is_surface=is_surface,
                is_outer_surf=is_surface & (~is_hole_edge),
                is_interior=~is_surface, surf_normal=d["surf_normal"])
    return mesh


def load_bc_and_obs():
    bc = pd.read_csv(os.path.join(DATA, "boundary_2024_benchmark.csv"))
    bc["datetime"] = pd.to_datetime(bc["datetime"])
    bc = bc.set_index("datetime").sort_index()
    bc.index = bc.index + pd.Timedelta(hours=8)   # UTC -> Beijing time
    ht = pd.read_csv(os.path.join(DATA, "measured_temperatures_2024.csv"))
    ht["DateTime"] = pd.to_datetime(ht["DateTime"])
    ht = ht.set_index("DateTime").sort_index()
    return bc, ht


def sensor_node_map(mesh):
    coords = mesh["coords"]
    mapping = {}
    for name, (sx, sy) in SENSOR_MM.items():
        target = np.array([sx, sy]) + SHIFT
        d = np.sqrt((coords[:, 0] - target[0]) ** 2 + (coords[:, 1] - target[1]) ** 2)
        mapping[name] = int(d.argmin())
    for name, (nx, ny) in SENSOR_FIX.items():     # calibrated channels win
        target = np.array([nx, ny]) + SHIFT
        d = np.sqrt((coords[:, 0] - target[0]) ** 2 + (coords[:, 1] - target[1]) ** 2)
        mapping[name] = int(d.argmin())
    return mapping


# ------------------------------------------------------------------ main run
def main(make_tables=False):
    t0 = time.time()
    print("[1/4] loading mesh / boundary / measurements ...", flush=True)
    mesh = load_mesh()
    bc, ht = load_bc_and_obs()
    hole_air = ht["N212"]
    geo_path = os.path.join(DATA, "dwg_geometry.json")
    geo = __import__("json").load(open(geo_path))
    holes = [geo["hole_left"], geo["hole_center"], geo["hole_right"]]
    sm = sensor_node_map(mesh)

    times = pd.date_range(START, END + pd.Timedelta(hours=23), freq='h')
    times = pd.DatetimeIndex([t for t in times if t in bc.index])
    bc_sub = bc.loc[times].copy()
    bc_sub.loc[bc_sub.index.month == 9, "t2m_C"] += SEP_T2M_FIX
    print("  hours=%d (%s ~ %s)" % (len(times), times[0], times[-1]), flush=True)

    print("[2/4] computing cos_inc ...", flush=True)
    ci = ff.compute_cos_inc_fast(mesh, times, geo["outer"], holes,
                                 use_raytracing=False)

    print("[3/4] annual FEM solve (%.0f h x 120 steps/h) ..."
          % len(times), flush=True)
    T = ff.run_fem_fast(mesh, bc_sub, times, ci, hole_air, **CONFIG)
    out_full = os.path.join(DATA, "fem_benchmark_annual_full.npz")
    np.savez_compressed(out_full, T=T.astype(np.float32),
                        times=np.array([t.isoformat() for t in times]),
                        coords=mesh["coords"])
    print("  saved %s (%.1f s)" % (out_full, time.time() - t0), flush=True)

    print("[4/4] building slim derivative ...", flush=True)
    make_slim_derivative(T, times, mesh, sm)
    if make_tables:
        write_validation_tables(T, times, mesh, sm, ht)
    print("done in %.1f s" % (time.time() - t0), flush=True)


def make_slim_derivative(T, times, mesh, sm):
    """Down-sampled derivative committed to the repo: full field for the
    representative clear day (2024-08-12) + annual hourly series at the
    15 sensor nodes and 7 representative nodes."""
    coords = mesh["coords"]
    is_int = mesh["is_interior"]
    ft = pd.DatetimeIndex(times)
    hs = np.where(mesh["is_hole_edge"])[0]

    def nearest(x, y, mask=None):
        d = np.sqrt((coords[:, 0] - x) ** 2 + (coords[:, 1] - y) ** 2)
        if mask is not None:
            d[~mask] = 1e18
        return int(d.argmin())

    locs = [('top', nearest(600, 600)), ('bottom', nearest(600, 0)),
            ('left', nearest(0, 300)), ('right', nearest(1200, 300)),
            ('center_int', nearest(600, 300, is_int)),
            ('mid_int', nearest(600, 150, is_int)),
            ('hole_edge', int(hs[len(hs) // 4]))]
    names = list(sm.keys()) + [n for n, _ in locs]
    idxs = [sm[k] for k in sm] + [i for _, i in locs]
    T_year_nodes = T[1:][:, idxs].astype(np.float32)

    mday = ft.date == pd.Timestamp("2024-08-12").date()
    T_day = T[1:][mday].astype(np.float32)

    out = os.path.join(DATA, "fem_benchmark_annual.npz")
    np.savez_compressed(out, T_day=T_day, T_year_nodes=T_year_nodes,
                        node_names=np.array(names), node_indices=np.array(idxs),
                        times=np.array([t.isoformat() for t in times]),
                        coords=coords, day="2024-08-12",
                        sensor_map={k: int(v) for k, v in sm.items()})
    print("  saved %s" % out, flush=True)


def write_validation_tables(T, times, mesh, sm, ht):
    """Recompute the manuscript validation tables (4-ch pooled / 14-ch)."""
    ft = pd.DatetimeIndex(times)
    Tp = T[1:]

    def stats(nm, t0w, t1w):
        i = sm[nm]
        fv = Tp[:, i]
        obs = ht[nm].reindex(ft).values
        m = np.isfinite(obs) & np.isfinite(fv) & (ft >= t0w) & (ft <= t1w)
        if m.sum() < 10:
            return None
        f, o = fv[m], obs[m]
        return dict(channel=nm,
                    RMSE=round(float(np.sqrt(np.mean((f - o) ** 2))), 3),
                    MAE=round(float(np.mean(np.abs(f - o))), 3),
                    r=round(float(np.corrcoef(f, o)[0, 1]), 3), N=int(m.sum()))

    def pool(chs, t0w, t1w):
        fs, os_ = [], []
        for nm in chs:
            i = sm[nm]
            fv = Tp[:, i]
            obs = ht[nm].reindex(ft).values
            m = np.isfinite(obs) & np.isfinite(fv) & (ft >= t0w) & (ft <= t1w)
            fs.append(fv[m]); os_.append(obs[m])
        f = np.concatenate(fs); o = np.concatenate(os_)
        return dict(channel='pooled',
                    RMSE=round(float(np.sqrt(np.mean((f - o) ** 2))), 3),
                    MAE=round(float(np.mean(np.abs(f - o))), 3),
                    r=round(float(np.corrcoef(f, o)[0, 1]), 3), N=int(len(f)))

    for tag, t0w, t1w in [("sunny6day", CLEAR0, CLEAR1),
                          ("longwindow", VAL0, VAL1)]:
        rows = [r for r in (stats(nm, t0w, t1w) for nm in STRUCT14) if r]
        rows.append(pool(STRUCT14, t0w, t1w))
        df = pd.DataFrame(rows)
        out = os.path.join(TABLES, "ch05_sensor_validation_%s.csv" % tag)
        df.to_csv(out, index=False)
        print("  wrote %s" % out, flush=True)


if __name__ == "__main__":
    main(make_tables=("--tables" in sys.argv))
