#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified 2-D transient thermal FEM for box-girder bridge sections.

Features
--------
* Parametric single-cell / multi-cell box sections (shapely-based)
* Cartesian grid with embedded boundary cut-cells (volume-fraction aware)
* Solar geometry + surface self-shadowing ray tracing
* Equivalent-temperature boundary condition
* Explicit FVM solver with stability-guarded time step
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

SIGMA = 5.67e-8  # Stefan-Boltzmann constant [W/(m^2 K^4)]


@dataclass
class FEMConfig:
    """Physical and numerical configuration for a box-girder run."""

    # material
    rho: float = 2400.0          # kg/m^3
    cp: float = 920.0            # J/(kg K)
    k: float = 1.80              # W/(m K)
    alpha_solar: float = 0.45    # solar absorptivity (calibrated)
    emissivity: float = 0.90     # long-wave emissivity

    # convection
    h_forced: float = 15.0       # W/(m^2 K), ws > ws_threshold
    h_natural: float = 10.0      # W/(m^2 K), ws <= ws_threshold
    h_hole: float = 5.0          # W/(m^2 K), internal cavity surfaces
    ws_threshold: float = 1.0    # m/s
    ws_scale: float = 1.0        # empirical wind-speed scaling factor

    # numerics
    dx: float = 0.025            # m (25 mm default, coarser than research FEM)
    dt: Optional[float] = None   # s, auto if None
    longwave_mode: str = "none"  # 'none' | 'simple'

    # geometry
    bearing_deg: float = 0.0     # bridge axis azimuth from north, clockwise
    face_normal_deg: float = 0.0 # outward normal of the analysed cross-section

    def auto_dt(self) -> float:
        """Return a stable explicit time step."""
        alpha = self.k / (self.rho * self.cp)
        # Fourier-like limit for 2-D conduction
        dt_cond = 0.20 * self.dx * self.dx / alpha
        # Surface node limit
        h_max = max(self.h_forced, self.h_natural, self.h_hole)
        dt_surf = self.rho * self.cp * self.dx / (2.0 * h_max)
        return min(dt_cond, dt_surf)


class BoxSection:
    """Parametric concrete box-girder cross section."""

    def __init__(
        self,
        width: float = 12.0,           # total deck width [m]
        height: float = 3.0,           # total depth [m]
        top_slab: float = 0.35,        # top slab thickness [m]
        bottom_slab: float = 0.30,     # bottom slab thickness [m]
        web_thick: float = 0.45,       # web thickness [m]
        n_cells: int = 3,              # number of cells
        overhang: float = 1.5,           # cantilever overhang [m]
    ):
        self.width = width
        self.height = height
        self.top_slab = top_slab
        self.bottom_slab = bottom_slab
        self.web_thick = web_thick
        self.n_cells = n_cells
        self.overhang = overhang
        self.polygon = self._build_polygon()

    def _build_polygon(self) -> Polygon:
        """Build shapely polygon of concrete area."""
        w, h = self.width, self.height
        ts, bs = self.top_slab, self.bottom_slab
        wt = self.web_thick
        oh = self.overhang
        # Outer contour (centred at origin, bottom at y=-h/2)
        outer = [
            (-w / 2.0, h / 2.0),       # top left corner
            (w / 2.0, h / 2.0),        # top right corner
            (w / 2.0, h / 2.0 - ts),   # right top slab bottom
            (w / 2.0 - oh, h / 2.0 - ts),  # right overhang end
            (w / 2.0 - oh, -h / 2.0 + bs),  # right bottom corner
            (-w / 2.0 + oh, -h / 2.0 + bs),  # left bottom corner
            (-w / 2.0 + oh, h / 2.0 - ts),   # left overhang end
            (-w / 2.0, h / 2.0 - ts),  # left top slab bottom
            (-w / 2.0, h / 2.0),       # close
        ]
        poly = Polygon(outer)
        if self.n_cells >= 1:
            # Remove rectangular cavities
            cavities = []
            usable_width = w - 2.0 * oh - 2.0 * wt
            if self.n_cells == 1:
                cell_w = usable_width
                xs = [-usable_width / 2.0]
            else:
                cell_w = (usable_width - wt * (self.n_cells - 1)) / self.n_cells
                xs = [-usable_width / 2.0 + i * (cell_w + wt) for i in range(self.n_cells)]
            for x0 in xs:
                cavity = Polygon([
                    (x0, h / 2.0 - ts),
                    (x0 + cell_w, h / 2.0 - ts),
                    (x0 + cell_w, -h / 2.0 + bs),
                    (x0, -h / 2.0 + bs),
                    (x0, h / 2.0 - ts),
                ])
                cavities.append(cavity)
            if cavities:
                poly = poly.difference(unary_union(cavities))
        return poly

    def bounds(self) -> Tuple[float, float, float, float]:
        return self.polygon.bounds


def _section_polygon(section) -> Union[Polygon, MultiPolygon]:
    """Accept BoxSection or arbitrary shapely geometry."""
    if hasattr(section, "polygon"):
        return section.polygon
    return section


def load_dwg_geometry(json_path: str, shift: Optional[np.ndarray] = None) -> Polygon:
    """Load a real bridge section from the JSON format used by the research FEM.

    The dwg_geometry.json files store coordinates in millimetres relative to the
    section centroid.  This function rescales to metres and optionally shifts.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        geo = json.load(f)
    outer = np.array(geo["outer"], dtype=float)
    hole_keys = [k for k in ("hole_left", "hole_center", "hole_right", "hole_1", "hole_2", "hole_3") if k in geo]
    holes = [np.array(geo[k], dtype=float) for k in hole_keys]

    # Drawings are in millimetres -> rescale to metres when needed.
    if np.max(np.abs(outer)) > 10.0:
        outer = outer / 1000.0
        holes = [h / 1000.0 for h in holes]
    if shift is not None:
        s = np.asarray(shift, dtype=float) / 1000.0
        outer = outer + s
        holes = [h + s for h in holes]

    polygon = Polygon(outer)
    for h in holes:
        hole_poly = Polygon(h)
        if hole_poly.is_valid and hole_poly.area > 1e-6:
            polygon = polygon.difference(hole_poly)
    return polygon


def build_cartesian_mesh(section, dx: float, padding: float = 0.0):
    """
    Build a Cartesian grid embedded in the section polygon.

    Parameters
    ----------
    section : BoxSection or shapely Polygon/MultiPolygon
    dx : grid spacing in m
    padding : extra bounding box padding in m

    Returns
    -------
    dict with:
        coords: (N,2) node coords in m
        cell_volume: (N,) approximate concrete volume fraction [0-1]
        is_concrete: (N,) bool inside concrete
        is_surface: (N,) bool boundary nodes
        is_hole: (N,) bool internal cavity surface nodes
        is_top: (N,) bool
        is_bot: (N,) bool
        is_left: (N,) bool
        is_right: (N,) bool
        surf_normal: (N,2) outward normal (only meaningful for surface nodes)
        neighbors: list of neighbor index lists
        nx, ny: grid dimensions
        section_polygon: shapely geometry used for meshing (for ray tracing)
    """
    polygon = _section_polygon(section)
    minx, miny, maxx, maxy = polygon.bounds
    minx -= padding; miny -= padding; maxx += padding; maxy += padding
    nx = int(np.ceil((maxx - minx) / dx)) + 1
    ny = int(np.ceil((maxy - miny) / dx)) + 1
    x = np.linspace(minx, minx + (nx - 1) * dx, nx)
    y = np.linspace(miny, miny + (ny - 1) * dx, ny)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    N = pts.shape[0]

    # Cell volumes via sub-sampling
    n_sub = 4
    sub_dx = dx / n_sub
    sub_offsets = np.linspace(-dx / 2.0 + sub_dx / 2.0, dx / 2.0 - sub_dx / 2.0, n_sub)
    volume = np.zeros(N)
    for ox in sub_offsets:
        for oy in sub_offsets:
            sample = pts + np.array([[ox, oy]])
            inside = np.array([polygon.contains(Point(p)) for p in sample])
            volume += inside.astype(float)
    volume /= n_sub * n_sub

    is_concrete = volume > 0.05
    concrete_idx = np.where(is_concrete)[0]

    # Build neighbor graph for concrete nodes only
    idx_map = -np.ones(N, dtype=np.int32)
    idx_map[concrete_idx] = np.arange(len(concrete_idx))
    coords = pts[concrete_idx]
    Nc = len(concrete_idx)

    def ij(idx):
        return idx // nx, idx % nx

    neighbors = [[] for _ in range(Nc)]
    for local_i, global_i in enumerate(concrete_idx):
        i, j = ij(global_i)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nx:
                global_j = ni * nx + nj
                if idx_map[global_j] >= 0:
                    neighbors[local_i].append(int(idx_map[global_j]))

    # Surface detection: concrete nodes with at least one non-concrete neighbor
    is_surface = np.zeros(Nc, dtype=bool)
    is_hole = np.zeros(Nc, dtype=bool)
    for i in range(Nc):
        if any(volume[concrete_idx[n]] < 0.95 for n in neighbors[i] if n != i):
            is_surface[i] = True

    # Estimate outward normal for each surface node from empty-neighbour directions
    surf_normal = np.zeros((Nc, 2), dtype=np.float64)
    is_top = np.zeros(Nc, dtype=bool)
    is_bot = np.zeros(Nc, dtype=bool)
    is_left = np.zeros(Nc, dtype=bool)
    is_right = np.zeros(Nc, dtype=bool)

    # Convert global node index -> (i,j) grid position for empty-neighbour scan
    def global_ij(gidx):
        return gidx // nx, gidx % nx

    for local_i in range(Nc):
        if not is_surface[local_i]:
            continue
        gidx = concrete_idx[local_i]
        ci, cj = global_ij(gidx)
        px, py = coords[local_i]
        dirs = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = ci + di, cj + dj
            empty = True
            if 0 <= ni < ny and 0 <= nj < nx:
                ngidx = ni * nx + nj
                if idx_map[ngidx] >= 0:
                    empty = False
            if empty:
                dirs.append(np.array([dj, di], dtype=float))  # x,y direction
        if dirs:
            normal = np.mean(dirs, axis=0)
            nrm = np.linalg.norm(normal)
            if nrm > 1e-9:
                normal /= nrm
            surf_normal[local_i] = normal
        else:
            # Fallback: use polygon boundary outward direction
            p = Point(px, py)
            boundary = unary_union(polygon).boundary
            nearest = boundary.interpolate(boundary.project(p))
            if isinstance(nearest, Point):
                d = np.array([nearest.x - px, nearest.y - py])
                nrm = np.linalg.norm(d)
                if nrm > 1e-9:
                    surf_normal[local_i] = d / nrm

        nx_, ny_ = surf_normal[local_i]
        # Use geometry-based thresholds that tolerate coarse meshes
        y_top_thr = maxy - 0.35
        y_bot_thr = miny + 0.35
        if ny_ > 0.5 and py >= y_top_thr:
            is_top[local_i] = True
        elif ny_ < -0.5 and py <= y_bot_thr:
            is_bot[local_i] = True
        elif nx_ < -0.6:
            is_left[local_i] = True
        elif nx_ > 0.6:
            is_right[local_i] = True

        # Distinguish outer surface from internal cavity surface
        p_out = Point(px + nx_ * dx * 0.6, py + ny_ * dx * 0.6)
        p_in = Point(px - nx_ * dx * 0.6, py - ny_ * dx * 0.6)
        out_inside = polygon.contains(p_out)
        in_inside = polygon.contains(p_in)
        if in_inside and out_inside:
            # Both sides inside concrete -> cavity surface
            is_hole[local_i] = True
        elif not in_inside and not out_inside:
            # Likely a thin feature; treat as outer surface
            is_hole[local_i] = False
        else:
            is_hole[local_i] = False  # out_inside and not in_inside -> outer surface

    return dict(
        coords=coords,
        cell_volume=volume[concrete_idx],
        is_concrete=is_concrete,
        is_surface=is_surface,
        is_hole=is_hole,
        is_top=is_top,
        is_bot=is_bot,
        is_left=is_left,
        is_right=is_right,
        surf_normal=surf_normal,
        neighbors=neighbors,
        nx=nx, ny=ny, dx=dx,
        section_polygon=polygon,
    )


def solar_geometry(dt: pd.Timestamp, lat: float) -> Tuple[float, float]:
    """Return solar elevation [rad] and azimuth [rad, 0=south, +westward]."""
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    doy = dt.dayofyear
    lat_r = np.radians(lat)
    decl = np.radians(23.45) * np.cos(2 * np.pi * (doy - 172) / 365.25)
    ha = np.radians((hour - 12) * 15.0)
    sin_elev = np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha)
    elev = np.arcsin(np.clip(sin_elev, -1.0, 1.0))
    if elev > 0:
        cos_az = np.clip(
            (np.sin(decl) - np.sin(elev) * np.sin(lat_r)) / (np.cos(elev) * np.cos(lat_r)),
            -1.0, 1.0,
        )
        azim = np.arccos(cos_az)
        if hour < 12:
            azim = -azim
    else:
        azim = 0.0
    return elev, azim


def sun_direction_2d(elev: float, azim: float, face_normal_deg: float) -> np.ndarray:
    """
    Incoming solar ray projected onto the cross-section plane.
    face_normal_deg: outward normal azimuth clockwise from geographic north.
    """
    if elev <= 0:
        return np.zeros(2)
    face_south = np.radians(face_normal_deg - 180.0)
    diff = azim - face_south
    sx = np.cos(elev) * np.cos(diff)
    sy = np.sin(elev)
    v = np.array([sx, sy])
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def compute_cos_incidence(
    mesh: dict,
    times: pd.DatetimeIndex,
    lat: float,
    face_normal_deg: float,
    use_raytracing: bool = True,
    section_polygon=None,
) -> np.ndarray:
    """Compute |n . s| incidence factor for each surface node and hour."""
    boundary = None
    if use_raytracing:
        if section_polygon is not None:
            boundary = unary_union(section_polygon).boundary
        elif "section_polygon" in mesh:
            boundary = unary_union(mesh["section_polygon"]).boundary
    NT, N = len(times), mesh["coords"].shape[0]
    cos_inc = np.zeros((NT, N), dtype=np.float32)
    surf = mesh["is_surface"]
    normals = mesh["surf_normal"]
    coords = mesh["coords"]

    for ti, t in enumerate(times):
        elev, azim = solar_geometry(t, lat)
        if elev <= 0:
            continue
        sx, sy = -sun_direction_2d(elev, azim, face_normal_deg)
        for ni in np.where(surf)[0]:
            nx_, ny_ = normals[ni]
            ci = nx_ * sx + ny_ * sy
            if ci >= 0:
                continue
            if use_raytracing and boundary is not None:
                px, py = coords[ni]
                ray = LineString([
                    (px + nx_ * 0.02, py + ny_ * 0.02),
                    (px + nx_ * 0.02 - sx * 200.0, py + ny_ * 0.02 - sy * 200.0),
                ])
                isect = ray.intersection(boundary)
                if not isect.is_empty:
                    hits = list(isect.geoms) if hasattr(isect, "geoms") else [isect]
                    if any(
                        hasattr(h, "distance") and h.distance(Point(px, py)) > 0.05
                        for h in hits
                    ):
                        continue
            cos_inc[ti, ni] = -ci
    return cos_inc


def run_fem(
    mesh: dict,
    bc: pd.DataFrame,
    times: pd.DatetimeIndex,
    cos_inc: np.ndarray,
    config: FEMConfig,
    hole_air: Optional[pd.Series] = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Explicit FVM solver.

    Parameters
    ----------
    mesh : dict from build_cartesian_mesh
    bc : DataFrame with columns t2m_C, ws_m_s, ssrd_W_m2 aligned with `times`
    times : hourly DatetimeIndex
    cos_inc : (NT, N) incidence factor
    config : FEMConfig
    hole_air : pd.Series indexed by datetime, optional internal cavity air temperature

    Returns
    -------
    T_all : (NT+1, N) hourly temperature field (T_all[0] = initial)
    """
    N = mesh["coords"].shape[0]
    coords = mesh["coords"]
    neighbors = mesh["neighbors"]
    dx = config.dx
    dt = config.dt or config.auto_dt()

    alpha = config.k / (config.rho * config.cp)
    kdx = config.k / dx
    dx2 = dx * dx

    interior = np.where(~mesh["is_surface"])[0]
    surface = np.where(mesh["is_surface"])[0]
    hole = np.where(mesh["is_hole"])[0]
    outer = np.setdiff1d(surface, hole)

    # Precompute neighbor arrays for vectorised interior update
    max_n = max(len(neighbors[i]) for i in interior) if len(interior) else 1
    nbr_idx = -np.ones((len(interior), max_n), dtype=np.int32)
    nbr_w = np.zeros((len(interior), max_n), dtype=np.float64)
    for ki, i in enumerate(interior):
        nbs = neighbors[i]
        nn = len(nbs)
        for kj, j in enumerate(nbs):
            nbr_idx[ki, kj] = j
            # distance weighting (works for Cartesian + cut cells)
            d2 = np.sum((coords[i] - coords[j]) ** 2) + 1e-12
            nbr_w[ki, kj] = 1.0 / d2
        nbr_w[ki, :nn] /= nbr_w[ki, :nn].sum()

    def surf_int_neighbors(i):
        nbs = neighbors[i]
        interior_nbs = [j for j in nbs if not mesh["is_surface"][j]]
        if not interior_nbs:
            interior_nbs = nbs
        return interior_nbs

    max_os = max((len(surf_int_neighbors(i)) for i in outer), default=1)
    max_hole = max((len(surf_int_neighbors(i)) for i in hole), default=1)
    os_idx = -np.ones((len(outer), max_os), dtype=np.int32)
    os_w = np.zeros((len(outer), max_os), dtype=np.float64)
    for ki, i in enumerate(outer):
        nbs = surf_int_neighbors(i)
        nn = len(nbs)
        os_w[ki, :nn] = 1.0 / nn
        for kj, j in enumerate(nbs):
            os_idx[ki, kj] = j
    hole_idx_arr = -np.ones((len(hole), max_hole), dtype=np.int32)
    hole_w = np.zeros((len(hole), max_hole), dtype=np.float64)
    for ki, i in enumerate(hole):
        nbs = surf_int_neighbors(i)
        nn = len(nbs)
        hole_w[ki, :nn] = 1.0 / nn
        for kj, j in enumerate(nbs):
            hole_idx_arr[ki, kj] = j

    # Time stepping
    steps_per_hour = int(np.ceil(3600.0 / dt))
    dt_actual = 3600.0 / steps_per_hour
    n_total = len(times) * steps_per_hour
    output_every = steps_per_hour

    T_init = float(np.mean(bc["t2m_C"].values[:24]))
    T = np.full(N, T_init, dtype=np.float64)
    T_all = np.zeros((len(times) + 1, N), dtype=np.float64)
    T_all[0] = T.copy()

    t2m_arr = bc["t2m_C"].values.astype(np.float64)
    ws_arr = bc["ws_m_s"].values.astype(np.float64) * config.ws_scale
    ssrd_arr = bc["ssrd_W_m2"].values.astype(np.float64)

    do_lw = config.longwave_mode not in (None, "none", False)
    svf = np.zeros(len(outer))
    if do_lw:
        for ki, i in enumerate(outer):
            if mesh["is_top"][i]:
                svf[ki] = 1.0
            elif mesh["is_bot"][i]:
                svf[ki] = 0.0
            else:
                svf[ki] = 0.5

    for step in range(1, n_total + 1):
        hi = (step - 1) // steps_per_hour
        if hi >= len(times):
            hi = len(times) - 1
        t2m = t2m_arr[hi]
        ws = ws_arr[hi]
        ssrd = ssrd_arr[hi]
        h_outer = config.h_forced if ws > config.ws_threshold else config.h_natural
        ci_row = cos_inc[hi]

        dT = np.zeros(N)

        # Interior conduction
        for ki, i in enumerate(interior):
            Ti = T[i]
            lap = 0.0
            for kj in range(max_n):
                j = nbr_idx[ki, kj]
                if j < 0:
                    break
                lap += (T[j] - Ti) * nbr_w[ki, kj]
            dT[i] = alpha * lap * (4.0 / dx2) * dt_actual

        coeff = 1.0 / (kdx + h_outer)
        for ki, i in enumerate(outer):
            T_int = np.mean(T[os_idx[ki][os_idx[ki] >= 0]]) if os_idx[ki][os_idx[ki] >= 0].size else T[i]
            q_sol = config.alpha_solar * ssrd * ci_row[i]
            q_lw = 0.0
            if do_lw and svf[ki] > 0.0:
                T_sky_C = t2m - 10.0 * (1.0 - np.clip(ssrd / 1000.0, 0.0, 1.0))
                q_lw = (
                    config.emissivity
                    * SIGMA
                    * svf[ki]
                    * ((T_sky_C + 273.15) ** 4 - (T[i] + 273.15) ** 4)
                )
            Teq = (kdx * T_int + h_outer * t2m + q_sol + q_lw) * coeff
            tau = config.rho * config.cp * dx * 0.5 / (h_outer + kdx)
            dT[i] = (Teq - T[i]) / tau * dt_actual

        # Internal cavity air temperature (e.g. measured by embedded sensor N212)
        hole_val = t2m
        if hole_air is not None:
            tnow = times[hi] if hi < len(times) else times[-1]
            if tnow in hole_air.index:
                hv = hole_air.loc[tnow]
                if pd.notna(hv):
                    hole_val = float(hv)

        for ki, i in enumerate(hole):
            T_int = np.mean(T[hole_idx_arr[ki][hole_idx_arr[ki] >= 0]]) if hole_idx_arr[ki][hole_idx_arr[ki] >= 0].size else T[i]
            Teq = (kdx * T_int + config.h_hole * hole_val) / (kdx + config.h_hole)
            tau = config.rho * config.cp * dx * 0.5 / (config.h_hole + kdx)
            dT[i] = (Teq - T[i]) / tau * dt_actual

        T += dT
        if not np.isfinite(T).all():
            raise RuntimeError(f"NaN/Inf detected at step {step}")
        if step % output_every == 0:
            T_all[step // output_every] = T.copy()

    if verbose:
        print(f"FEM finished: dt={dt_actual:.1f}s, steps={n_total}, nodes={N}")
    return T_all


def extract_section_indicators(T_all: np.ndarray, mesh: dict) -> dict:
    """Extract cluster-relevant temperature indicators from a FEM run."""
    coords = mesh["coords"]
    y = coords[:, 1]
    y_max, y_min = y.max(), y.min()
    band = 0.15  # m, thickness band used to define top/bottom surfaces on coarse grids

    # Top / bottom surface nodes: nodes within band of the respective extreme
    top_idx = np.where(mesh["is_surface"] & (y >= y_max - band))[0]
    bot_idx = np.where(mesh["is_surface"] & (y <= y_min + band))[0]

    T_top_mean = np.mean(T_all[:, top_idx], axis=1) if len(top_idx) else np.full(T_all.shape[0], np.nan)
    T_bot_mean = np.mean(T_all[:, bot_idx], axis=1) if len(bot_idx) else np.full(T_all.shape[0], np.nan)
    dT = T_top_mean - T_bot_mean

    # Peak equivalent surface temperature (solar + air) for reference
    # Extracted as the hottest single surface node at each hour
    surf_idx = np.where(mesh["is_surface"])[0]
    T_surf_peak_hourly = np.max(T_all[:, surf_idx], axis=1)

    return {
        "T_peak_max": float(np.max(T_all)),
        "T_surf_peak": float(np.max(T_surf_peak_hourly)),
        "T_top_peak": float(np.nanmax(T_top_mean)),
        "T_bot_peak": float(np.nanmax(T_bot_mean)),
        "T_top_mean": float(np.nanmean(T_top_mean)),
        "T_bot_mean": float(np.nanmean(T_bot_mean)),
        "max_vertical_gradient": float(np.nanmax(dT)),
        "mean_vertical_gradient": float(np.nanmean(dT)),
        "diurnal_range_top": float(np.nanmax(T_top_mean) - np.nanmin(T_top_mean)),
    }


if __name__ == "__main__":
    # Quick self-test
    sec = BoxSection(width=10.0, height=2.8, top_slab=0.35, bottom_slab=0.30,
                     web_thick=0.45, n_cells=3, overhang=1.2)
    cfg = FEMConfig(dx=0.05, dt=None)
    mesh = build_cartesian_mesh(sec, cfg.dx)
    print("Nodes:", mesh["coords"].shape[0], "Surface:", mesh["is_surface"].sum())
