# -*- coding: utf-8 -*-
"""
fsa_fast.py — SCI 初稿完整算法（fem_sensitivity_analysis.py 的 run_fem + compute_cos_inc）
的**严格数值等价**向量化实现（只改实现方式，不改任何公式/参数/边界条件）。

原版每步用 Python 双层循环遍历 2784 内部节点 + 360 外表节点 + 382 孔边节点；
本版把邻居求和改为 (N, max_nbs) 填充数组上的向量化运算，逐元素公式完全一致。

已保留的 SCI 初稿算法要素：
  · 10mm 规则网格（phase1_data.npz, 3526 节点）
  · DT=30s 显式积分（Courant=0.2446）
  · 逐节点 cos_inc（外法向 · 太阳方向），可选自遮挡射线追踪（shapely 边界，同 _node_is_lit 语义）
  · 表面：FVM 半格热容层 tau=(ρcp·DX/2)/(h+k/DX)，Teq=(kdx·T_int + h·T_a + q_sol + q_lw)/(kdx+h)
  · 孔边：对流（孔内气温用 N212 实测，缺失回退 T_a）
  · 长波：longwave_mode='explicit' → εσ·SVF·(T_sky⁴−T_s⁴)，SVF 顶1/侧0.5/底0，T_sky=T_a−10(1−I/1000)
  · 全参数化：K/RHO/CP/ALPHA_SOLAR/H_FORCED/H_NATURAL/H_HOLE/ws_scale
  · 可选桥面系遮荫（deck_shading，SCI 初稿无此项，默认关闭）
"""
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, LineString, Point

LAT = 31.06
FACE_NORMAL_DEG = 140.6
DT = 30.0
DX = 0.010
SIGMA = 5.67e-8
LW_EPS = 0.9


# ---------------- 基础几何/太阳 ----------------
def solar_geometry(dt):
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    doy = dt.dayofyear
    lat_r = np.radians(LAT)
    decl = np.radians(23.45) * np.cos(2 * np.pi * (doy - 172) / 365.25)
    ha = np.radians((hour - 12) * 15)
    elev = np.arcsin(np.sin(lat_r) * np.sin(decl) + np.cos(lat_r) * np.cos(decl) * np.cos(ha))
    if elev > 0:
        cos_az = np.clip((np.sin(decl) - np.sin(elev) * np.sin(lat_r)) /
                         (np.cos(elev) * np.cos(lat_r)), -1, 1)
        azim = np.arccos(cos_az)
        if hour < 12:
            azim = -azim
    else:
        azim = 0.0
    return elev, azim


def sun_direction_2d(elev, azim):
    """与 SCI 初稿 fem_sensitivity_analysis.sun_direction_2d 完全一致。"""
    if elev <= 0:
        return np.array([0.0, 0.0])
    face_south = np.radians(FACE_NORMAL_DEG - 180.0)
    diff = azim - face_south
    sx, sy = np.cos(elev) * np.cos(diff), np.sin(elev)
    v = np.array([sx, sy])
    n = np.sqrt(v.dot(v))
    return v / n if n > 1e-9 else v


def _point_in_poly(x, y, poly):
    poly = np.asarray(poly, float)
    n = len(poly)
    inside = np.zeros(np.shape(x), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        cond = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi)
        inside ^= cond
        j = i
    return inside


# ---------------- 逐节点 cos_inc（含自遮挡） ----------------
def compute_cos_inc_fast(mesh, times_sub, geo_outer, geo_holes,
                         use_raytracing=True, deck=None):
    """向量化版 SCI 初稿 compute_cos_inc。
    geo_*: DWG 原始坐标（中心原点, mm）；内部转 FEM 坐标（+600/+300）。
    deck: None 或 dict(width=12600, clearance=1200) → 桥面系遮阳（mm）。
    """
    N = mesh["N"]
    NT = len(times_sub)
    cos_inc = np.zeros((NT, N), dtype=np.float32)
    surf_mask = mesh["is_outer_surf"]
    normals = mesh["surf_normal"]
    coords = mesh["coords"]  # mm, FEM 坐标(0..1200, 0..600)
    is_top = mesh["is_top"]; is_bot = mesh["is_bot"]

    outer = np.asarray(geo_outer, float) + np.array([600.0, 300.0])
    holes = [np.asarray(h, float) + np.array([600.0, 300.0]) for h in geo_holes]

    sidx = np.where(surf_mask)[0]
    P = coords[sidx]                    # (Ns,2)
    Nn = normals[sidx]                  # (Ns,2)
    ymax = outer[:, 1].max()
    xc = outer[:, 0].mean()          # 截面横向中心（FEM 坐标）
    if deck is not None:
        y_deck = ymax + deck["clearance"]
        xdL, xdR = xc - deck["width"] / 2.0, xc + deck["width"] / 2.0

    for ti in range(NT):
        elev, azim = solar_geometry(times_sub[ti])
        if elev <= 0:
            continue
        s_sun = sun_direction_2d(elev, azim)      # 表面→太阳
        s_inc = -s_sun                            # 太阳→截面
        ci = Nn[:, 0] * s_inc[0] + Nn[:, 1] * s_inc[1]   # = -n·s_sun
        lit = ci < 0                              # 面朝太阳
        if not lit.any():
            continue
        vals = -ci                                # cos_inc = n·s_sun
        if use_raytracing:
            # 自遮挡：从 P+n*eps 沿朝太阳方向步进，若进入截面内部则被遮
            eps = 3.0
            cur = P + Nn * eps
            dx, dy = s_sun[0], s_sun[1]
            hit = np.zeros(len(P), dtype=bool)
            for k in range(1, 61):
                q = cur + np.array([dx, dy]) * (eps * 2.0 * k)
                insec = _point_in_poly(q[:, 0], q[:, 1], outer)
                for h in holes:
                    insec &= ~_point_in_poly(q[:, 0], q[:, 1], h)
                hit |= insec
                if hit.all():
                    break
            vals = np.where(hit, 0.0, vals)
        # 桥面系遮阳（可选）：仅顶面节点
        if deck is not None:
            sun_up = s_sun[1] > 1e-6
            for i in np.where(lit & is_top[sidx])[0]:
                px, py = P[i]
                if sun_up:
                    x_cross = px + (y_deck - py) * (s_sun[0] / s_sun[1])
                    if xdL <= x_cross <= xdR:
                        vals[i] = 0.0
        v = vals.copy()
        v[~lit] = 0.0
        cos_inc[ti, sidx] = np.maximum(v, 0.0)
    return cos_inc


# ---------------- 向量化 FEM（严格等价 SCI run_fem） ----------------
def _build_ops(mesh):
    N = mesh["N"]
    coords = mesh["coords"] / 1000.0
    ei = mesh["edge_index"]; src, dst = ei[0], ei[1]
    neighbors = [[] for _ in range(N)]
    for e in range(ei.shape[1]):
        i, j = src[e], dst[e]
        neighbors[i].append(j); neighbors[j].append(i)

    def pad(idxs, weights_rule=None):
        K = max(len(neighbors[i]) for i in idxs)
        I = -np.ones((len(idxs), K), dtype=np.int64)
        W = np.zeros((len(idxs), K), dtype=np.float64)
        for ki, i in enumerate(idxs):
            for kj, j in enumerate(neighbors[i]):
                I[ki, kj] = j
                if weights_rule == 'inv_d2':
                    W[ki, kj] = 1.0 / (np.sum((coords[i] - coords[j]) ** 2) + 1e-12)
                else:
                    W[ki, kj] = 1.0
        return I, W

    interior_idx = np.where(mesh["is_interior"])[0]
    I_int, W_int = pad(interior_idx, 'inv_d2')
    int_n = 1.0 / np.array([len(neighbors[i]) for i in interior_idx], dtype=np.float64)

    def surf_ops(idx_arr):
        I = -np.ones((len(idx_arr), 1), dtype=np.int64); rows = []
        for ki, i in enumerate(idx_arr):
            inl = [j for j in neighbors[i] if mesh["is_interior"][j]] or neighbors[i]
            rows.append(inl)
        K = max(len(r) for r in rows)
        I = -np.ones((len(idx_arr), K), dtype=np.int64)
        for ki, r in enumerate(rows):
            for kj, j in enumerate(r):
                I[ki, kj] = j
        n = 1.0 / np.array([len(r) for r in rows], dtype=np.float64)
        return I, n

    outer_surf_idx = np.where(mesh["is_outer_surf"])[0]
    hole_idx = np.where(mesh["is_hole_edge"])[0]
    I_os, os_n = surf_ops(outer_surf_idx)
    I_hole, hole_n = surf_ops(hole_idx)
    return dict(interior_idx=interior_idx, I_int=I_int, W_int=W_int, int_n=int_n,
                outer_surf_idx=outer_surf_idx, I_os=I_os, os_n=os_n,
                hole_idx=hole_idx, I_hole=I_hole, hole_n=hole_n)


def _gather(T, I):
    idx = np.where(I < 0, 0, I)
    return T[idx], (I >= 0)


def run_fem_fast(mesh, bc, times_sub, cos_inc, hole_air, ops=None,
                 K=1.80, RHO=2400.0, CP=920.0, ALPHA_SOLAR=0.6,
                 H_FORCED=15.0, H_NATURAL=8.0, H_HOLE=5.0,
                 ws_scale=0.39, longwave_mode='none', longwave_svf_side=0.5,
                 longwave_svf_top=1.0, t2m_offset=0.0, lw_sky_drop=10.0):
    N = mesh["N"]
    if ops is None:
        ops = _build_ops(mesh)
    ALPHA = K / (RHO * CP)
    interior_idx = ops["interior_idx"]
    I_int, W_int, int_n = ops["I_int"], ops["W_int"], ops["int_n"]
    outer_surf_idx = ops["outer_surf_idx"]; I_os, os_n = ops["I_os"], ops["os_n"]
    hole_idx = ops["hole_idx"]; I_hole, hole_n = ops["I_hole"], ops["hole_n"]
    if not interior_idx.size:      # 全部节点都是表面(异常保护)
        pass

    do_lw = (longwave_mode not in (None, 'none', False))
    svf = np.zeros(len(outer_surf_idx))
    if do_lw:
        for ki, i in enumerate(outer_surf_idx):
            if mesh["is_top"][i]:
                svf[ki] = longwave_svf_top
            elif mesh["is_bot"][i]:
                svf[ki] = 0.0
            else:
                svf[ki] = longwave_svf_side

    n_hours = len(times_sub)
    steps_per_hour = int(3600 / DT)
    n_total = n_hours * steps_per_hour

    T_init = np.mean(bc.iloc[:24]["t2m_C"].values)
    T = np.full(N, T_init, dtype=np.float64)
    T_all = np.zeros((n_hours + 1, N), dtype=np.float64)
    T_all[0] = T.copy()

    kdx = K / DX
    DXS = DX * 0.5
    tau_outer = RHO * CP * DXS / (H_FORCED + kdx)
    tau_nat = RHO * CP * DXS / (H_NATURAL + kdx)
    tau_hole = RHO * CP * DXS / (H_HOLE + kdx)

    t2m_arr = bc["t2m_C"].values.astype(np.float64) + t2m_offset
    ws_arr = bc["ws_m_s"].values.astype(np.float64) * ws_scale
    ssrd_arr = bc["ssrd_W_m2"].values.astype(np.float64)
    # 孔内气温逐时序列（N212 实测；缺失回退 T_a）
    if hole_air is not None:
        hole_arr = hole_air.reindex(times_sub).values.astype(np.float64)
        hole_arr = np.where(np.isfinite(hole_arr), hole_arr, t2m_arr[:len(times_sub)])
    else:
        hole_arr = t2m_arr[:len(times_sub)]

    II = np.where(I_int < 0, 0, I_int); MI = I_int >= 0
    IO = np.where(I_os < 0, 0, I_os); MO = I_os >= 0
    IH = np.where(I_hole < 0, 0, I_hole); MH = I_hole >= 0

    dT = np.zeros(N)
    for step in range(1, n_total + 1):
        hi = step // steps_per_hour
        if hi >= n_hours: hi = n_hours - 1
        t2m = t2m_arr[hi]; ws = ws_arr[hi]; ssrd = ssrd_arr[hi]
        h_outer = H_FORCED if ws > 1.0 else H_NATURAL
        tau_surf = tau_outer if ws > 1.0 else tau_nat
        ci_row = cos_inc[hi]
        hole_val = hole_arr[hi]

        dT[:] = 0.0
        # 内部节点
        if interior_idx.size:
            Tnb = T[II]
            lap = (((Tnb - T[interior_idx][:, None]) * W_int) * MI).sum(axis=1)
            dT[interior_idx] = ALPHA * lap * int_n
        # 外表面
        coeff_outer = 1.0 / (kdx + h_outer)
        Tnb = T[IO]
        T_int_avg = ((Tnb * MO).sum(axis=1)) * os_n
        q_sol = ALPHA_SOLAR * ssrd * ci_row[outer_surf_idx]
        q_lw = np.zeros(len(outer_surf_idx))
        if do_lw:
            T_sky_C = t2m - lw_sky_drop * (1.0 - np.clip(ssrd / 1000.0, 0.0, 1.0))
            q_lw = LW_EPS * SIGMA * svf * ((T_sky_C + 273.15) ** 4 - (T[outer_surf_idx] + 273.15) ** 4)
        Teq = (kdx * T_int_avg + h_outer * t2m + q_sol + q_lw) * coeff_outer
        dT[outer_surf_idx] = (Teq - T[outer_surf_idx]) / tau_surf
        # 孔边
        coeff_hole = 1.0 / (kdx + H_HOLE)
        Tnb = T[IH]
        T_int_avg = ((Tnb * MH).sum(axis=1)) * hole_n
        Teq = (kdx * T_int_avg + H_HOLE * hole_val) * coeff_hole
        dT[hole_idx] = (Teq - T[hole_idx]) / tau_hole

        T += dT * DT
        if step % steps_per_hour == 0:
            T_all[step // steps_per_hour] = T
        if not np.isfinite(T).all():
            raise RuntimeError("NaN at step %d" % step)
    return T_all
