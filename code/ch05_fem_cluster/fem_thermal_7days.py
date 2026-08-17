#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
梅溪河特大桥 M04 截面 · 有限元瞬态热传导分析
==============================================
- 三孔箱梁截面 (1200×600mm), 10mm 规则网格, 显式时间积分
- 动态太阳辐射 (逐节点 cos_inc 计算)
- 热边界: 外表面=对流+太阳辐射, 孔洞=对流(自然), 内部=纯导热
- 3 天起转 + 7 个连续晴天温度场模拟
- 输出: 每日温度云图 / 7 天时程曲线 / 15:00 深度剖面

材料: C50 混凝土, k=1.80 W/(m·K), ρ=2400 kg/m³, cp=920 J/(kg·K)
"""
import numpy as np
import pandas as pd
import os, json, sys, warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 0. 路径与参数
# ============================================================
ROOT = r"D:\陈潘HBEU\Desktop\气象桥梁温度场"
FEM_DIR = os.path.join(ROOT, "12梅溪河特大桥截面", "FEM")
PIPE = os.path.join(ROOT, "12梅溪河特大桥截面", "gnn_pipeline")
OUT = os.path.join(FEM_DIR, "FEMcode", "output")
os.makedirs(OUT, exist_ok=True)

K = 1.80; RHO = 2400.0; CP = 920.0
ALPHA = K / (RHO * CP)   # 8.152e-7 m²/s
DX = 0.010                # 10mm → m

H_FORCED = 15.0; H_NATURAL = 8.0; H_HOLE = 5.0
ALPHA_SOLAR = 0.6         # 太阳辐射吸收率
LAT = 31.06
FACE_NORMAL_DEG = 140.6   # 截面法向方位角

DT = 30.0                 # 显式时间步长 30s (α·Δt/Δx² = 0.245 < 0.25 稳定)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 数据加载
# ============================================================
def load_data():
    print("=" * 60)
    print("1. 加载数据")
    print("=" * 60)
    mesh_path = os.path.join(PIPE, "phase1_data.npz")
    if not os.path.exists(mesh_path):
        print("ERROR: 网格文件不存在"); sys.exit(1)

    d = np.load(mesh_path, allow_pickle=True)
    coords = d["valid_coords"]
    edge_index = d["edge_index"]
    is_top = d["is_top"]; is_bot = d["is_bot"]
    is_right = d["is_right"]; is_left = d["is_left"]
    is_hole_edge = d["is_hole_edge"]; is_surface = d["is_surface"]
    surf_normal = d["surf_normal"]

    N = coords.shape[0]; E = edge_index.shape[1]
    is_outer_surf = is_surface & (~is_hole_edge)
    is_interior = ~is_surface

    print(f"  网格: {N} 节点, {E} 边")
    print(f"  表面: {is_surface.sum()} (外={is_outer_surf.sum()}, 孔={is_hole_edge.sum()}), "
          f"内部={is_interior.sum()}")
    print(f"  顶/底/右/左: {is_top.sum()}/{is_bot.sum()}/{is_right.sum()}/{is_left.sum()}")

    bc = pd.read_csv(os.path.join(FEM_DIR, "boundary_2024_M04.csv"))
    bc["datetime"] = pd.to_datetime(bc["datetime"])
    bc = bc.set_index("datetime").sort_index()
    # ERA5 数据是 UTC 时间, 转换为 UTC+8 中国本地时间 (用于太阳几何计算)
    bc.index = bc.index + pd.Timedelta(hours=8)
    print(f"  边界条件: {len(bc)}h ({bc.index[0]}~{bc.index[-1]}) [已 UTC→UTC+8]")
    print(f"    T: {bc['t2m_C'].min():.1f}~{bc['t2m_C'].max():.1f}°C  "
          f"ws: {bc['ws_m_s'].min():.1f}~{bc['ws_m_s'].max():.1f} m/s  "
          f"ssrd: {bc['ssrd_W_m2'].min():.0f}~{bc['ssrd_W_m2'].max():.0f} W/m²")

    hole_air = None
    ht_path = os.path.join(FEM_DIR, "cleaned_temperatures_2024.csv")
    if os.path.exists(ht_path):
        ht = pd.read_csv(ht_path); ht["DateTime"] = pd.to_datetime(ht["DateTime"])
        ht = ht.set_index("DateTime").sort_index()
        if "N212" in ht.columns:
            common = bc.index.intersection(ht.index)
            if len(common) > 0:
                hole_air = ht["N212"].loc[common]
                print(f"  孔内气温 N212: {len(hole_air)}h")

    mesh = {"coords": coords, "edge_index": edge_index,
            "is_top": is_top, "is_bot": is_bot, "is_right": is_right, "is_left": is_left,
            "is_hole_edge": is_hole_edge, "is_surface": is_surface,
            "is_outer_surf": is_outer_surf, "is_interior": is_interior,
            "surf_normal": surf_normal, "N": N, "E": E}
    return mesh, bc, hole_air


# ============================================================
# 2. 寻找 7 个连续晴天
# ============================================================
def find_clear_days(bc, n_days=7, spinup_days=3):
    print("\n" + "=" * 60)
    print("2. 寻找连续晴天 (按辐射总和高且日内曲线平滑)")
    print("=" * 60)

    bc_w = bc.copy()
    bc_w["date"] = bc_w.index.date; bc_w["hour"] = bc_w.index.hour
    dates = sorted(bc_w["date"].unique())

    scores = {}
    for d in dates:
        day = bc_w[bc_w["date"] == d]
        ssrd = day["ssrd_W_m2"].values
        hours = day["hour"].values.astype(float)
        if ssrd.max() < 100: continue
        # 仅取白天小时(9-16时本地时间)
        dm = (hours >= 9) & (hours <= 16)
        if dm.sum() < 4: continue

        # 日总辐射 (W·h/m²)
        daily_sum = ssrd.sum()
        # 峰值辐射
        peak = ssrd.max()
        # 平滑度: 实际曲线与理想曲线的相似度
        h_day = hours[dm]
        # 理想晴空曲线: 9-16时的半正弦
        ideal = np.sin(np.pi * (h_day - 9) / 7); ideal = np.maximum(ideal, 0)

        ssrd_d = ssrd[dm]; sm = ssrd_d.max()
        if sm > 0:
            corr = np.corrcoef(ssrd_d / sm, ideal)[0, 1]
            if np.isnan(corr): corr = 0
        else:
            corr = 0
        # 综合评分: 日总辐射 * 平滑度
        # 用 |corr| 处理负相关(可能是日出/日落高山区特性)
        scores[d] = {"score": abs(corr) * daily_sum * 1e-4, "corr": corr,
                      "daily_sum": daily_sum, "ssrd_max": peak}

    print(f"  有效评分日: {len(scores)}")

    sd = sorted(scores.keys())
    best_score, best_start = -1, None
    for i in range(len(sd) - n_days + 1):
        w = sd[i:i+n_days]
        if not all((w[j+1]-w[j]).days == 1 for j in range(n_days-1)): continue
        avg = np.mean([scores[d]["score"] for d in w])
        if avg > best_score: best_score, best_start = avg, w[0]

    if best_start is None: raise RuntimeError("未找到连续晴天!")

    import datetime as _dt
    spinup_start = best_start - _dt.timedelta(days=spinup_days)
    clear_dates = [best_start + _dt.timedelta(days=i) for i in range(n_days)]
    spinup_list = [spinup_start + _dt.timedelta(days=i) for i in range(spinup_days)]

    print(f"  起转: {spinup_start}~{spinup_list[-1]}, 晴天: {clear_dates[0]}~{clear_dates[-1]}")
    for d in clear_dates:
        s = scores[d]
        print(f"    {d}: score={s['score']:.1f} corr={s['corr']:.3f} "
              f"sum={s['daily_sum']:.0f} max={s['ssrd_max']:.0f}")
    return spinup_list + clear_dates, spinup_list, clear_dates, scores


# ============================================================
# 3. 太阳位置与 cos_inc
# ============================================================
def solar_geometry(dt):
    hour = dt.hour + dt.minute/60.0 + dt.second/3600.0
    doy = dt.dayofyear; lat_r = np.radians(LAT)
    decl = np.radians(23.45) * np.cos(2*np.pi*(doy-172)/365.25)
    ha = np.radians((hour-12)*15)
    elev = np.arcsin(np.sin(lat_r)*np.sin(decl) + np.cos(lat_r)*np.cos(decl)*np.cos(ha))
    if elev > 0:
        cos_az = np.clip((np.sin(decl)-np.sin(elev)*np.sin(lat_r))
                         / (np.cos(elev)*np.cos(lat_r)), -1, 1)
        azim = np.arccos(cos_az)
        if hour < 12: azim = -azim
    else:
        azim = 0.0
    return elev, azim


def sun_direction_2d(elev, azim):
    if elev <= 0: return np.array([0.0, 0.0])
    diff = np.radians(FACE_NORMAL_DEG-90) - azim
    sx, sy = np.cos(elev)*np.cos(diff), np.sin(elev)
    v = np.array([sx, sy]); n = np.sqrt(v.dot(v))
    return v/n if n > 1e-9 else v


def compute_cos_inc_all(mesh, bc_sub, times_sub):
    print("\n" + "=" * 60)
    print("3. 计算逐节点太阳入射角")
    print("=" * 60)
    NT, N = len(times_sub), mesh["N"]
    cos_inc = np.zeros((NT, N), dtype=np.float32)
    surf_mask = mesh["is_outer_surf"]; normals = mesh["surf_normal"]

    for ti in range(NT):
        if ti % 500 == 0: print(f"  cos_inc: {ti}/{NT}")
        elev, azim = solar_geometry(times_sub[ti])
        if elev > 0:
            sx, sy = sun_direction_2d(elev, azim)
            for ni in range(N):
                if not surf_mask[ni]: continue
                ci = normals[ni,0]*sx + normals[ni,1]*sy
                cos_inc[ti, ni] = max(ci, 0.0)

    n_lit = (cos_inc.max(axis=0) > 0).sum()
    print(f"  cos_inc 完成. 照射节点: {n_lit}/{surf_mask.sum()}")
    return cos_inc


# ============================================================
# 4. 显式 FEM 求解器 (与 compute_fem_explicit.py 一致)
# ============================================================
def run_fem_explicit(mesh, bc, times_sub, cos_inc, hole_air):
    """显式时间积分 FEM (Euler forward) — 预计算+向量化加速."""
    print("\n" + "=" * 60)
    print("4. 运行显式 FEM 时间积分(预计算邻居结构)")
    print("=" * 60)

    N = mesh["N"]; coords = mesh["coords"] / 1000.0
    ei = mesh["edge_index"]; src, dst = ei[0], ei[1]

    # ---- 预计算邻居结构 ----
    # 先收集所有边
    neighbors = [[] for _ in range(N)]
    for e in range(ei.shape[1]):
        i, j = src[e], dst[e]
        neighbors[i].append(j); neighbors[j].append(i)

    # 内部节点: 预计算邻居idx + 1/d²
    interior_idx = np.where(mesh["is_interior"])[0]
    N_int = len(interior_idx)
    # 用 ragged array: 先找最大邻居数
    max_nbs = max(len(neighbors[i]) for i in interior_idx)
    int_nbr_idx = -np.ones((N_int, max_nbs), dtype=np.int32)
    int_nbr_w = np.zeros((N_int, max_nbs), dtype=np.float64)   # 1/d²
    int_nbr_n = np.zeros(N_int, dtype=np.float64)               # 1/n_neighbors
    for ki, i in enumerate(interior_idx):
        nbs = neighbors[i]
        nn = len(nbs); int_nbr_n[ki] = 1.0 / nn
        for kj, j in enumerate(nbs):
            int_nbr_idx[ki, kj] = j
            d2 = np.sum((coords[i] - coords[j])**2) + 1e-12
            int_nbr_w[ki, kj] = 1.0 / d2

    # 表面节点: 预计算内部邻居
    outer_surf_idx = np.where(mesh["is_outer_surf"])[0]
    hole_idx = np.where(mesh["is_hole_edge"])[0]
    N_os = len(outer_surf_idx); N_hole = len(hole_idx)

    max_inl = 0
    for i in np.concatenate([outer_surf_idx, hole_idx]):
        inl = [j for j in neighbors[i] if mesh["is_interior"][j]]
        if not inl: inl = neighbors[i]
        max_inl = max(max_inl, len(inl))

    os_int_nbr = -np.ones((N_os, max_inl), dtype=np.int32)
    hole_int_nbr = -np.ones((N_hole, max_inl), dtype=np.int32)
    os_int_n = np.zeros(N_os, dtype=np.float64)
    hole_int_n = np.zeros(N_hole, dtype=np.float64)

    for ki, i in enumerate(outer_surf_idx):
        inl = [j for j in neighbors[i] if mesh["is_interior"][j]]
        if not inl: inl = neighbors[i]
        nn = len(inl); os_int_n[ki] = 1.0 / nn
        for kj, j in enumerate(inl):
            os_int_nbr[ki, kj] = j

    for ki, i in enumerate(hole_idx):
        inl = [j for j in neighbors[i] if mesh["is_interior"][j]]
        if not inl: inl = neighbors[i]
        nn = len(inl); hole_int_n[ki] = 1.0 / nn
        for kj, j in enumerate(inl):
            hole_int_nbr[ki, kj] = j

    # 时间参数
    n_hours = len(times_sub)
    steps_per_hour = int(3600 / DT)
    n_total = n_hours * steps_per_hour
    output_step = steps_per_hour

    T_init = np.mean(bc.iloc[:24]["t2m_C"].values)
    T = np.full(N, T_init, dtype=np.float64)
    print(f"  初始温度: {T_init:.1f}°C, 总步数={n_total}, DT={DT}s")
    print(f"  内部节点: {N_int}, 外表面: {N_os}, 孔边缘: {N_hole}")

    T_all = np.zeros((n_hours + 1, N), dtype=np.float64)
    T_all[0] = T.copy()

    kdx = K / DX

    # 预计算不随时间变化的量
    # 表面热容层厚度 = DX/2 (FVM 边界控制体积半格, 与 ANSYS FEM 半层热容一致)
    DXS = DX * 0.5
    tau_outer = RHO * CP * DXS / (H_FORCED + kdx)
    tau_nat = RHO * CP * DXS / (H_NATURAL + kdx)
    tau_hole = RHO * CP * DXS / (H_HOLE + kdx)

    # 预提取 BC 数组
    t2m_arr = bc["t2m_C"].values.astype(np.float64)
    ws_arr = bc["ws_m_s"].values.astype(np.float64)
    ssrd_arr = bc["ssrd_W_m2"].values.astype(np.float64)
    cos_inc_arr = cos_inc  # [NT, N]

    for step in range(1, n_total + 1):
        hi = step // steps_per_hour
        if hi >= n_hours: hi = n_hours - 1

        t2m = t2m_arr[hi]; ws = ws_arr[hi]; ssrd = ssrd_arr[hi]
        h_outer = H_FORCED if ws > 1.0 else H_NATURAL
        tau_surf = tau_outer if ws > 1.0 else tau_nat
        ci_row = cos_inc_arr[hi]

        # 孔内气温 (NaN-safe)
        hole_val = t2m  # fallback
        if hole_air is not None:
            tnow = times_sub[hi]
            if tnow in hole_air.index:
                hv = hole_air.loc[tnow]
                if pd.notna(hv):
                    hole_val = float(hv)

        dT = np.zeros(N, dtype=np.float64)

        # ---- 内部节点: 向量化拉普拉斯 ----
        for ki in range(N_int):
            i = interior_idx[ki]
            T_i = T[i]
            lap = 0.0
            for kj in range(int_nbr_w.shape[1]):
                j = int_nbr_idx[ki, kj]
                if j < 0: break
                lap += (T[j] - T_i) * int_nbr_w[ki, kj]
            dT[i] = ALPHA * lap * int_nbr_n[ki]

        # ---- 外表面: 对流+辐射 ----
        coeff_outer = 1.0 / (kdx + h_outer)
        for ki in range(N_os):
            i = outer_surf_idx[ki]
            T_int_sum = 0.0; nn = 0
            for kj in range(os_int_nbr.shape[1]):
                j = os_int_nbr[ki, kj]
                if j < 0: break
                T_int_sum += T[j]; nn += 1
            if nn == 0: continue
            T_int_avg = T_int_sum * os_int_n[ki]
            q_sol = ALPHA_SOLAR * ssrd * ci_row[i]
            Teq = (kdx * T_int_avg + h_outer * t2m + q_sol) * coeff_outer
            dT[i] = (Teq - T[i]) / tau_surf

        # ---- 孔边缘: 对流 ----
        coeff_hole = 1.0 / (kdx + H_HOLE)
        for ki in range(N_hole):
            i = hole_idx[ki]
            T_int_sum = 0.0; nn = 0
            for kj in range(hole_int_nbr.shape[1]):
                j = hole_int_nbr[ki, kj]
                if j < 0: break
                T_int_sum += T[j]; nn += 1
            if nn == 0: continue
            T_int_avg = T_int_sum * hole_int_n[ki]
            Teq = (kdx * T_int_avg + H_HOLE * hole_val) * coeff_hole
            dT[i] = (Teq - T[i]) / tau_hole

        T += dT * DT

        # NaN guard: 若出现 NaN, 终止仿真
        if not np.isfinite(T).all():
            bad_idx = np.where(~np.isfinite(T))[0]
            n_bad = len(bad_idx)
            bad_hole = sum(1 for i in bad_idx if mesh["is_hole_edge"][i])
            bad_outer = sum(1 for i in bad_idx if mesh["is_outer_surf"][i])
            bad_int = sum(1 for i in bad_idx if mesh["is_interior"][i])
            print(f"  NaN step={step} hi={hi}, total_bad={n_bad} "
                  f"(hole={bad_hole} outer={bad_outer} int={bad_int}), "
                  f"t2m={t2m:.1f} ws={ws:.2f} ssrd={ssrd:.0f}")
            break

        if step % output_step == 0:
            T_all[step // output_step] = T.copy()

        if step % 4000 == 0:
            n_nan = (~np.isfinite(T)).sum()
            print(f"  FEM {step}/{n_total}: "
                  f"T={T.min():.1f}~{T.max():.1f}°C mean={T.mean():.1f} NaN={n_nan}")
            if n_nan > 0 and n_nan < 20:
                bad_idx = np.where(~np.isfinite(T))[0]
                for bi in bad_idx[:5]:
                    ntype = ("int" if mesh["is_interior"][bi] else
                             ("outer" if mesh["is_outer_surf"][bi] else "hole"))
                    print(f"    nan {bi}: type={ntype} pos=({mesh['coords'][bi,0]:.0f},{mesh['coords'][bi,1]:.0f})")

    print(f"  FEM 完成. T: {T.min():.1f}~{T.max():.1f}°C")
    return T_all


# ============================================================
# 5. 可视化
# ============================================================
def plot_section_temperature(ax, coords_mm, T, title, vmin=None, vmax=None,
                              outline=True, cmap='jet'):
    x, y = coords_mm[:, 0], coords_mm[:, 1]
    if vmin is None: vmin = T.min()
    if vmax is None: vmax = T.max()

    # 内部先画, 表面后画 (突出表面)
    sc = ax.scatter(x, y, c=T, cmap=cmap, s=2, alpha=0.95,
                    vmin=vmin, vmax=vmax)

    if outline:
        geo_path = os.path.join(FEM_DIR, "dwg_geometry.json")
        if os.path.exists(geo_path):
            with open(geo_path) as f: geo = json.load(f)
            OFF_X, OFF_Y = 600.0, 300.0
            op = np.array([(p[0]+OFF_X, p[1]+OFF_Y) for p in geo["outer"]])
            ax.plot(op[:,0], op[:,1], 'k-', lw=1.2)
            for hk in ["hole_left","hole_center","hole_right"]:
                hp = np.array([(p[0]+OFF_X, p[1]+OFF_Y) for p in geo[hk]])
                ax.fill(hp[:,0], hp[:,1], facecolor='white', edgecolor='gray', lw=0.8)

    ax.set_aspect('equal'); ax.set_xlim(-10, 1210); ax.set_ylim(-10, 610)
    ax.set_title(title, fontsize=9)
    return sc


def plot_daily_fields_24h(T_daily, mesh, times_daily, clear_dates, day_idx, out_dir):
    """一天 24 小时温度云图 (4×6=24 幅)."""
    d = clear_dates[day_idx]
    coords_mm = mesh["coords"]
    vmin, vmax = T_daily.min(), T_daily.max()

    fig, axes = plt.subplots(4, 6, figsize=(26, 18))
    fig.suptitle(f'2024-{d.month:02d}-{d.day:02d}  24 小时截面温度场 (°C)',
                 fontsize=15, fontweight='bold')

    for hour in range(24):
        ax = axes[hour // 6, hour % 6]
        tidx = hour
        if tidx >= len(T_daily): tidx = len(T_daily) - 1
        Tp = T_daily[tidx]
        sc = plot_section_temperature(ax, coords_mm, Tp,
                                       f'{hour:02d}:00  {Tp.min():.1f}~{Tp.max():.1f}°C',
                                       vmin=vmin, vmax=vmax)

    cbar_ax = fig.add_axes([0.93, 0.06, 0.012, 0.88])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Temperature (°C)', fontsize=11)

    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    out_path = os.path.join(out_dir, f"温度云图_day{day_idx+1}_{d}_24h.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  24h 云图: {out_path}")
    return out_path


def plot_time_history(T_all, mesh, times_all, clear_start_idx, clear_end_idx, out_dir):
    """7 天温度时程 - 外表面 + 内部节点."""
    print("\n" + "=" * 60)
    print("5b. 温度时程曲线")
    print("=" * 60)

    coords = mesh["coords"]
    # 选点
    top_n = np.argmin(np.abs(coords[:,0]-600) + np.abs(coords[:,1]-600))
    bot_n = np.argmin(np.abs(coords[:,0]-600) + np.abs(coords[:,1]-0))
    right_n = np.argmin(np.abs(coords[:,0]-1200) + np.abs(coords[:,1]-300))
    left_n = np.argmin(np.abs(coords[:,0]-0) + np.abs(coords[:,1]-300))
    center_n = np.argmin(np.abs(coords[:,0]-600) + np.abs(coords[:,1]-300))
    mid_n = np.argmin(np.abs(coords[:,0]-600) + np.abs(coords[:,1]-150))
    hole_ns = np.where(mesh["is_hole_edge"])[0]
    hole_r = hole_ns[len(hole_ns)//4]

    sel = {"顶面": top_n, "底面": bot_n, "右侧": right_n, "左侧": left_n,
           "中心": center_n, "内部中点": mid_n, "孔边缘": hole_r}

    T_clear = T_all[clear_start_idx:clear_end_idx+1]
    t_clear = times_all[clear_start_idx:clear_end_idx+1]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle('7 天连续晴天 · 截面温度时程', fontsize=14, fontweight='bold')
    cols = ['#D32F2F','#1976D2','#388E3C','#F57C00','#7B1FA2','#0097A7','#C62828']

    for li, (label, idx) in enumerate([("顶面", top_n), ("底面", bot_n),
                                         ("右侧", right_n), ("左侧", left_n)]):
        ax1.plot(t_clear, T_clear[:,idx], color=cols[li], lw=1.2,
                 label=f'{label} [{coords[idx,0]:.0f},{coords[idx,1]:.0f}]mm')
    ax1.set_ylabel('Temperature (°C)'); ax1.legend(ncol=2, fontsize=8)
    ax1.grid(alpha=0.3); ax1.set_title('外表面节点')

    for li, (label, idx) in enumerate([("中心", center_n), ("内部中点", mid_n),
                                         ("孔边缘", hole_r)]):
        ax2.plot(t_clear, T_clear[:,idx], color=cols[li+3], lw=1.2,
                 label=f'{label} [{coords[idx,0]:.0f},{coords[idx,1]:.0f}]mm')
    ax2.set_xlabel('Time'); ax2.set_ylabel('Temperature (°C)')
    ax2.legend(ncol=2, fontsize=8); ax2.grid(alpha=0.3); ax2.set_title('内部节点')

    plt.tight_layout()
    out_path = os.path.join(out_dir, "温度时程曲线_7天.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  时程曲线: {out_path}")
    return out_path


def plot_depth_profile(T_all, mesh, times_all, clear_start_idx, clear_end_idx, out_dir):
    """15:00 沿深度温度分布."""
    print("\n" + "=" * 60)
    print("5c. 15:00 深度剖面")
    print("=" * 60)

    coords = mesh["coords"]
    mask = (np.abs(coords[:,0]-600) < 8) & ~mesh["is_hole_edge"]
    col_nodes = np.where(mask)[0]
    col_sorted = col_nodes[np.argsort(coords[col_nodes, 1])]

    n_hours_total = clear_end_idx - clear_start_idx + 1
    n_days = n_hours_total // 24
    n_days = min(n_days, 7)

    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_days))

    for di in range(n_days):
        pm3_idx = clear_start_idx + di*24 + 15  # 15:00
        if pm3_idx >= len(T_all): pm3_idx = len(T_all)-1
        T_3pm = T_all[pm3_idx]
        depths = coords[col_sorted, 1]; temps = T_3pm[col_sorted]
        tnow = times_all[pm3_idx]
        ax.plot(temps, depths, color=colors[di], lw=1.8, marker='o', ms=3,
                label=f'{pd.Timestamp(tnow).strftime("%m-%d")} 15:00')

    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('y (mm)', fontsize=12)
    ax.set_title('沿中轴线 (x=600mm) 深度-温度分布 @15:00', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.axhline(y=300, color='gray', ls='--', lw=0.8, alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "深度温度剖面_15时.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  深度剖面: {out_path}")
    return out_path


# ============================================================
# 6. 主流程
# ============================================================
def main():
    print("=" * 60)
    print("梅溪河特大桥 M04 截面 · FEM 热传导分析")
    print("=" * 60)

    mesh, bc, hole_air = load_data()
    all_dates, spinup_dates, clear_dates, scores = find_clear_days(bc, n_days=7, spinup_days=3)

    # 时间窗口
    start_ts = pd.Timestamp(all_dates[0])
    end_ts = pd.Timestamp(all_dates[-1]) + pd.Timedelta(hours=23)
    times_hourly = pd.date_range(start_ts, end_ts, freq='h')
    times_available = pd.DatetimeIndex(sorted([t for t in times_hourly if t in bc.index]))

    print(f"\n  模拟: {times_available[0]}~{times_available[-1]}, {len(times_available)}h")

    # cos_inc
    cos_inc = compute_cos_inc_all(mesh, bc.loc[times_available], times_available)

    # 运行 FEM
    T_all = run_fem_explicit(mesh, bc.loc[times_available], times_available, cos_inc, hole_air)

    # 晴天期索引
    n_spinup = len(spinup_dates) * 24  # 小时
    clear_start_h = n_spinup
    clear_end_h = len(times_available)  # +1 因为有 t=0

    print(f"\n  FAM 晴天: {clear_start_h}~{clear_end_h}h")

    # ---- 可视化 (可选, 被文件锁时可跳过) ----
    if os.environ.get("FEM_SKIP_PLOT") == "1":
        print("  跳过可视化 (FEM_SKIP_PLOT=1)")
    else:
        print("\n" + "=" * 60)
        print("5. 生成可视化")
        print("=" * 60)
        for di in range(len(clear_dates)):
            ds = clear_start_h + di*24
            de = ds + 24
            T_daily = T_all[ds:de+1]
            t_daily = times_available[ds:de+1]
            plot_daily_fields_24h(T_daily, mesh, t_daily, clear_dates, di, OUT)

        plot_time_history(T_all, mesh, times_available, clear_start_h, clear_end_h-1, OUT)
        plot_depth_profile(T_all, mesh, times_available, clear_start_h, clear_end_h-1, OUT)

    # ---- 保存 ----
    print("\n" + "=" * 60)
    print("6. 保存结果")
    print("=" * 60)
    save_path = os.path.join(OUT, "fem_result_7days_v2.npz")
    np.savez_compressed(save_path,
        T_all=T_all.astype(np.float32),
        times=np.array([t.isoformat() for t in times_available]),
        coords=mesh["coords"],
        is_surface=mesh["is_surface"], is_top=mesh["is_top"],
        is_bot=mesh["is_bot"], is_right=mesh["is_right"], is_left=mesh["is_left"],
        is_hole_edge=mesh["is_hole_edge"],
        clear_dates=np.array([str(d) for d in clear_dates]),
        clear_start_h=clear_start_h, clear_end_h=clear_end_h,
    )
    print(f"  {save_path}")

    # 统计
    T_clear = T_all[clear_start_h:clear_end_h]
    top_t = T_clear[:, mesh["is_top"]].max()
    bot_t = T_clear[:, mesh["is_bot"]].min()
    print(f"\n  FEM 完成!")
    print(f"  7天温度: {T_clear.min():.1f}~{T_clear.max():.1f}°C")
    print(f"  顶面最高: {top_t:.1f}°C, 底面最低: {bot_t:.1f}°C")
    print(f"  日较差均值: {T_clear.max(axis=1).mean()-T_clear.min(axis=1).mean():.1f}°C")
    print(f"  输出目录: {OUT}")


if __name__ == "__main__":
    main()
