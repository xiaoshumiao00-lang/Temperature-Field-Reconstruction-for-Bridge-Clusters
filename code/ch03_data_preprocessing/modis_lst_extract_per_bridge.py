# -*- coding: utf-8 -*-
"""MODIS LST 逐桥提取：从 308 个 HDF4 文件中为 298 座桥提取每日 LST。
输出: modis_lst_per_bridge.npy [N=298, T=76]  每桥每日平均LST(℃)，缺值=NaN。
"""
import os, re, glob, math, csv, numpy as np
from pyhdf.SD import SD, SDC
import datetime as _dt

R = 6371007.181
XDim = YDim = 1200

# ---- 读桥坐标 ----
lats, lons = [], []
with open("bridge_cluster_Fengjie_Chongqing.csv") as f:
    for r in csv.DictReader(f):
        lats.append(float(r["lat"])); lons.append(float(r["lon"]))
lats = np.array(lats); lons = np.array(lons)
N = len(lats)

def lonlat_to_meters(lon, lat):
    phi = math.radians(lat); lam = math.radians(lon)
    return R * lam * math.cos(phi), R * phi

def get_tile_bounds(meta):
    """返回 (x0,y0,x1,y1) 正弦投影米"""
    ul = re.search(r"UpperLeftPointMtrs=\(([-\d.]+),([-\d.]+)\)", meta)
    lr = re.search(r"LowerRight(?:Point)?Mtrs=\(([-\d.]+),([-\d.]+)\)", meta)
    return float(ul.group(1)), float(ul.group(2)), float(lr.group(1)), float(lr.group(2))

def meters_to_rc(x, y, x0, y0, x1, y1):
    col = (x - x0) / ((x1 - x0) / XDim)
    row = (y0 - y) / ((y0 - y1) / YDim)
    return row, col

def best_pixel(arr, row_f, col_f):
    r0 = int(round(row_f)); c0 = int(round(col_f))
    best = None; best_d = 999
    for dr in range(-5, 6):
        for dc in range(-5, 6):
            r, c = r0+dr, c0+dc
            if 0 <= r < 1200 and 0 <= c < 1200:
                v = int(arr[r, c])
                if 7500 <= v <= 65535:
                    d = dr*dr + dc*dc
                    if d < best_d:
                        best_d = d; best = v
    return best

# ---- 预计算：每座桥在哪个tile的投影位置 ----
# 先从一个文件取元数据确定tile边界
sample_files = {}
for f in sorted(glob.glob("modis_lst_2025/*.hdf"))[:10]:
    m = re.search(r"\.(h\d+v\d+)\.", f)
    if m:
        tid = m.group(1)
        if tid not in sample_files:
            sample_files[tid] = f

tile_info = {}   # tid -> (x0,y0,x1,y1)
for tid, fp in sample_files.items():
    sd = SD(fp, SDC.READ)
    meta = sd.attributes().get("StructMetadata.0", "")
    tile_info[tid] = get_tile_bounds(meta)
    sd.end()
print(f"Tile 边界: {tile_info}")

# 预算每桥在各tile的(row,col)
bridge_tiles = {}  # tid -> [(idx, row_f, col_f)]
for tid, (x0,y0,x1,y1) in tile_info.items():
    blist = []
    for i in range(N):
        xm, ym = lonlat_to_meters(lons[i], lats[i])
        if x0 <= xm <= x1 and y1 <= ym <= y0:  # 在tile内
            rf, cf = meters_to_rc(xm, ym, x0, y0, x1, y1)
            if 0 <= rf < XDim and 0 <= cf < YDim:
                blist.append((i, rf, cf))
    bridge_tiles[tid] = blist
    print(f"  {tid}: {len(blist)} 座桥在覆盖范围内")

total_covered = sum(len(v) for v in bridge_tiles.values())
print(f"共 {total_covered}/{N} 座桥有 MODIS 覆盖")

# ---- 逐日提取 ----
T = 76  # 2025-04-01 起 76 天
lst_matrix = np.full((N, T), np.nan, dtype=np.float32)

files_by_day = {}  # doy -> [filepath]
for f in sorted(glob.glob("modis_lst_2025/*.hdf")):
    m = re.search(r"A2025(\d{3})\.(h\d+v\d+)\.", f)
    if not m: continue
    doy = int(m.group(1)); tid = m.group(2)
    date = _dt.date(2025, 1, 1) + _dt.timedelta(days=doy-1)
    day_idx = (date - _dt.date(2025, 4, 1)).days
    if 0 <= day_idx < T:
        files_by_day.setdefault(day_idx, []).append(f)

print(f"有效天数: {len(files_by_day)}")

for di in range(T):
    if di not in files_by_day:
        continue
    for fp in files_by_day[di]:
        m = re.search(r"\.(h\d+v\d+)\.", fp)
        tid = m.group(1) if m else None
        if tid not in bridge_tiles or not bridge_tiles[tid]:
            continue
        sd = SD(fp, SDC.READ)
        for dsname in ["LST_Day_1km", "LST_Night_1km"]:
            try:
                arr = sd.select(dsname)[:]
                for bi, rf, cf in bridge_tiles[tid]:
                    v = best_pixel(arr, rf, cf)
                    if v is not None:
                        t_c = v * 0.02 - 273.15
                        if np.isnan(lst_matrix[bi, di]):
                            lst_matrix[bi, di] = t_c
                        else:
                            lst_matrix[bi, di] = (lst_matrix[bi, di] + t_c) / 2.0  # 昼夜平均
            except Exception:
                pass
        sd.end()
    if (di+1) % 15 == 0 or di == T-1:
        n_valid = int(np.sum(~np.isnan(lst_matrix[:, di])))
        print(f"  Day {di+1}/{T}: {n_valid} 桥有LST")

n_total_valid = int(np.sum(~np.isnan(lst_matrix)))
print(f"\n总有效值: {n_total_valid} / {N*T}")
print(f"LST 范围: [{np.nanmin(lst_matrix):.1f}, {np.nanmax(lst_matrix):.1f}]℃")
np.save("modis_lst_per_bridge.npy", lst_matrix)
print("✅ 已保存 modis_lst_per_bridge.npy")
