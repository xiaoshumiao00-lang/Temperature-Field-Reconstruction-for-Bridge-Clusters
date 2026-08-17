# -*- coding: utf-8 -*-
"""ERA5 真实驱动数据下载（Copernicus CDS）— 2025-04-01~2025-06-16 同期窗口
变量: 2m气温/10m风/地表太阳短波辐射/总云量/地表气压
输出: era5_fengjie_2025.nc (精确裁剪到 2025-04-01..2025-06-16)
"""
import os
TOKEN_FILE = ".cds_token.txt"
token = open(TOKEN_FILE).read().strip()
rc = os.path.expanduser("~/.cdsapirc")
open(rc, "w").write(f"url: https://cds.climate.copernicus.eu/api\nkey: {token}\n")
import cdsapi
c = cdsapi.Client()

AREA = [31.45, 108.80, 30.60, 110.20]   # 奉节周边 [北,西,南,东]
RAW = "era5_fengjie_2025_raw.nc"
OUT = "era5_fengjie_2025.nc"

if os.path.exists(OUT) and os.path.getsize(OUT) > 1000:
    print("已存在，跳过:", OUT); raise SystemExit

req = {
    "product_type": "reanalysis",
    "variable": [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "surface_solar_radiation_downwards",
        "total_cloud_cover",
        "surface_pressure",
    ],
    "year": "2025",
    "month": ["04", "05", "06"],
    "day": [f"{d:02d}" for d in range(1, 31)],
    "time": [f"{h:02d}:00" for h in range(24)],
    "area": AREA,
    "format": "netcdf",
}
print("下载 2025-04~06 (含6月全月, 稍后裁剪) ->", RAW)
c.retrieve("reanalysis-era5-single-levels", req, RAW)

# 精确裁剪到 2025-04-01 .. 2025-06-16 23:00
import netCDF4, numpy as np
ds = netCDF4.Dataset(RAW, "r")
tvar = ds.variables["time"]
tunits = tvar.units  # 例如 hours since 1900-01-01 00:00:0.0
import datetime as dt
def num2dt(num, units):
    # 解析 "hours since YYYY-MM-DD ..."
    import re
    m = re.search(r"since ([\d\-: ]+)", units)
    base = dt.datetime.strptime(m.group(1).strip(), "%Y-%m-%d %H:%M:%S")
    return base + dt.timedelta(hours=float(num))
times = [num2dt(x, tunits) for x in tvar[:]]
idx = [i for i, tm in enumerate(times) if dt.datetime(2025,4,1) <= tm <= dt.datetime(2025,6,16,23,0)]
print(f"总时次 {len(times)}, 裁剪保留 {len(idx)} (2025-04-01..06-16)")
out = netCDF4.Dataset(OUT, "w")
for d in ds.dimensions:
    if d == "time":
        out.createDimension("time", len(idx))
    else:
        out.createDimension(d, len(ds.dimensions[d]))
for vn in ds.variables:
    v = ds.variables[vn]
    nv = out.createVariable(vn, v.dtype, v.dimensions)
    nv.setncatts({k: v.getncattr(k) for k in v.ncattrs()})
    if "time" in v.dimensions:
        nv[:] = v[ idx, ...]
    else:
        nv[:] = v[:]
out.close(); ds.close()
print("✅ 裁剪完成 ->", OUT)
print("变量:", list(netCDF4.Dataset(OUT).variables.keys()))
