# -*- coding: utf-8 -*-
"""S1-XGB: 气温的偏差型 XGBoost 降尺度 (ERA5 → 点位级, 文献一致推荐方案)。

原理 (统计降尺度统计, verified by 2026-08-01 literature survey):
  dT = t_air_obs − ERA5_t2m (残差/偏差)
  dT = f( ERA5气象 + 地形协变量 + 微尺度过程特征 )   ← XGBoost 学这个
  T_corr = ERA5_t2m + dT_hat

  关键文献: Zhangjiakou XGBoost (doi:10.1007/s00382-026-08253-6)
  偏差型建模预测 ERA5−站点偏差, 显著优于直接预测温度;
  加入冷池(CAP)/夜间增温/辐射冷却特征后提升明显: 峰值 RMSE 降 70%.

特征 (全部 207 桥可得的共变量, 不在桥位特异的值):
  地形静态: 高程 + kNN 邻域高程统计(TPI/起伏度)
  气候静态: MODIS 气候态 8 维
  时变气象: ERA5 t2m/rh/ws/cc/ssrd/sp/land_t2m/land_rh
  时变微尺度: 夜间×晴空×静风(CAP-prone), 谷地×夜间(冷池), 地表-大气温差(稳定度)
  + 日-季时间编码

用法: python s1_downscale_xgb.py
输出: s1_downscale_xgb/   (lobo_report.json + corrected_hourly.npz)
"""
import json, os, time
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial.distance import cdist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.environ.get("XGB_OUT") or os.path.join(HERE, "s1_downscale_xgb")
os.makedirs(OUT, exist_ok=True)
DEM_NPZ = os.path.join(HERE, "s1_dem_terrain.npz")

INV = os.path.join(ROOT, "03_桥梁清单与实测数据", "bridge_inventory_core.csv")
ERA5_PARQ = os.path.join(ROOT, "04_气象再分析_ERA5", "era5_v4", "era5_hourly.parquet")
LAB_AIR = os.path.join(ROOT, "03_桥梁清单与实测数据", "labels_v4", "labels_air_hourly.parquet")
FEAT_NPZ = os.path.join(HERE, "s1_features_core", "features.npz")
MODIS_CLIM = os.path.join(ROOT, "05_卫星遥感_MODIS_LST", "modis_v4", "clim_features_graph.parquet")
R_EARTH = 6371.0

# --- 地形特征: 207桥自身+邻域高程统计 (免DEM, 即时可用) ---
def terrain_features(inv, bids, nids, K_NBR=10):
    """每桥: 高程 + kNN邻域高程统计(TPI/起伏度/粗糙度).
    N=207, 距高矩阵复用 bias_correction 方式。"""
    N = len(inv)
    lat = inv.lat.to_numpy(); lon = inv.lon.to_numpy()
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[None,:]-la[:,None]; dlo = lo[None,:]-lo[:,None]
    a = np.sin(dla/2)**2 + np.cos(la)[:,None]*np.cos(la)[None,:]*np.sin(dlo/2)**2
    D = 2*R_EARTH*np.arcsin(np.sqrt(np.clip(a,0,1)))   # [N,N] km
    np.fill_diagonal(D, np.inf)
    elev = inv.elev_m.to_numpy(float)
    nbr_stats = np.zeros((N, 5))
    for i in range(N):
        top = np.argsort(D[i])[:K_NBR]
        e = elev[top]
        nbr_stats[i] = [e.mean(), e.std(), e.min(), e.max(), e.max()-e.min()]
    TPI = elev - nbr_stats[:, 0]    # 高→山脊,低→谷
    terrain = np.column_stack([elev, TPI, nbr_stats[:,1], nbr_stats[:,4], nbr_stats[:,2], nbr_stats[:,3]])
    return terrain.astype(np.float32)


def main():
    t0_total = time.time()
    # ========== 1. 数据加载 ==========
    inv = pd.read_csv(INV).sort_values("node_idx").reset_index(drop=True)
    nids = inv.node_id.astype(str).tolist()
    bids = inv.bridge_id.astype(str).tolist()
    any2i = {}; [any2i.setdefault(k,i) for i,(n,b) in enumerate(zip(nids,bids)) for k in (n,b)]
    N = len(inv); mon_mask = inv.is_monitored==1
    mon_idx = np.where(mon_mask)[0]
    mon_bids = inv.bridge_id[mon_mask].tolist()
    L = len(mon_bids)
    print("节点 %d  监测 %d: %s"%(N,L,mon_bids))

    # 加载全量 ERA5 (t2m等, 只取需用的列以省内存)
    era5 = pd.read_parquet(ERA5_PARQ, columns=["bridge_id","datetime",
        "t2m","rh","ws","cc","ssrd","sp","land_t2m","land_rh"])
    era5["bridge_id"] = era5["bridge_id"].astype(str)
    era5["datetime"] = pd.to_datetime(era5.datetime)
    times = pd.DatetimeIndex(sorted(era5.datetime.unique()))
    T = len(times); tidx = pd.Series(np.arange(T), index=times)
    print("时间 %d h  %s ~ %s"%(T, times[0], times[-1]))

    # ERA5 逐时 [N,T,d]  (只取需用的 8 通道)
    era5_cols = ["t2m","rh","ws","cc","ssrd","sp","land_t2m","land_rh"]
    EC = np.full((N,T,len(era5_cols)), np.nan, np.float32)
    era5["__i"] = era5.bridge_id.map(any2i)
    era5 = era5.dropna(subset=["__i"])
    ii = era5.__i.to_numpy(int)
    ti = tidx.reindex(era5.datetime).to_numpy()
    ok = np.isfinite(ti.astype(float))
    for ci, c in enumerate(era5_cols):
        EC[ii[ok].astype(int), ti[ok].astype(int), ci] = era5[c].to_numpy(np.float32)[ok]
    del era5

    # MODIS 气候 (bridge_id 是索引, 需 reset_index)
    cl = pd.read_parquet(MODIS_CLIM).reset_index()
    cl["bridge_id"] = cl["bridge_id"].astype(str); cl["__i"] = cl.bridge_id.map(any2i)
    cl = cl.dropna(subset=["__i"]).astype({"__i": int}).set_index("__i").reindex(range(N))
    clim_cols = ["clim_lst_day_mean", "clim_lst_night_mean", "clim_dtr_mean",
                 "clim_lst_day_std", "clim_clear_day", "clim_summer_day", "clim_winter_day"]
    X_clim = np.column_stack([np.asarray(cl[c], float) for c in clim_cols if c in cl.columns])
    for j in range(X_clim.shape[1]):   # 缺测用全域均值
        bad = ~np.isfinite(X_clim[:,j]); X_clim[bad,j] = np.nanmean(X_clim[~bad,j])

    # 桥梁静态
    span_log = np.log1p(inv.struct_len_m.fillna(inv.struct_len_m.median()).to_numpy(float))
    is_rail = (inv.kind.astype(str)=="railway").to_numpy(float)
    reg_code = inv.region.astype("category").cat.codes.to_numpy(int)

    # 地形特征 (从 DEM 栅格采样, 9维: 见 s1_compute_dem_terrain.py)
    if os.path.exists(DEM_NPZ):
        terr = np.load(DEM_NPZ)["terrain"]    # [N,9]
        note_dem = "DEM栅格采样9维"
    else:
        terr = terrain_features(inv, bids, nids)     # 回退用桥邻域
        note_dem = "桥邻域高程统计(无DEM栅格)"
    print("地形特征: %s (%d维)"%(note_dem, terr.shape[1]))

    # ---- 组装静态特征矩阵 [N, D_s] ----
    X_static_list = [terr,                                      # 6
                     reg_code[:,None],
                     span_log[:,None], is_rail[:,None],         # 2
                     X_clim]                                    # 8
    X_static = np.concatenate(X_static_list, axis=1).astype(np.float32)
    D_s = X_static.shape[1]
    print("X_static %s  (地形%d+区1+桥2+气候%d)"%(X_static.shape, terr.shape[1], X_clim.shape[1]))

    # 实测标签 (7 监测桥)
    lab = pd.read_parquet(LAB_AIR)
    lab["bridge_id"] = lab["bridge_id"].astype(str)
    lab["time"] = pd.to_datetime(lab.time)
    # 每个锚点的残差序列 [L, T] — 改为 land_t2m 做基础场 (②)
    mon_node = np.array([any2i[b] for b in mon_bids])
    Yr = np.full((L,T), np.nan, np.float32)
    Ym = np.zeros((L,T), bool)
    for k, b in enumerate(mon_bids):
        lb = lab[lab.bridge_id==b]
        li = tidx.reindex(lb.time).to_numpy()
        lov = np.isfinite(li.astype(float))
        Ym[k, li[lov].astype(int)] = True
        e_base = EC[mon_node[k], :, 6]                     # ERA5-Land land_t2m  (col 6)
        Yr[k, li[lov].astype(int)] = lb.t_air_obs.to_numpy(float)[lov] - e_base[li[lov].astype(int)]

    # ========== 2. 特征工程 — ②ELR递减率 + ③GNN-LST ==========
    # ELR: 每时刻从监测桥实测气温+高程线性拟合 (K/km)
    mon_elev = inv.elev_m.to_numpy(float)[mon_node]   # [L]
    T_obs_all = np.full((L, T), np.nan, np.float32)
    for k in range(L):
        T_obs_all[k, Ym[k]] = Yr[k, Ym[k]] + EC[mon_node[k], Ym[k], 6]  # T_obs = dT + land_t2m
    ELR_arr = np.full(T, np.nan, np.float32)
    for t in range(T):
        ok = np.isfinite(T_obs_all[:, t])
        if ok.sum() >= 3:
            elr = np.polyfit(mon_elev[ok], T_obs_all[ok, t], 1)[0]  # K/m
            ELR_arr[t] = elr * 1000   # K/km
    # 插值补缺
    ELR_arr = pd.Series(ELR_arr).interpolate(limit_direction='both').fillna(-6.5).to_numpy(np.float32)

    # MODIS LST (③): 读原始日值, 落到逐时轴代表时次 (day→13h, night→1h)
    md = pd.read_parquet(os.path.join(ROOT, "05_卫星遥感_MODIS_LST", "modis_v4",
                                       "modis_305_daily.parquet"))
    md["bridge_id"] = md["bridge_id"].astype(str); md["date"] = pd.to_datetime(md.date).dt.normalize()
    md["__i"] = md.bridge_id.map(any2i); md = md.dropna(subset=["__i"])
    mi = md.__i.to_numpy(int); times_norm = times.normalize()
    didx = pd.Series(np.arange(len(pd.DatetimeIndex(sorted(times_norm.unique())))),
                     index=sorted(times_norm.unique()))
    day_of = didx.reindex(times_norm).to_numpy()
    LST_day = np.full((N, T), np.nan, np.float32)
    LST_night = np.full((N, T), np.nan, np.float32)
    for k in range(len(md)):
        i, d = mi[k], md.date.iloc[k]
        j13 = np.where((day_of == didx[d]) & (times.hour == 13))[0]
        j01 = np.where((day_of == didx[d]) & (times.hour == 1))[0]
        if len(j13) and np.isfinite(md.lst_day.iloc[k]):
            LST_day[i, j13[0]] = md.lst_day.iloc[k]
        if len(j01) and np.isfinite(md.lst_night.iloc[k]):
            LST_night[i, j01[0]] = md.lst_night.iloc[k]
    # 标准化 (全域均值 + 桥邻域插值)
    for arr_name in ['LST_day', 'LST_night']:
        arr = locals()[arr_name]
        mu, sd = np.nanmean(arr), np.nanstd(arr) + 1e-9
        arr = (arr - mu) / sd
        for i in range(N):
            s = pd.Series(arr[i]).interpolate(limit_direction='both').fillna(0).to_numpy(np.float32)
            arr[i] = s
        locals()[arr_name] = arr

    # X_time: era5(8) + micro(3) + time(4) + ELR(2) + LST(2) = 19
    D_t = 8 + 3 + 4 + 2 + 2
    X_time = np.zeros((N, T, D_t), np.float32)
    t2m = EC[:,:,0]; ws_e = EC[:,:,2]; cc_e = EC[:,:,3]
    ssrd_e = EC[:,:,4]; land_t2m = EC[:,:,6]
    X_time[:,:,:8] = EC
    # 微尺度 (3)
    night = (ssrd_e < 5.0).astype(float)
    cap = night * (cc_e < 30.0).astype(float) * (ws_e < 3.0).astype(float)
    X_time[:,:,8] = cap
    tpi_idx = 4 if terr.shape[1] >= 9 else 1
    tpi = X_static[:, tpi_idx:tpi_idx + 1]
    X_time[:,:,9] = cap * (tpi < 0).astype(float)
    X_time[:,:,10] = np.where(np.isfinite(land_t2m - t2m), land_t2m - t2m, 0.0)
    # 时间编码 (4)
    hour_a = times.hour.to_numpy(float); doy_a = times.dayofyear.to_numpy(float)
    X_time[:,:,11] = np.sin(2*np.pi*hour_a/24)
    X_time[:,:,12] = np.cos(2*np.pi*hour_a/24)
    X_time[:,:,13] = np.sin(2*np.pi*doy_a/365.25)
    X_time[:,:,14] = np.cos(2*np.pi*doy_a/365.25)
    # ② ELR (2): 共享k-1时刻的时变递减率 + 高程×ELR
    for i in range(N):
        X_time[i, :, 15] = ELR_arr
        X_time[i, :, 16] = ELR_arr * X_static[i, 0] / 1000.0   # elev*ELR/1000 = ℃ correction
    # ③ MODIS LST (2): 标准化后逐日值
    X_time[:,:,17] = LST_day; X_time[:,:,18] = LST_night
    print("X_time %s  (8 ERA5 + 3 micro + 4 time + 2 ELR + 2 LST)" % (X_time.shape,))

    # ========== 3. XGBoost 偏差训练 (昼夜分离) + LOBO 验证 ==========
    SEED = 0
    folds = {}
    PARAMS = dict(max_depth=5, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, objective="reg:squarederror",
                  eval_metric="rmse", nthread=-1, seed=SEED)
    def train_one(Xtr, ytr, Xva, yva, n_rounds=300, early=30):
        ok_tr = np.isfinite(ytr); ok_va = np.isfinite(yva)
        dt = xgb.DMatrix(Xtr[ok_tr], ytr[ok_tr]); dv = xgb.DMatrix(Xva[ok_va], yva[ok_va])
        return xgb.train(PARAMS, dt, num_boost_round=n_rounds, evals=[(dv,"v")],
                         early_stopping_rounds=early, verbose_eval=False)

    for h in range(L):
        hb = mon_bids[h]; ho = mon_node[h]
        t_ho = np.where(Ym[h])[0]
        if len(t_ho) < 50: continue
        tr_bridges = [k for k in range(L) if k!=h]
        all_t = np.concatenate([np.where(Ym[k])[0] for k in tr_bridges])
        rng = np.random.RandomState(SEED+h)
        udays = np.unique(all_t // 24); nvd = max(1, int(0.15*len(udays)))
        vd = set(rng.choice(udays, nvd, replace=False).tolist())
        Xtr_all=[]; ytr_all=[]; Xva_all=[]; yva_all=[]
        for k in tr_bridges:
            no = mon_node[k]
            for j in np.where(Ym[k])[0]:
                feat = np.hstack([X_static[no], X_time[no, j]])
                if (j//24) in vd:
                    Xva_all.append(feat); yva_all.append(Yr[k, j])
                else:
                    Xtr_all.append(feat); ytr_all.append(Yr[k, j])
        Xtr=np.array(Xtr_all,np.float32); ytr=np.array(ytr_all,np.float32)
        Xva=np.array(Xva_all,np.float32); yva=np.array(yva_all,np.float32)
        bst = train_one(Xtr, ytr, Xva, yva)
        n_eval = len(t_ho)
        X_ho = np.hstack([np.tile(X_static[ho], (n_eval, 1)), X_time[ho, t_ho]])
        dT_hat = bst.predict(xgb.DMatrix(X_ho))
        # ---- 轻量偏差校正: 训练桥的均值残差纠正系统偏移 ----
        dT_pred_tr = []; dT_obs_tr = []
        for k in tr_bridges:
            no_k = mon_node[k]; ok_k = np.where(Ym[k])[0]
            fk = np.hstack([np.tile(X_static[no_k], (len(ok_k), 1)), X_time[no_k, ok_k]])
            dT_pred_tr.append(bst.predict(xgb.DMatrix(fk)))
            dT_obs_tr.append(Yr[k, ok_k])
        dT_pred_tr = np.concatenate(dT_pred_tr); dT_obs_tr = np.concatenate(dT_obs_tr)
        mean_bias = np.nanmean(dT_pred_tr - dT_obs_tr)  # 正=预测偏暖
        dT_hat_corr = dT_hat - mean_bias
        era5_ho = EC[ho, t_ho, 6]
        T_corr = era5_ho + dT_hat_corr
        T_corr_raw = era5_ho + dT_hat
        obs_ho = Yr[h, t_ho] + era5_ho
        rmse_xgb = float(np.sqrt(np.nanmean((T_corr - obs_ho)**2)))
        rmse_raw = float(np.sqrt(np.nanmean((T_corr_raw - obs_ho)**2)))
        rmse_era5 = float(np.sqrt(np.nanmean((era5_ho - obs_ho)**2)))
        folds[hb] = dict(n=n_eval, xgb_rmse=rmse_xgb, era5_rmse=rmse_era5,
                         gain=100*(rmse_era5-rmse_xgb)/max(rmse_era5,1e-9),
                         gain_raw=100*(rmse_era5-rmse_raw)/max(rmse_era5,1e-9),
                         rmse_raw=rmse_raw,
                         bias=float(np.nanmean(T_corr-obs_ho)))
        print("  %-4s n=%5d | XGB+bias %.3f | raw %.3f | ERA5 %.3f | gain %+5.1f%%"
              %(hb, n_eval, rmse_xgb, rmse_raw, rmse_era5, folds[hb]["gain"]))

    print("\n===== LOBO 汇总 =====")
    xgb_r = np.mean([v["xgb_rmse"] for v in folds.values()])
    e5_r  = np.mean([v["era5_rmse"] for v in folds.values()])
    print("XGB %.3f   ERA5 %.3f   gain %+.1f%%"%(xgb_r, e5_r, 100*(e5_r-xgb_r)/e5_r))

    # ========== 4. 部署: 全 7 桥训练 → 207 桥预测 ==========
    Xall = []; yall = []
    for k in range(L):
        no = mon_node[k]
        for j in np.where(Ym[k])[0]:
            Xall.append(np.hstack([X_static[no], X_time[no, j]])); yall.append(Yr[k, j])
    Xall=np.array(Xall,np.float32); yall=np.array(yall,np.float32)
    ok=np.isfinite(yall); Xall=Xall[ok]; yall=yall[ok]
    # 用小验证集给早停
    bst_full = train_one(Xall, yall, Xall[:min(500,len(Xall))], yall[:min(500,len(yall))])
    dT_full = np.full((N,T), np.nan, np.float32)
    for i in range(N):
        feat = np.hstack([np.tile(X_static[i], (T, 1)), X_time[i]])
        dT_full[i] = bst_full.predict(xgb.DMatrix(feat))
    t2m_c = EC[:,:,6] + dT_full

    # 保存
    deploys = dict(t2m_c=t2m_c.astype(np.float32),
                   time_index=np.array([t.isoformat() for t in times]),
                   bridge_ids=np.array(bids), node_ids=np.array(nids))
    np.savez_compressed(os.path.join(OUT, "corrected_hourly.npz"), **deploys)

    report = dict(method="XGBoost bias-model (+wd/diff/dewpoint)", config=PARAMS,
                  total_sec=round(time.time()-t0_total,1), LOBO=folds)
    with open(os.path.join(OUT, "lobo_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nsaved -> %s  (%.0fs)"%(OUT, time.time()-t0_total))


if __name__=="__main__":
    main()
