# -*- coding: utf-8 -*-
"""s1_physics_elr_residual.py

常数 ELR 物理降尺度 + 实测残差空间插值修正的正式 LOBO 验证。
输出 JSON 报告、CSV 表格、诊断图。

用法:
    python s1_physics_elr_residual.py

输出:
    s1_physics_elr_residual/lobo_report.json
    s1_physics_elr_residual/fold_rmse.csv
    s1_physics_elr_residual/diagnostics.png
"""
import os, json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "s1_physics_elr_residual")
os.makedirs(OUT, exist_ok=True)

INV = os.path.join(ROOT, "03_桥梁清单与实测数据", "bridge_inventory_core.csv")
ERA5 = os.path.join(ROOT, "04_气象再分析_ERA5", "era5_v4", "era5_hourly.parquet")
LAB = os.path.join(ROOT, "03_桥梁清单与实测数据", "labels_v5_raw_hourly.parquet")
GRID_ELEV = os.path.join(HERE, "era5_grid_elevation_9km.npz")

R_EARTH = 6371.0


def haversine(lat1, lon1, lat2, lon2):
    """向量化的 haversine 距离 (km)。"""
    la1, lo1 = np.radians(lat1), np.radians(lon1)
    la2, lo2 = np.radians(lat2), np.radians(lon2)
    dla = la2 - la1
    dlo = lo2 - lo1
    a = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_data():
    inv = pd.read_csv(INV)
    mon = inv[inv.is_monitored == 1].copy().reset_index(drop=True)
    mon_bids = mon.bridge_id.tolist()
    n_mon = len(mon_bids)

    # 读取 ERA5-Land land_t2m
    era5 = pd.read_parquet(ERA5, columns=["bridge_id", "datetime", "land_t2m"])
    era5["datetime"] = pd.to_datetime(era5.datetime)
    era5 = era5[era5.bridge_id.isin(mon_bids)].copy()

    # 读取标签
    lab = pd.read_parquet(LAB)
    lab["time"] = pd.to_datetime(lab.time)

    # 共同时间
    etimes = pd.DatetimeIndex(sorted(era5.datetime.unique()))
    tidx = pd.Series(np.arange(len(etimes)), index=etimes)
    T = len(etimes)

    # 构建数组
    node = {b: i for i, b in enumerate(mon_bids)}
    E = np.full((n_mon, T), np.nan, dtype=np.float32)
    for b in mon_bids:
        sub = era5[era5.bridge_id == b].set_index("datetime").reindex(etimes)
        E[node[b], :] = sub["land_t2m"].to_numpy()

    Y = np.full((n_mon, T), np.nan, dtype=np.float32)
    for b in mon_bids:
        sub = lab[lab.bridge_id == b].set_index("time").reindex(etimes)
        Y[node[b], :] = sub["t_air_obs"].to_numpy()

    M = np.isfinite(Y)

    # 高程
    z = mon.elev_m.to_numpy(float)

    # 网格高程
    if os.path.exists(GRID_ELEV):
        grid = np.load(GRID_ELEV, allow_pickle=True)
        grid_map = {str(b): float(e) for b, e in zip(grid["bridge_id"], grid["grid_elev"])}
        z_grid = np.array([grid_map.get(b, np.nan) for b in mon_bids], float)
    else:
        z_grid = z.copy()  # fallback，无 ELR 修正
        print("[WARN] 未找到 ERA5 网格高程文件，ELR 修正退化为 0")

    # 距离矩阵
    D = haversine(mon.lat.values[:, None], mon.lon.values[:, None],
                  mon.lat.values[None, :], mon.lon.values[None, :])
    np.fill_diagonal(D, np.inf)

    regions = mon.region.tolist()
    return mon, mon_bids, E, Y, M, z, z_grid, D, regions, etimes


def physical_baseline(E, z, z_grid, gamma):
    """纯物理 ELR 降尺度。"""
    T_phys = np.full_like(E, np.nan)
    for i in range(E.shape[0]):
        T_phys[i, :] = E[i, :] + (gamma / 1000.0) * (z[i] - z_grid[i])
    return T_phys


def residual_interpolation(residual, target, train_idx, D, regions, mode="idw_region"):
    """
    对 target 桥做残差空间插值。
    residual: (n_mon, T)
    train_idx: 训练桥下标列表
    mode: "none", "global_mean", "idw_region", "nn_region"
    """
    T = residual.shape[1]
    eps_hat = np.zeros(T, dtype=np.float32)

    if mode == "none":
        return eps_hat

    if mode == "global_mean":
        for t in range(T):
            vals = residual[train_idx, t][np.isfinite(residual[train_idx, t])]
            eps_hat[t] = np.nanmean(vals) if len(vals) else 0.0
        return eps_hat

    # 同 region 训练桥
    same_region = [i for i in train_idx if regions[i] == regions[target]]

    if mode == "idw_region":
        for t in range(T):
            vals, weights = [], []
            for i in same_region:
                if np.isfinite(residual[i, t]):
                    vals.append(residual[i, t])
                    weights.append(1.0 / (D[target, i] ** 2 + 1.0))
            if len(vals) >= 1:
                eps_hat[t] = np.average(vals, weights=weights)
            else:
                vals = residual[train_idx, t][np.isfinite(residual[train_idx, t])]
                eps_hat[t] = np.nanmean(vals) if len(vals) else 0.0
        return eps_hat

    if mode == "nn_region":
        for t in range(T):
            best = None
            for i in same_region:
                if np.isfinite(residual[i, t]):
                    if best is None or D[target, i] < best[0]:
                        best = (D[target, i], residual[i, t])
            if best is not None:
                eps_hat[t] = best[1]
            else:
                vals = residual[train_idx, t][np.isfinite(residual[train_idx, t])]
                eps_hat[t] = np.nanmean(vals) if len(vals) else 0.0
        return eps_hat

    raise ValueError(f"Unknown mode: {mode}")


def lobo_cv(E, Y, M, z, z_grid, D, regions, gamma, residual_mode, etimes=None):
    """LOBO 交叉验证。"""
    n_mon = E.shape[0]
    T_phys = physical_baseline(E, z, z_grid, gamma)
    residual = Y - T_phys

    folds = {}
    preds = {}
    for h in range(n_mon):
        t_h = np.where(M[h])[0]
        if len(t_h) == 0:
            continue
        train_idx = [i for i in range(n_mon) if i != h]

        eps_hat = residual_interpolation(residual, h, train_idx, D, regions, mode=residual_mode)

        T_pred = T_phys[h, t_h] + eps_hat[t_h]
        obs = Y[h, t_h]

        rmse = float(np.sqrt(np.nanmean((T_pred - obs) ** 2)))
        mae = float(np.nanmean(np.abs(T_pred - obs)))
        bias = float(np.nanmean(T_pred - obs))
        corr = float(np.corrcoef(T_pred, obs)[0, 1]) if len(T_pred) > 1 else 0.0

        era5_rmse = float(np.sqrt(np.nanmean((E[h, t_h] - obs) ** 2)))
        phys_rmse = float(np.sqrt(np.nanmean((T_phys[h, t_h] - obs) ** 2)))

        folds[h] = {
            "n": int(len(t_h)),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "bias": round(bias, 4),
            "corr": round(corr, 4),
            "era5_rmse": round(era5_rmse, 4),
            "phys_rmse": round(phys_rmse, 4),
            "gain_vs_era5": round(100 * (era5_rmse - rmse) / era5_rmse, 2),
            "gain_vs_phys": round(100 * (phys_rmse - rmse) / phys_rmse, 2),
        }
        preds[h] = {"time": etimes[t_h] if etimes is not None else t_h,
                    "obs": obs, "pred": T_pred,
                    "era5": E[h, t_h], "phys": T_phys[h, t_h]}

    return folds, preds, T_phys, residual


def run_experiments():
    mon, mon_bids, E, Y, M, z, z_grid, D, regions, etimes = load_data()
    n_mon = len(mon_bids)

    report = {
        "meta": {
            "script": "s1_physics_elr_residual.py",
            "timestamp": pd.Timestamp.now().isoformat(),
            "n_monitored": n_mon,
            "monitored_bridges": mon_bids,
            "grid_elev": {b: float(z_grid[i]) for i, b in enumerate(mon_bids)},
            "bridge_elev": {b: float(z[i]) for i, b in enumerate(mon_bids)},
        },
        "experiments": []
    }

    best_exp = None
    best_rmse = np.inf

    for residual_mode in ["none", "global_mean", "idw_region", "nn_region"]:
        for gamma in [-9.0, -8.5, -8.0, -7.5, -7.0, -6.5, -6.0, -5.5, -5.0]:
            if residual_mode != "none" and gamma not in [-8.0, -7.0, -6.5, -6.0]:
                continue  # 残差模式只在几个 gamma 上跑
            folds, preds, T_phys, residual = lobo_cv(
                E, Y, M, z, z_grid, D, regions, gamma, residual_mode, etimes=etimes
            )

            # 用 mon_bids 替换 fold 键
            folds_named = {}
            for h, v in folds.items():
                v["bridge_id"] = mon_bids[h]
                folds_named[mon_bids[h]] = v

            rmse_vals = [v["rmse"] for v in folds_named.values()]
            mean_rmse = float(np.mean(rmse_vals))
            era5_vals = [v["era5_rmse"] for v in folds_named.values()]
            mean_era5 = float(np.mean(era5_vals))
            phys_vals = [v["phys_rmse"] for v in folds_named.values()]
            mean_phys = float(np.mean(phys_vals))

            exp = {
                "gamma": float(gamma),
                "residual_mode": residual_mode,
                "mean_rmse": round(mean_rmse, 4),
                "mean_era5_rmse": round(mean_era5, 4),
                "mean_phys_rmse": round(mean_phys, 4),
                "gain_vs_era5": round(100 * (mean_era5 - mean_rmse) / mean_era5, 2),
                "gain_vs_phys": round(100 * (mean_phys - mean_rmse) / mean_phys, 2),
                "folds": folds_named,
            }
            report["experiments"].append(exp)

            if residual_mode == "none" and mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_exp = exp

            print(f"[γ={gamma:+.1f}, mode={residual_mode}] mean RMSE={mean_rmse:.3f} | "
                  f"ERA5={mean_era5:.3f} | phys={mean_phys:.3f}")

    # 标记最优物理基线
    report["best_physical"] = {
        "gamma": best_exp["gamma"],
        "mean_rmse": best_exp["mean_rmse"],
        "mean_era5_rmse": best_exp["mean_era5_rmse"],
        "gain_vs_era5": best_exp["gain_vs_era5"],
        "folds": best_exp["folds"],
    }

    # 保存 JSON
    with open(os.path.join(OUT, "lobo_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 保存 CSV 表格
    rows = []
    for exp in report["experiments"]:
        for bid, fd in exp["folds"].items():
            rows.append({
                "gamma": exp["gamma"],
                "residual_mode": exp["residual_mode"],
                "bridge_id": bid,
                "region": mon[mon.bridge_id == bid].region.iloc[0],
                "n": fd["n"],
                "era5_rmse": fd["era5_rmse"],
                "phys_rmse": fd["phys_rmse"],
                "final_rmse": fd["rmse"],
                "mae": fd["mae"],
                "bias": fd["bias"],
                "corr": fd["corr"],
                "gain_vs_era5": fd["gain_vs_era5"],
                "gain_vs_phys": fd["gain_vs_phys"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "fold_rmse.csv"), index=False, encoding="utf-8-sig")

    # 绘制诊断图
    plot_diagnostics(report, mon, mon_bids, OUT)

    print(f"\n✓ 报告已保存: {OUT}")
    print(f"最佳物理 ELR: γ={best_exp['gamma']} K/km, 7桥平均 RMSE={best_exp['mean_rmse']:.3f} °C")
    return report


def plot_diagnostics(report, mon, mon_bids, out_dir):
    """绘制 LOBO RMSE 对比图。"""
    best = report["best_physical"]
    gamma_best = best["gamma"]

    # 提取各实验的 mean RMSE
    sweep = [(e["gamma"], e["mean_rmse"]) for e in report["experiments"] if e["residual_mode"] == "none"]
    sweep = sorted(sweep, key=lambda x: x[0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：gamma 敏感性
    ax = axes[0]
    gammas, rmses = zip(*sweep)
    ax.plot(gammas, rmses, 'o-', color='#2E86AB', linewidth=2, markersize=6)
    ax.axvline(gamma_best, color='red', linestyle='--', alpha=0.7, label=f'best γ={gamma_best}')
    ax.set_xlabel('ELR γ (K/km)', fontsize=12)
    ax.set_ylabel('7-bridge mean RMSE (°C)', fontsize=12)
    ax.set_title('ELR sensitivity (pure physics)', fontsize=13)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 右：各桥 ERA5 vs 物理 ELR
    ax = axes[1]
    bids = list(best["folds"].keys())
    x = np.arange(len(bids))
    width = 0.35
    era5_vals = [best["folds"][b]["era5_rmse"] for b in bids]
    phys_vals = [best["folds"][b]["phys_rmse"] for b in bids]
    ax.bar(x - width/2, era5_vals, width, label='ERA5-Land', color='#A23B72')
    ax.bar(x + width/2, phys_vals, width, label=f'ELR γ={gamma_best}', color='#2E86AB')
    ax.set_xticks(x)
    ax.set_xticklabels(bids, fontsize=10)
    ax.set_ylabel('RMSE (°C)', fontsize=12)
    ax.set_title(f'Per-bridge LOBO RMSE: ERA5 vs ELR', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "diagnostics.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("✓ diagnostics.png saved")


if __name__ == "__main__":
    report = run_experiments()
