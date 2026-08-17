# -*- coding: utf-8 -*-
"""s1_plot_results.py

绘制气温降尺度结果可视化（论文级）。
1. 重新训练 4 桥（M01/M03/M04/M07）单 XGBoost 偏差订正模型，保存预测序列；
2. 绘制时间序列对比图、散点图、RMSE 柱状图、残差分布图。

输出: 数据包_气温降尺度/07_示例加载脚本/s1_plots/
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# 中文字体
for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
    try:
        rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        continue
rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "s1_plots")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, HERE)
from s1_xgb_screen_v6 import load_data, build_features

# 物理 ELR 单桥 RMSE（来自 s1_physics_elr_residual，γ=-8.0）
PHYS_ELR_RMSE = {"M01": 1.58, "M03": 3.37, "M04": 2.04, "M07": 2.24}

PARAMS = dict(max_depth=6, learning_rate=0.03, n_estimators=500,
              subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
              eval_metric="rmse", n_jobs=-1, random_state=0, reg_lambda=1.0,
              min_child_weight=3)


def train_and_predict(data):
    """LOBO 训练，返回每桥的预测序列。"""
    L = data["L"]
    times = data["times"]
    preds = {}
    for h in range(L):
        bid = data["mon_bids"][h]
        tr = [i for i in range(L) if i != h]
        Xtr, ytr = [], []
        for i in tr:
            Mi = data["M"][i]
            Xi = build_features(data, i)
            Ri = data["land_t2m"][i] - data["Y"][i]
            Xtr.append(Xi[Mi]); ytr.append(Ri[Mi])
        Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
        Mh = data["M"][h]
        Xte = build_features(data, h)[Mh]
        land = data["land_t2m"][h, Mh]
        obs = data["Y"][h, Mh]
        model = xgb.XGBRegressor(**PARAMS)
        model.fit(Xtr, ytr)
        Rpred = model.predict(Xte).astype(np.float32)
        tpred = land - Rpred
        preds[bid] = {
            "time": times[Mh],
            "obs": obs,
            "pred": tpred,
            "era5": land,
            "residual": Rpred,
        }
    return preds


def plot_timeseries(preds, out_dir):
    """2x2 时间序列对比（每桥选 15 天代表性时段）。"""
    bids = ["M01", "M03", "M04", "M07"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    for ax, bid in zip(axes.ravel(), bids):
        p = preds[bid]
        t = pd.DatetimeIndex(p["time"])
        # 取数据中段 15 天
        mid = t[len(t) // 2]
        start = mid - pd.Timedelta(days=7.5)
        end = mid + pd.Timedelta(days=7.5)
        mask = (t >= start) & (t <= end)
        tt = t[mask]
        ax.plot(tt, p["obs"][mask], "k-", lw=1.4, label="实测 (观测)")
        ax.plot(tt, p["era5"][mask], "#2E86AB", lw=1.0, alpha=0.7, ls="--", label="ERA5-Land 基线")
        ax.plot(tt, p["pred"][mask], "#E63946", lw=1.2, label="XGBoost 降尺度")
        ax.set_title(f"{bid}", fontsize=13, fontweight="bold")
        ax.set_ylabel("气温 (°C)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle("气温降尺度时间序列对比（XGBoost 偏差订正）", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "fig_timeseries.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_timeseries.png")


def plot_scatter(preds, out_dir):
    """2x2 散点图（实测 vs 预测）。"""
    bids = ["M01", "M03", "M04", "M07"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, bid in zip(axes.ravel(), bids):
        p = preds[bid]
        obs, pred = p["obs"], p["pred"]
        rmse = float(np.sqrt(mean_squared_error(obs, pred)))
        r = float(np.corrcoef(obs, pred)[0, 1])
        ax.scatter(obs, pred, s=6, alpha=0.35, color="#457B9D", edgecolors="none")
        lo = min(obs.min(), pred.min()); hi = max(obs.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.6)
        ax.set_xlabel("实测气温 (°C)", fontsize=11)
        ax.set_ylabel("预测气温 (°C)", fontsize=11)
        ax.set_title(f"{bid}  RMSE={rmse:.2f}°C  r={r:.3f}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle("实测 vs XGBoost 预测散点图", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "fig_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_scatter.png")


def plot_rmse_bar(preds, out_dir):
    """RMSE 柱状图：XGBoost vs ERA5-Land vs 物理 ELR。"""
    bids = ["M01", "M03", "M04", "M07"]
    xgb_rmse = [float(np.sqrt(mean_squared_error(preds[b]["obs"], preds[b]["pred"]))) for b in bids]
    era5_rmse = [float(np.sqrt(mean_squared_error(preds[b]["obs"], preds[b]["era5"]))) for b in bids]
    phys_rmse = [PHYS_ELR_RMSE[b] for b in bids]

    x = np.arange(len(bids))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(x - w, era5_rmse, w, label="ERA5-Land", color="#A8DADC")
    ax.bar(x, phys_rmse, w, label="物理 ELR (γ=-8.0)", color="#457B9D")
    ax.bar(x + w, xgb_rmse, w, label="XGBoost 降尺度", color="#E63946")
    ax.set_xticks(x)
    ax.set_xticklabels(bids, fontsize=12)
    ax.set_ylabel("LOBO RMSE (°C)", fontsize=12)
    ax.set_title("各桥 LOBO RMSE 对比", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    for i in range(len(bids)):
        ax.text(x[i] - w, era5_rmse[i] + 0.08, f"{era5_rmse[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i], phys_rmse[i] + 0.08, f"{phys_rmse[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + w, xgb_rmse[i] + 0.08, f"{xgb_rmse[i]:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_rmse_bar.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_rmse_bar.png")


def plot_residual(preds, out_dir):
    """残差分布直方图（XGBoost 预测 - 实测）。"""
    bids = ["M01", "M03", "M04", "M07"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, bid in zip(axes.ravel(), bids):
        p = preds[bid]
        res = p["pred"] - p["obs"]
        ax.hist(res, bins=50, color="#457B9D", alpha=0.7, edgecolor="white")
        mu, sd = res.mean(), res.std()
        ax.axvline(0, color="k", lw=1.2, ls="--")
        ax.axvline(mu, color="#E63946", lw=1.5, label=f"均值 {mu:.2f}°C")
        ax.set_xlabel("预测误差 (°C)", fontsize=11)
        ax.set_ylabel("样本数", fontsize=11)
        ax.set_title(f"{bid}  标准差 σ={sd:.2f}°C", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("XGBoost 预测误差分布", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(out_dir, "fig_residual.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✓ fig_residual.png")


if __name__ == "__main__":
    data = load_data(["M05", "M06", "M02"], clean=True)
    preds = train_and_predict(data)

    # 保存预测序列为 npz
    np.savez(os.path.join(OUT, "predictions.npz"),
             **{b: np.array([preds[b]["time"].astype("int64"), preds[b]["obs"],
                             preds[b]["pred"], preds[b]["era5"]]) for b in preds})

    # 汇总指标
    summary = {}
    for b in preds:
        summary[b] = {
            "rmse": float(np.sqrt(mean_squared_error(preds[b]["obs"], preds[b]["pred"]))),
            "era5_rmse": float(np.sqrt(mean_squared_error(preds[b]["obs"], preds[b]["era5"]))),
            "mae": float(np.mean(np.abs(preds[b]["pred"] - preds[b]["obs"]))),
            "bias": float(np.mean(preds[b]["pred"] - preds[b]["obs"])),
            "corr": float(np.corrcoef(preds[b]["obs"], preds[b]["pred"])[0, 1]),
        }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    mean_rmse = float(np.mean([v["rmse"] for v in summary.values()]))
    print(f"\n4 桥平均 RMSE = {mean_rmse:.3f} °C")
    for b, v in summary.items():
        print(f"  {b}: RMSE={v['rmse']:.3f}, ERA5={v['era5_rmse']:.3f}, bias={v['bias']:.3f}, corr={v['corr']:.3f}")

    plot_timeseries(preds, OUT)
    plot_scatter(preds, OUT)
    plot_rmse_bar(preds, OUT)
    plot_residual(preds, OUT)
    print(f"\n图表已保存到: {OUT}")
