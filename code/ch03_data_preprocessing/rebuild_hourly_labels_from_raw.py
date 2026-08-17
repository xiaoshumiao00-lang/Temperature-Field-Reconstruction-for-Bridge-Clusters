# -*- coding: utf-8 -*-
"""rebuild_hourly_labels_from_raw.py

从 中铁大桥气象数据 下 7 个桥梁原始 CSV 重建逐小时实测气温标签。
各桥采样间隔不同（1 min / 10 min / 1 h），时间窗也不同，本脚本按小时取平均，
并保留每小时原始样本数以供质量评估。

输出:
    labels_v6_raw_hourly.parquet
    labels_v6_summary.csv
"""
import os
import re
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(ROOT, "..", "..", "中铁大桥气象数据"))
OUT_DIR = os.path.join(ROOT, "..", "01_桥梁清单与标签")

# CSV 文件名前缀 → bridge_id 映射
NAME2BID = {
    "01贵南高铁桐子园双线特大桥": "M01",
    "02盘龙柳江大桥": "M02",
    "03郑万高铁大宁河大桥": "M03",
    "04郑万铁路梅溪河双线特大桥": "M04",
    "05大瑞铁路怒江特大桥": "M05",
    "06郑万铁路梅溪河桥梁": "M06",
    "07渝昆高铁洛泽河特大桥": "M07",
}


def find_raw_csvs(src_dir):
    """扫描 src_dir 下 7 个桥 CSV 文件。"""
    files = {}
    for name, bid in NAME2BID.items():
        path = os.path.join(src_dir, f"{name}.csv")
        if os.path.exists(path):
            files[bid] = path
        else:
            raise FileNotFoundError(f"未找到 {path}")
    return files


def read_and_resample(bid, path):
    """读取单个桥 CSV，并做小时平均。"""
    df = pd.read_csv(path, encoding="gbk")
    if "DateTime" not in df.columns:
        raise ValueError(f"{path}: 缺少 DateTime 列")

    # 解析时间
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")

    # 温度列：除 DateTime 外第一个数值列
    temp_cols = [c for c in df.columns if c != "DateTime" and df[c].dtype.kind in "fi"]
    if not temp_cols:
        # 尝试强制转换所有非时间列为数值
        for c in df.columns:
            if c == "DateTime":
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
        temp_cols = [c for c in df.columns if c != "DateTime" and df[c].dtype.kind in "fi"]
    if not temp_cols:
        raise ValueError(f"{path}: 未找到温度列")
    temp_col = temp_cols[0]

    # 只保留有效时间和温度
    df = df.dropna(subset=["DateTime", temp_col]).copy()
    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    df = df.dropna(subset=[temp_col])

    # 简单异常值过滤：剔除明显不合理的值（<-50 或 >70）
    df = df[(df[temp_col] > -50) & (df[temp_col] < 70)].copy()

    # 按小时重采样：取平均并统计原始样本数
    df = df.set_index("DateTime").sort_index()
    hourly = df[[temp_col]].resample("1h").agg({temp_col: "mean"})
    hourly["n_raw_samples"] = df[temp_col].resample("1h").count()
    hourly = hourly.rename(columns={temp_col: "t_air_obs"})
    hourly = hourly.reset_index()
    hourly = hourly.rename(columns={"DateTime": "time"})
    hourly["bridge_id"] = bid
    hourly = hourly[["bridge_id", "time", "t_air_obs", "n_raw_samples"]]
    return hourly


def main():
    files = find_raw_csvs(SRC_DIR)
    all_hourly = []
    summary_rows = []

    for bid, path in files.items():
        print(f"\n处理 {bid}: {os.path.basename(path)}")
        hourly = read_and_resample(bid, path)
        all_hourly.append(hourly)
        summary_rows.append({
            "bridge_id": bid,
            "file": os.path.basename(path),
            "time_start": hourly["time"].min(),
            "time_end": hourly["time"].max(),
            "n_hours": len(hourly),
            "n_hours_with_data": int((hourly["n_raw_samples"] > 0).sum()),
            "mean_samples_per_hour": float(hourly["n_raw_samples"].mean()),
            "max_gap_hours": int(hourly["time"].diff().dt.total_seconds().div(3600).max()),
            "t_min": float(hourly["t_air_obs"].min()),
            "t_max": float(hourly["t_air_obs"].max()),
        })
        print(f"  小时数: {len(hourly)}, 时间范围: {hourly['time'].min()} ~ {hourly['time'].max()}")
        print(f"  温度范围: {hourly['t_air_obs'].min():.2f} ~ {hourly['t_air_obs'].max():.2f} °C")

    labels = pd.concat(all_hourly, ignore_index=True)
    labels["time"] = pd.to_datetime(labels["time"])
    labels = labels.sort_values(["bridge_id", "time"]).reset_index(drop=True)

    out_path = os.path.join(OUT_DIR, "labels_v6_raw_hourly.parquet")
    labels.to_parquet(out_path, index=False)
    print(f"\n保存: {out_path}  ({len(labels)} 行)")

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "labels_v6_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"保存: {summary_path}")
    print("\n汇总:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
