# -*- coding: utf-8 -*-
"""validate_benchmark_sensors.py — Recompute the manuscript Section 5.1
validation statistics from the committed slim result file
``data/fem_benchmark_annual.npz`` + ``data/measured_temperatures_2024.csv``.

Writes (to repository ``tables/``):
  ch05_sensor_validation.csv            4 calibrated channels, long window
  ch05_sensor_validation_longwindow.csv all 14 structural channels, long window
  ch05_sensor_validation_sunny6day.csv  all 14 structural channels, sunny period
  ch05_daily_feature_stats.csv          daily max / min / range bias & MAE

Expected key numbers (manuscript Tables 5-6, Section 5.1):
  long window 2024-02-26..12-31, 4 calibrated channels pooled:
      RMSE = 1.119 °C, MAE = 0.811 °C, r = 0.991, N = 25471
  sunny period 2024-08-12..08-17, 4 calibrated channels pooled:
      RMSE = 0.977 °C, MAE = 0.719 °C, r = 0.825, N = 573
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(REPO_ROOT, "data")
TABLES = os.path.join(REPO_ROOT, "tables")

VAL0 = pd.Timestamp("2024-02-26")
VAL1 = pd.Timestamp("2024-12-31 23:00")
CLEAR0 = pd.Timestamp("2024-08-12")
CLEAR1 = pd.Timestamp("2024-08-17 23:00")
PICK4 = ['N101', 'N109', 'N102', 'N209']
STRUCT14 = ['N101', 'N102', 'N103', 'N107', 'N108', 'N109',
            'N204', 'N205', 'N206', 'N207', 'N208', 'N209', 'N210', 'N211']


def load():
    z = np.load(os.path.join(DATA, "fem_benchmark_annual.npz"), allow_pickle=True)
    times = pd.DatetimeIndex(z["times"])
    names = list(z["node_names"])
    Tn = z["T_year_nodes"].astype(float)
    ht = pd.read_csv(os.path.join(DATA, "measured_temperatures_2024.csv"))
    ht["DateTime"] = pd.to_datetime(ht["DateTime"])
    ht = ht.set_index("DateTime").sort_index()
    return times, names, Tn, ht


def channel_stats(names, Tn, ht, nm, t0w, t1w, times):
    i = names.index(nm)
    fv = Tn[:, i]
    obs = ht[nm].reindex(times).values
    m = np.isfinite(obs) & np.isfinite(fv) & (times >= t0w) & (times <= t1w)
    if m.sum() < 10:
        return None
    f, o = fv[m], obs[m]
    return dict(channel=nm,
                RMSE=round(float(np.sqrt(np.mean((f - o) ** 2))), 3),
                MAE=round(float(np.mean(np.abs(f - o))), 3),
                r=round(float(np.corrcoef(f, o)[0, 1]), 3), N=int(m.sum()))


def pooled_stats(names, Tn, ht, chs, t0w, t1w, times):
    fs, os_ = [], []
    for nm in chs:
        i = names.index(nm)
        fv = Tn[:, i]
        obs = ht[nm].reindex(times).values
        m = np.isfinite(obs) & np.isfinite(fv) & (times >= t0w) & (times <= t1w)
        fs.append(fv[m]); os_.append(obs[m])
    f = np.concatenate(fs); o = np.concatenate(os_)
    return dict(channel='pooled',
                RMSE=round(float(np.sqrt(np.mean((f - o) ** 2))), 3),
                MAE=round(float(np.mean(np.abs(f - o))), 3),
                r=round(float(np.corrcoef(f, o)[0, 1]), 3), N=int(len(f)))


def daily_features(names, Tn, ht, times):
    rows = []
    for nm in PICK4:
        i = names.index(nm)
        fv = Tn[:, i]
        obs = ht[nm].reindex(times).values
        m = np.isfinite(obs) & np.isfinite(fv) & (times >= VAL0) & (times <= VAL1)
        d = times[m].date
        f = fv[m]; o = obs[m]
        fd = pd.Series(f, index=d).groupby(level=0)
        od = pd.Series(o, index=d).groupby(level=0)
        for tag, a, b in [("daily_max", fd.max(), od.max()),
                          ("daily_min", fd.min(), od.min()),
                          ("daily_range", fd.max() - fd.min(),
                           od.max() - od.min())]:
            a = a.dropna(); b = b.reindex(a.index).dropna()
            rows.append(dict(channel=nm, metric=tag,
                             mean_bias_C=round(float((a - b).mean()), 3),
                             MAE_C=round(float(np.abs(a - b).mean()), 3),
                             N_days=int(len(a))))
    return pd.DataFrame(rows)


def main():
    times, names, Tn, ht = load()

    # long window, 14 channels
    rows = [r for r in (channel_stats(names, Tn, ht, nm, VAL0, VAL1, times)
                        for nm in STRUCT14) if r]
    rows.append(pooled_stats(names, Tn, ht, STRUCT14, VAL0, VAL1, times))
    df14 = pd.DataFrame(rows)
    df14.to_csv(os.path.join(TABLES, "ch05_sensor_validation_longwindow.csv"),
                index=False)

    # long window, 4-channel summary in manuscript Table 6 wording
    rows4 = [r for r in (channel_stats(names, Tn, ht, nm, VAL0, VAL1, times)
                         for nm in PICK4) if r]
    rows4.append(pooled_stats(names, Tn, ht, PICK4, VAL0, VAL1, times))
    t6 = pd.DataFrame(rows4)
    t6.loc[t6.channel == 'pooled', 'channel'] = '4 calibrated channels pooled'
    t6.to_csv(os.path.join(TABLES, "ch05_sensor_validation.csv"), index=False)

    # sunny 6-day window, 14 channels + 4-channel pooled
    rows = [r for r in (channel_stats(names, Tn, ht, nm, CLEAR0, CLEAR1, times)
                        for nm in STRUCT14) if r]
    rows.append(pooled_stats(names, Tn, ht, STRUCT14, CLEAR0, CLEAR1, times))
    rows.append(pooled_stats(names, Tn, ht, PICK4, CLEAR0, CLEAR1, times))
    df_sunny = pd.DataFrame(rows)
    df_sunny.loc[df_sunny.channel == 'pooled', 'channel'] = \
        ['14ch pooled', '4ch pooled']
    df_sunny.to_csv(
        os.path.join(TABLES, "ch05_sensor_validation_sunny6day.csv"), index=False)

    # daily feature statistics
    daily_features(names, Tn, ht, times).to_csv(
        os.path.join(TABLES, "ch05_daily_feature_stats.csv"), index=False)

    p4 = t6[t6.channel == '4 calibrated channels pooled'].iloc[0]
    print("long window  4ch pooled: RMSE=%.3f MAE=%.3f r=%.3f N=%d"
          % (p4.RMSE, p4.MAE, p4.r, p4.N))
    print("tables written to %s" % TABLES)


if __name__ == "__main__":
    main()
