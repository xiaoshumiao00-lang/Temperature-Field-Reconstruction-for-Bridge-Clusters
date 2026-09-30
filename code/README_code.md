# Code guide

Scripts are grouped by the manuscript chapter they belong to. All scripts are written in Python 3.10+.

## `ch03_data_preprocessing/`

| Script | Purpose |
|--------|---------|
| `rebuild_hourly_labels_from_raw.py` | Reconstruct quality-controlled hourly air-temperature labels from the 7 monitored bridges' raw CSV loggers. |
| `download_era5_2025.py` | Download ERA5-Land hourly variables over the study area via the CDS API. |
| `modis_lst_extract_per_bridge.py` | Extract clear-sky MODIS LST per bridge after QC filtering. |

## `ch04_downscaling/`

| Script | Purpose |
|--------|---------|
| `s1_physics_elr_residual.py` | Physics-based air-temperature downscaling with constant environmental lapse rate (ELR) and residual spatial interpolation. |
| `s1_downscale_xgb.py` | XGBoost residual-correction model using ERA5-Land, MODIS LST, DEM, and temporal features. |
| `s1_plot_results.py` | Generate downscaling validation figures (scatter, RMSE bars, residual maps). |
| `s1_plot_m03_full_year_en.py` | Reproduce the M03 full-year observed-vs-downscaled temperature time-series figure. |

## `ch05_fem_cluster/`

| Script | Purpose |
|--------|---------|
| `fsa_fast.py` | Vectorised explicit FVM solver (30 s time step) and per-node solar-incidence computation; numerically equivalent to the original loop implementation used for the paper. |
| `run_benchmark_annual.py` | **Reproduce the Section 5.1 annual benchmark run**: loads the committed 10 mm mesh (`data/benchmark_mesh.npz`), the 2024 boundary CSV and the measured N212 cavity temperature, runs the calibrated CEBM configuration (Table 5 of the manuscript) and writes the full annual field `fem_benchmark_annual_full.npz` plus the slim derivative `fem_benchmark_annual.npz`. `--tables` also recomputes the 14-channel validation tables. |
| `validate_benchmark_sensors.py` | Recompute the Section 5.1 validation statistics (sunny 6-day window 2024-08-12..17 and long window 2024-02-26..12-31) from the committed slim result file; writes the `ch05_sensor_validation*.csv` and `ch05_daily_feature_stats.csv` tables. |
| `plot_benchmark_figures.py` | Regenerate the three core Section 5.1 figures (sunny-day 6-panel temperature fields, annual 4-channel comparison with clear-period inset, depth temperature profiles) from the committed slim result file. |
| `fem_box_core.py` | Core 2-D transient heat-conduction FEM used for the 207-bridge cluster: solar geometry, ray-tracing self-shading, finite-volume matrix assembly, and Monte-Carlo uncertainty propagation. |
| `dxf_parser.py` | Parse a DXF/DWG-derived JSON geometry into the FEM mesh format used by `fem_box_core.py`. |
| `draw_dwg.py` | Helper for drawing and saving the benchmark cross-section geometry. |
| `streamlit_app.py` | Interactive web app for single-bridge and small-cluster temperature-field calculation. |
| `batch_cluster.py` | Batch FEM computation for the full 207-bridge cluster using downscaled boundary conditions. |
| `plot_cluster_figures.py` | Spatial-distribution and uncertainty figures for the 207-bridge cluster. |
| `plot_end_to_end_figure.py` | End-to-end workflow diagram for the web application. |

## Reproducing the Section 5.1 benchmark results

```bash
cd code/ch05_fem_cluster
python validate_benchmark_sensors.py     # -> tables, key numbers:
                                          #    long-window 4ch pooled
                                          #    RMSE = 1.119 C, MAE = 0.811 C,
                                          #    r = 0.991, N = 25471
python plot_benchmark_figures.py         # -> figures/ch05_fem_cluster/*.png
python run_benchmark_annual.py --tables  # full annual re-run (long!)
```

## Notes

- Most scripts originally used absolute paths on the author's workstation. The four benchmark scripts above are self-contained and resolve paths relative to the repository root; the remaining scripts may still need the constants at the top of each file (e.g., `ROOT`, `DATA_DIR`) pointed at your local copy.
- The large feature tensors (`features.npz`) and raw NetCDF/HDF inputs are not tracked in Git. See `data/README_data.md` for download instructions.
