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
| `fem_box_core.py` | Core 2-D transient heat-conduction FEM: solar geometry, ray-tracing self-shading, finite-volume matrix assembly, and Monte-Carlo uncertainty propagation. |
| `dxf_parser.py` | Parse a DXF/DWG-derived JSON geometry into the FEM mesh format used by `fem_box_core.py`. |
| `draw_dwg.py` | Helper for drawing and saving the benchmark cross-section geometry. |
| `fem_thermal_7days.py` | Run the reference 7-day FEM for the benchmark arch bridge. |
| `streamlit_app.py` | Interactive web app for single-bridge and small-cluster temperature-field calculation. |
| `batch_cluster.py` | Batch FEM computation for the full 207-bridge cluster using downscaled boundary conditions. |
| `plot_cluster_figures.py` | Spatial-distribution and uncertainty figures for the 207-bridge cluster. |
| `plot_end_to_end_figure.py` | End-to-end workflow diagram for the web application. |

## Notes

- Most scripts originally used absolute paths on the author's workstation. Before running, check the constants at the top of each file (e.g., `ROOT`, `DATA_DIR`) and point them to your local copy of the repository.
- The large feature tensors (`features.npz`) and raw NetCDF/HDF inputs are not tracked in Git. See `data/README_data.md` for download instructions.
