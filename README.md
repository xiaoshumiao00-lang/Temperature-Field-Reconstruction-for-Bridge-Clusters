# Temperature Field Reconstruction for Bridge Clusters

Reproducible data and code for the manuscript *"Multi-source satellite data fusion for temperature field reconstruction of bridge clusters in mountainous canyons"* (working title).

This repository is organized by chapter so that figures and tables can be re-numbered independently of the original draft. The chapter structure itself is fixed.

---

## Repository structure

```text
.
├── README.md                         # this file
├── LICENSE                           # MIT License
├── requirements.txt                  # Python dependencies
├── data/                             # small, shareable datasets
│   ├── README_data.md                # how to obtain the large raw datasets
│   ├── bridge_inventory.csv/.json    # 207-bridge study inventory
│   ├── measured_air_temperature.parquet
│   ├── sensor_coordinates.json
│   ├── era5_per_bridge.pkl
│   ├── modis_lst_fengjie_2025.csv
│   ├── modis_lst_per_bridge.npy
│   ├── dwg_geometry.json             # benchmark cross-section (DXF-parsed)
│   ├── benchmark_mesh.npz            # 10 mm CEBM mesh, 3526 nodes
│   ├── boundary_2024_benchmark.csv   # 2024 hourly boundary conditions (benchmark)
│   ├── measured_temperatures_2024.csv# 14 channels + cavity air, benchmark bridge
│   ├── fem_benchmark_annual.npz      # slim annual CEBM result (benchmark)
│   └── cluster_results/              # 207-bridge cluster FEM outputs
├── code/                             # core reproducible scripts, by chapter
│   ├── ch03_data_preprocessing/
│   ├── ch04_downscaling/
│   └── ch05_fem_cluster/
├── figures/                          # figures organized by chapter (no figure numbers)
│   ├── ch02_methods/
│   ├── ch03_data_preprocessing/
│   ├── ch04_downscaling/
│   └── ch05_fem_cluster/
└── tables/                           # CSV versions of key tables
    ├── ch03_data_sources.csv
    ├── ch04_model_features.csv
    ├── ch04_lobo_validation.csv
    ├── ch04_wind_speed_correction.csv
    ├── ch05_fem_parameters.csv
    ├── ch05_sensor_calibration.csv
    ├── ch05_sensor_validation.csv
    ├── ch05_sensor_validation_longwindow.csv
    ├── ch05_sensor_validation_sunny6day.csv
    ├── ch05_daily_feature_stats.csv
    ├── ch05_temp_diff_envelope.csv
    ├── ch05_vertical_gradient_stats.csv
    └── ch05_sensor_validation_cluster.csv (placeholder)
```

---

## What is included

### Data
- **Bridge inventory** (`bridge_inventory.csv/.json`): 207 concrete box-girder bridges in the Three Gorges Reservoir area.
- **Measured air temperature** (`measured_air_temperature.parquet`): hourly air-temperature records from the 7 monitored bridges after quality control.
- **ERA5-Land per bridge** (`era5_per_bridge.pkl`): extracted hourly ERA5-Land variables at each bridge location.
- **MODIS LST** (`modis_lst_*.csv/.npy`): clear-sky MODIS land-surface temperature aggregated per bridge.
- **Benchmark cross-section geometry** (`dwg_geometry.json`): exact 2-D triple-cell box-girder polygons (1200 mm x 600 mm) of the 340 m benchmark arch bridge, parsed from the original DXF drawing.
- **Benchmark mesh** (`benchmark_mesh.npz`): the 10 mm Cartesian embedded mesh used in the paper (3526 nodes, edges, surface flags, outward normals).
- **Benchmark boundary conditions** (`boundary_2024_benchmark.csv`): hourly 2024 air temperature, wind speed and shortwave radiation at the benchmark bridge site (UTC).
- **Benchmark measurements** (`measured_temperatures_2024.csv`): hourly structure temperatures at the 14 mid-span channels (N101-N211) plus cavity air N212.
- **Benchmark annual result** (`fem_benchmark_annual.npz`): down-sampled CEBM output — full 2-D field for the representative clear day 2024-08-12 and annual hourly series at the 15 sensor nodes + 7 representative nodes. The full 8777 h x 3526 node field can be regenerated with `run_benchmark_annual.py`.
- **Cluster results** (`cluster_results/`): pre-computed temperature indicators and time series for the 207-bridge cluster.

Large raw datasets (MODIS HDF tiles, SRTM DEM tiles, BNUSSR NetCDF, full ERA5 NetCDF) are **not** included. See [`data/README_data.md`](data/README_data.md) for download instructions.

### Code
- **Chapter 3 – Data preprocessing**: bridge-label reconstruction, ERA5 download script, MODIS extraction script.
- **Chapter 4 – Meteorological downscaling**: physics-based ELR residual correction, XGBoost residual model, and plotting scripts.
- **Chapter 5 – FEM and cluster analysis**: 
  - *Benchmark case (Section 5.1)*: `fsa_fast.py` (vectorised solver), `run_benchmark_annual.py` (full annual re-run), `validate_benchmark_sensors.py` (validation tables), `plot_benchmark_figures.py` (core figures). These four scripts are self-contained and resolve paths relative to the repository root.
  - *Cluster scale (Section 5.2)*: 2-D transient heat-conduction FEM core (`fem_box_core.py`), DXF parser, Streamlit web app, batch cluster computation, and figure-generation helpers.

### Figures and tables
All figures are supplied as high-resolution PNG (vector PDF where available). File names are descriptive rather than numbered, grouped by chapter. The `tables/` folder contains CSV versions of the manuscript tables, including the calibrated sensor positions (`ch05_sensor_calibration.csv`) and the validation statistics (`ch05_sensor_validation*.csv`).

---

## Quick start

1. Create a Python environment (3.10-3.13 recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

2. Download the large raw datasets listed in [`data/README_data.md`](data/README_data.md) and place them under a local `raw_data/` folder (or edit the hard-coded paths in the scripts).

3. Reproduce the Section 5.1 benchmark results (self-contained, uses only committed files):

```bash
cd code/ch05_fem_cluster
python validate_benchmark_sensors.py    # validation tables (seconds)
python plot_benchmark_figures.py        # core figures (seconds)
python run_benchmark_annual.py --tables # full annual FEM re-run (~1e5 s CPU)
```

Expected key numbers (manuscript Section 5.1): 4 calibrated channels, long window
2024-02-26 to 2024-12-31 — RMSE = 1.119 °C, MAE = 0.811 °C, r = 0.991, N = 25471;
sunny period 2024-08-12 to 08-17 — RMSE = 0.977 °C, MAE = 0.719 °C, r = 0.825, N = 573.

4. Run the remaining scripts in chapter order:
   - `code/ch03_data_preprocessing/`
   - `code/ch04_downscaling/`
   - `code/ch05_fem_cluster/` (cluster scripts)

> **Note:** scripts other than the four benchmark scripts originally used absolute paths on the author's workstation. Before running them, adjust the path constants at the top of each script.

---

## Web application

The interactive Streamlit app is in `code/ch05_fem_cluster/streamlit_app.py`. Launch it with:

```bash
streamlit run code/ch05_fem_cluster/streamlit_app.py
```

It supports DXF/CAD cross-section upload, date selection, node-level temperature time-series inspection, and solar-shadow visualization.

---

## Citation

If you use this code or data, please cite the manuscript (DOI to be added upon publication).

---

## Contact

Pan Chen  
School of Civil Engineering, Hubei Engineering University, Xiaogan, China  
chenpan@hbeu.edu.cn
