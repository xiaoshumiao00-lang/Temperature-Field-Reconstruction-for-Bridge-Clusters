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
├── paper/
│   └── manuscript_v7.docx            # latest manuscript draft
├── data/                             # small, shareable datasets
│   ├── README_data.md                # how to obtain the large raw datasets
│   ├── bridge_inventory.csv/.json    # 207-bridge study inventory
│   ├── measured_air_temperature.parquet
│   ├── sensor_coordinates.json
│   ├── era5_per_bridge.pkl
│   ├── modis_lst_fengjie_2025.csv
│   ├── modis_lst_per_bridge.npy
│   ├── fem_result_7days.npz
│   ├── dwg_geometry.json
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
    └── ch05_sensor_validation.csv
```

---

## What is included

### Data
- **Bridge inventory** (`bridge_inventory.csv/.json`): 207 concrete box-girder bridges in the Three Gorges Reservoir area.
- **Measured air temperature** (`measured_air_temperature.parquet`): hourly air-temperature records from the 7 monitored bridges after quality control.
- **ERA5-Land per bridge** (`era5_per_bridge.pkl`): extracted hourly ERA5-Land variables at each bridge location.
- **MODIS LST** (`modis_lst_*.csv/.npy`): clear-sky MODIS land-surface temperature aggregated per bridge.
- **FEM example output** (`fem_result_7days.npz`): 7-day reference temperature field for the benchmark arch bridge.
- **Geometry** (`dwg_geometry.json`): 2-D benchmark box-girder cross-section parsed from the original DXF drawing.
- **Cluster results** (`cluster_results/`): pre-computed temperature indicators and time series for the 207-bridge cluster.

Large raw datasets (MODIS HDF tiles, SRTM DEM tiles, BNUSSR NetCDF, full ERA5 NetCDF) are **not** included. See [`data/README_data.md`](data/README_data.md) for download instructions.

### Code
- **Chapter 3 – Data preprocessing**: bridge-label reconstruction, ERA5 download script, MODIS extraction script.
- **Chapter 4 – Meteorological downscaling**: physics-based ELR residual correction, XGBoost residual model, and plotting scripts.
- **Chapter 5 – FEM and cluster analysis**: 2-D transient heat-conduction FEM core, DXF parser, Streamlit web app, batch cluster computation, and figure-generation helpers.

### Figures and tables
All figures are supplied as both high-resolution PNG and vector PDF. File names are descriptive rather than numbered, grouped by chapter.

---

## Quick start

1. Create a Python environment (3.10–3.13 recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

2. Download the large raw datasets listed in [`data/README_data.md`](data/README_data.md) and place them under a local `raw_data/` folder (or edit the hard-coded paths in the scripts).

3. Run the scripts in chapter order:
   - `code/ch03_data_preprocessing/`
   - `code/ch04_downscaling/`
   - `code/ch05_fem_cluster/`

> **Note:** many scripts originally used absolute paths on the author's workstation. Before running, adjust the path constants at the top of each script or set the `PROJECT_ROOT` environment variable.

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
