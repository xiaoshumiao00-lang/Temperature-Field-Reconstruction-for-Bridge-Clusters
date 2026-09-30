# Data availability and download guide

This folder contains only the small, shareable datasets needed to reproduce the tables and figures. Large raw or intermediate files are excluded from the GitHub repository and must be downloaded separately.

---

## Datasets included here

### Bridge-cluster data (Chapter 3-4)

| File | Description | Size |
|------|-------------|------|
| `bridge_inventory.csv` / `bridge_inventory.json` | 207 concrete box-girder bridges in the study area | ~10 KB |
| `measured_air_temperature.parquet` | Quality-controlled hourly air temperature from 7 monitored bridges | ~400 KB |
| `sensor_coordinates.json` | Coordinates of the 7-bridge temperature sensors | ~2 KB |
| `era5_per_bridge.pkl` | ERA5-Land variables extracted at each bridge location, hourly | ~6.6 MB |
| `modis_lst_fengjie_2025.csv` | Clear-sky MODIS LST summary for the study area | ~8 KB |
| `modis_lst_per_bridge.npy` | MODIS LST aggregated per bridge | ~90 KB |
| `cluster_results/` | Pre-computed 207-bridge cluster temperature indicators and time series | ~1.2 MB |

### Benchmark arch-bridge case (Chapter 5.1)

Benchmark case: M04, a 340 m deck-type reinforced-concrete arch bridge
(single box with three cells, 1200 mm x 600 mm mid-span arch-ring
cross-section). All core results of Section 5.1 are reproducible from the
files below plus `code/ch05_fem_cluster/run_benchmark_annual.py`.

| File | Description | Size |
|------|-------------|------|
| `dwg_geometry.json` | Exact benchmark cross-section polygons parsed from the original DXF drawing (mm, origin at section centroid; keys: `outer`, `hole_left`, `hole_center`, `hole_right`) | ~1 KB |
| `benchmark_mesh.npz` | 10 mm Cartesian embedded mesh actually used in the paper: 3526 node coordinates, edge index, surface/hole flags and outward normals | ~50 KB |
| `boundary_2024_benchmark.csv` | Hourly bridge-site boundary conditions for 2024 (UTC): air temperature (degC), 10 m wind speed (m/s), downward shortwave radiation (W/m2). ERA5-Land/BNUSSR-derived with the Chapter-4 corrections (wind scaling 0.39, September +2.5 degC applied by the driver script) | ~400 KB |
| `measured_temperatures_2024.csv` | Hourly measured structure temperatures at the 14 mid-span cross-section channels (N101-N211) plus cavity air N212, 2024-01-02 to 12-31 | ~750 KB |
| `fem_benchmark_annual.npz` | **Down-sampled** annual CEBM result committed for reproduction: full 2-D field (24 hourly slices) for the representative clear day 2024-08-12, annual hourly series at the 15 sensor nodes + 7 representative nodes, all timestamps and node coordinates. Regenerate the full 8777 h x 3526 node field with `run_benchmark_annual.py` | ~1 MB |

The calibrated sensor positions (recorded vs. calibrated coordinates and the
nearest FEM node of each channel) are given in `tables/ch05_sensor_calibration.csv`.

---

## Large raw datasets to download

### 1. ERA5-Land hourly reanalysis

- **Source**: Copernicus Climate Data Store (CDS)  
  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
- **Variables used**: `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `surface_solar_radiation_downwards`, `total_cloud_cover`, `surface_pressure`
- **Spatial domain**: 108-111 deg E, 30-32 deg N
- **Temporal coverage**: 2025-01-01 to 2025-12-31, hourly (cluster); 2024 full year (benchmark boundary reconstruction)
- **Access**: free; requires a CDS account and the `cdsapi` Python package.

After downloading, you can extract per-bridge time series with the script in `code/ch03_data_preprocessing/download_era5_2025.py`.

### 2. MODIS Land Surface Temperature (MOD11A1 / MYD11A1)

- **Source**: NASA LAADS DAAC / USGS  
  https://ladsweb.modaps.eosdis.nasa.gov/search/order/1/MOD11A1  
  https://e4ftl01.cr.usgs.gov/MOLA/MYD11A1.061/
- **Layers used**: `LST_Day_1km`, `LST_Night_1km`, `QC_Day`, `QC_Night`
- **Temporal coverage**: full year of 2025 (and optionally 2023 for comparison)
- **Spatial coverage**: h26v05 / h27v05 (Yangtze River Three Gorges region)
- **Access**: free; requires a NASA Earthdata login.

Quality-control flags are applied to remove cloudy pixels before aggregating per bridge.

### 3. SRTM 30 m DEM

- **Source**: USGS EarthExplorer  
  https://earthexplorer.usgs.gov/
- **Product**: SRTM 1 Arc-Second Global (30 m), tiles around 30-32 deg N, 108-111 deg E
- **Alternative**: NASADEM or ASTER GDEM can be used, but slope/aspect derivatives will differ slightly.
- **Access**: free.

### 4. BNUSSR hourly surface solar radiation

- **Dataset**: BNU-SSR V1.1 hourly 0.05 deg surface solar radiation
- **Source**: National Tibetan Plateau Data Center / BNU Global Solar Radiation archive  
  http://data.tpdc.ac.cn (search "BNUSSR") or the BNU Atmospheric Radiation group page.
- **Variable**: downward shortwave radiation (`rsds`)
- **Temporal coverage**: 2025 full year, hourly (cluster); 2024 (benchmark boundary)
- **Access**: usually free for research use; registration may be required.

> **Why it is needed**: The original ERA5-Land shortwave radiation has a +25 W/m2 bias and does not resolve canyon shading. BNUSSR is used as the solar-radiation boundary condition after spatial and temporal interpolation to each bridge site.

### 5. In-situ bridge monitoring data

The raw logger files for the 7 monitored bridges (cluster air temperature) and for the benchmark arch bridge (14 structure-temperature channels) are project-specific and cannot be redistributed. Cleaned, quality-controlled hourly files (`measured_air_temperature.parquet`, `measured_temperatures_2024.csv`) are included in this repository.

---

## Large intermediate files not included

These files can be regenerated from the raw datasets above using the scripts in `code/`.

| File | Approx. size | How to regenerate |
|------|-------------:|-------------------|
| `fem_benchmark_annual_full.npz` (full benchmark annual field, 8777 h x 3526 nodes) | ~100 MB | Run `code/ch05_fem_cluster/run_benchmark_annual.py` (uses the committed mesh + boundary CSV; ~1e5 s on a modern CPU). |
| `features.npz` (per-bridge hourly feature tensor) | ~110 MB | Run `s1_physics_elr_residual.py` / `s1_downscale_xgb.py` preprocessing with ERA5 + MODIS + DEM inputs. |
| Raw MODIS HDF tiles (MOD11A1/MYD11A1) | ~5 GB | Download from NASA LAADS DAAC for the full year. |
| BNUSSR NetCDF files (12 monthly files) | ~4.5 GB | Download from BNUSSR archive. |
| SRTM tiles and derived terrain features | ~500 MB | Download SRTM tiles and run the DEM processing pipeline. |
| `thermal_field_207bridges.npz` (full cluster thermal fields) | ~60 MB | Run `batch_cluster.py` with the downscaled boundary conditions. |

---

## Data licence notes

- ERA5-Land: Copernicus licence.
- MODIS: NASA open-data policy.
- SRTM: USGS open-data policy.
- BNUSSR: subject to the data provider's research-use terms.
- The cleaned in-situ and bridge-inventory files are provided with this repository for reproducibility only.
