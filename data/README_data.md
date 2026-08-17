# Data availability and download guide

This folder contains only the small, shareable datasets needed to reproduce the tables and figures. Large raw or intermediate files are excluded from the GitHub repository and must be downloaded separately.

---

## Datasets included here

| File | Description | Size |
|------|-------------|------|
| `bridge_inventory.csv` / `bridge_inventory.json` | 207 concrete box-girder bridges in the study area | ~10 KB |
| `measured_air_temperature.parquet` | Quality-controlled hourly air temperature from 7 monitored bridges | ~400 KB |
| `sensor_coordinates.json` | Coordinates of the 7-bridge temperature sensors | ~2 KB |
| `era5_per_bridge.pkl` | ERA5-Land variables extracted at each bridge location, hourly | ~6.6 MB |
| `modis_lst_fengjie_2025.csv` | Clear-sky MODIS LST summary for the study area | ~8 KB |
| `modis_lst_per_bridge.npy` | MODIS LST aggregated per bridge | ~90 KB |
| `fem_result_7days.npz` | Reference 7-day FEM temperature field for the benchmark bridge | ~2.5 MB |
| `dwg_geometry.json` | 2-D benchmark box-girder cross-section from the original DXF | ~1 KB |
| `cluster_results/` | Pre-computed 207-bridge cluster temperature indicators and time series | ~1.2 MB |

---

## Large raw datasets to download

### 1. ERA5-Land hourly reanalysis

- **Source**: Copernicus Climate Data Store (CDS)  
  https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land
- **Variables used**: `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`, `surface_solar_radiation_downwards`, `total_cloud_cover`, `surface_pressure`
- **Spatial domain**: 108–111° E, 30–32° N
- **Temporal coverage**: 2025-01-01 to 2025-12-31, hourly
- **Access**: free; requires a CDS account and the `cdsapi` Python package.

After downloading, you can extract per-bridge time series with the script in `code/ch03_data_preprocessing/download_era5_2025.py` (or adapt `era5_2025_process.py` from the original project).

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
- **Product**: SRTM 1 Arc-Second Global (30 m), tiles around 30–32° N, 108–111° E
- **Alternative**: NASADEM or ASTER GDEM can be used, but slope/aspect derivatives will differ slightly.
- **Access**: free.

### 4. BNUSSR hourly surface solar radiation

- **Dataset**: BNU-SSR V1.1 hourly 0.05° surface solar radiation
- **Source**: National Tibetan Plateau Data Center / BNU Global Solar Radiation archive  
  http://data.tpdc.ac.cn (search "BNUSSR") or the BNU Atmospheric Radiation group page.
- **Variable**: downward shortwave radiation (`rsds`)
- **Temporal coverage**: 2025 full year, hourly
- **Access**: usually free for research use; registration may be required.

> **Why it is needed**: The original ERA5-Land shortwave radiation has a +25 W/m² bias and does not resolve canyon shading. BNUSSR is used as the solar-radiation boundary condition after spatial and temporal interpolation to each bridge site.

### 5. In-situ bridge monitoring data

The raw logger files for the 7 monitored bridges are project-specific and cannot be redistributed. A cleaned, quality-controlled hourly file (`measured_air_temperature.parquet`) is included in this repository.

---

## Large intermediate files not included

These files can be regenerated from the raw datasets above using the scripts in `code/`.

| File | Approx. size | How to regenerate |
|------|-------------:|-------------------|
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
