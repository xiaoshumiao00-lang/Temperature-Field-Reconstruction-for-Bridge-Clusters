#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bridge Temperature Field Calculator — Streamlit Web App (compact single-page layout)
"""
from __future__ import annotations

import io
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dxf_parser import dxf_to_box_section
from fem_box_core import (
    FEMConfig, BoxSection, build_cartesian_mesh, compute_cos_incidence,
    run_fem, solar_geometry, sun_direction_2d, load_dwg_geometry,
)

st.set_page_config(page_title="BridgeTemp Field", page_icon="🌉", layout="wide")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
BENCHMARK_GEO_JSON = ROOT_DIR / "12梅溪河特大桥截面" / "gnn_pipeline" / "dwg_geometry.json"
BENCHMARK_BC_CSV = ROOT_DIR / "12梅溪河特大桥截面" / "FEM" / "boundary_2024_M04.csv"
BENCHMARK_OBS_CSV = ROOT_DIR / "12梅溪河特大桥截面" / "FEM" / "cleaned_temperatures_2024.csv"

BENCHMARK_DEFAULTS = dict(
    lat=31.06,
    lon=107.50,
    face_normal=140.6,
    start=datetime(2024, 8, 12, 0, 0),
    end=datetime(2024, 8, 12, 23, 0),
    alpha_s=0.60,
    k=1.80,
    rho=2400.0,
    cp=920.0,
    longwave_mode="simple",
    dx_mm=25,
    dt=None,
    ws_scale=0.39,
)


def make_demo_weather(n_hours: int = 24 * 7, start: datetime = datetime(2024, 8, 12)) -> pd.DataFrame:
    """Create a synthetic but physically plausible summer weather sequence."""
    times = pd.date_range(start, periods=max(1, n_hours), freq="h")
    hour = times.hour.to_numpy() + times.minute.to_numpy() / 60.0
    rng = np.random.default_rng(int(times[0].timestamp()) % (2 ** 31))
    t2m = 28.0 + 6.0 * np.sin(2 * np.pi * (hour - 6) / 24.0) + 0.5 * rng.standard_normal(len(times))
    hrad = (hour - 12) / 6.0
    ssrd = np.maximum(0.0, 900.0 * (1 - hrad ** 2))
    ws = 1.5 + 0.8 * np.sin(2 * np.pi * hour / 24.0) + 0.3 * rng.standard_normal(len(times))
    ws = np.clip(ws, 0.0, 8.0)
    return pd.DataFrame({
        "datetime": times,
        "t2m_C": t2m,
        "ws_m_s": ws,
        "ssrd_W_m2": ssrd,
    }).set_index("datetime")


def parse_uploaded_weather(file) -> pd.DataFrame:
    """Accept CSV/Excel with columns datetime, t2m_C, ws_m_s, ssrd_W_m2."""
    suffix = file.name.lower().split(".")[-1]
    if suffix == "csv":
        df = pd.read_csv(file)
    elif suffix in ("xlsx", "xls"):
        df = pd.read_excel(file)
    else:
        raise ValueError("Weather file must be CSV or Excel.")
    df["datetime"] = pd.to_datetime(df.datetime)
    return df.set_index("datetime")[["t2m_C", "ws_m_s", "ssrd_W_m2"]]


def load_benchmark_weather(start: datetime, end: datetime) -> tuple[pd.DataFrame, pd.Series | None]:
    """Load measured weather and internal cavity air temperature for the benchmark case."""
    bc = pd.read_csv(BENCHMARK_BC_CSV)
    bc["datetime"] = pd.to_datetime(bc.datetime)
    bc = bc.set_index("datetime").sort_index()
    bc.index = bc.index + pd.Timedelta(hours=8)

    obs = pd.read_csv(BENCHMARK_OBS_CSV)
    obs["DateTime"] = pd.to_datetime(obs.DateTime)
    obs = obs.set_index("DateTime").sort_index()
    hole_air = obs["N212"] if "N212" in obs.columns else None

    times = pd.date_range(start, end, freq="h")
    bc = bc.reindex(times)
    bc["t2m_C"] = bc["t2m_C"].interpolate(method="linear").ffill().bfill()
    bc["ws_m_s"] = bc["ws_m_s"].interpolate(method="linear").ffill().bfill()
    bc["ssrd_W_m2"] = bc["ssrd_W_m2"].interpolate(method="linear").ffill().bfill()

    if hole_air is not None:
        hole_air = hole_air.reindex(times).interpolate(method="linear").ffill().bfill()
    return bc, hole_air


@st.cache_data(show_spinner=False)
def cached_mesh(section_key: str, dx: float):
    """Cache mesh building for fast re-runs."""
    if section_key == "benchmark":
        poly = load_dwg_geometry(str(BENCHMARK_GEO_JSON))
        return build_cartesian_mesh(poly, dx)
    # Fallback: deserialize is not trivial; return None and build live
    return None


@st.cache_data(show_spinner=False)
def cached_cos_inc(mesh_key: str, section_key: str, times_tuple: tuple, lat: float, face_normal: float):
    """Cache cos-incidence matrix."""
    return None


def plot_temperature_field(mesh: dict, T: np.ndarray, title: str = "Temperature field",
                           sunlit=None, sun_vec=None) -> plt.Figure:
    """Plot the section temperature field in the same style as the research FEM figures."""
    coords = mesh["coords"]
    x_mm = coords[:, 0] * 1000.0
    y_mm = coords[:, 1] * 1000.0

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    # Use tripcolor for a continuous, filled look over the concrete nodes.
    # Mask triangles whose centroid falls outside the concrete polygon so cavities stay white.
    from matplotlib.tri import Triangulation
    from shapely.geometry import Point
    tri = Triangulation(x_mm, y_mm)
    section_poly = mesh.get("section_polygon")
    if section_poly is not None:
        centroids_x = np.mean(x_mm[tri.triangles], axis=1)
        centroids_y = np.mean(y_mm[tri.triangles], axis=1)
        # convert mm -> m for polygon test
        inside = np.array([section_poly.contains(Point(cx / 1000.0, cy / 1000.0))
                           for cx, cy in zip(centroids_x, centroids_y)])
        tri.set_mask(~inside)
    # Round color limits to 5 °C for clean color-bar ticks
    t_min, t_max = float(np.percentile(T, 0.5)), float(np.percentile(T, 99.5))
    vmin = 5 * np.floor(t_min / 5.0)
    vmax = 5 * np.ceil(t_max / 5.0)
    tc = ax.tripcolor(tri, T, cmap="jet", vmin=vmin, vmax=vmax, shading="gouraud", zorder=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=10)
    ax.set_ylabel("y (mm)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=9)

    if sunlit is not None and np.any(sunlit):
        lit_coords = coords[sunlit]
        ax.scatter(lit_coords[:, 0] * 1000, lit_coords[:, 1] * 1000, marker="o",
                   facecolors="none", edgecolors="lime", s=18, linewidths=1.0,
                   label="Sunlit edges", zorder=6)
    if sun_vec is not None and np.linalg.norm(sun_vec) > 1e-9:
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        span_x = xlim[1] - xlim[0]; span_y = ylim[1] - ylim[0]
        L = min(span_x, span_y) * 0.16
        x0 = xlim[0] + 0.07 * span_x
        y0 = ylim[0] + 0.07 * span_y
        v = np.asarray(sun_vec, dtype=float)
        v = v / np.linalg.norm(v)
        ax.arrow(x0, y0, v[0] * L, v[1] * L, head_width=L * 0.10,
                 head_length=L * 0.14, fc="darkorange", ec="darkorange", zorder=7)
        ax.text(x0 + v[0] * L * 1.12, y0 + v[1] * L * 1.12, "Sun",
                fontsize=8, color="darkorange", zorder=7)
    if (sunlit is not None and np.any(sunlit)) or (sun_vec is not None and np.linalg.norm(sun_vec) > 1e-9):
        ax.legend(loc="upper left", fontsize=7)

    cbar = fig.colorbar(tc, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Temperature (°C)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def plot_time_series(mesh: dict, T_all: np.ndarray, times,
                     point_xy=None, node_idx=None) -> plt.Figure:
    """Compact time-series plot."""
    times_arr = pd.to_datetime(times).to_numpy()
    top_idx = np.where(mesh["is_top"])[0]
    bot_idx = np.where(mesh["is_bot"])[0]
    if len(top_idx) == 0:
        top_idx = np.where(mesh["is_surface"] & (mesh["coords"][:, 1] > 0))[0]
    if len(bot_idx) == 0:
        bot_idx = np.where(mesh["is_surface"] & (mesh["coords"][:, 1] < 0))[0]

    T_top = np.mean(T_all[1:, top_idx], axis=1)
    T_bot = np.mean(T_all[1:, bot_idx], axis=1)

    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    ax.plot(times_arr, T_top, label="Top surface", linewidth=1.4)
    ax.plot(times_arr, T_bot, label="Bottom surface", linewidth=1.4)

    if node_idx is not None:
        px, py = mesh["coords"][node_idx] * 1000.0
        ax.plot(times_arr, T_all[1:, node_idx],
                label=f"Node #{node_idx} ({px:.0f}, {py:.0f}) mm", linewidth=1.6)
    elif point_xy is not None:
        px, py = point_xy
        dist = np.sum((mesh["coords"] - np.array([[px / 1000.0, py / 1000.0]])) ** 2, axis=1)
        node = int(np.argmin(dist))
        ax.plot(times_arr, T_all[1:, node], label=f"Probe ({px:.0f}, {py:.0f}) mm", linewidth=1.6)

    ax.set_xlabel("Date/time", fontsize=9)
    ax.set_ylabel("Temperature (°C)", fontsize=9)
    ax.set_title("Temperature time series", fontsize=10)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def compact_preview_figure(coords: np.ndarray) -> plt.Figure:
    """Small mesh preview figure."""
    fig, ax = plt.subplots(figsize=(2.6, 2.0))
    ax.scatter(coords[:, 0] * 1000, coords[:, 1] * 1000, s=1.2, alpha=0.7, c="gray")
    ax.set_aspect("equal")
    ax.set_title("Mesh preview", fontsize=9)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.set_ylabel("y (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def main():
    # ---- Top bar: title + run button ----
    c_title, c_run = st.columns([5, 1])
    with c_title:
        st.title("🌉 Bridge Temperature Field Calculator")
    with c_run:
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        run_clicked = st.button("🚀 Run FEM", type="primary", use_container_width=True)

    st.caption("Calculate the 2-D transient thermal field of a concrete box-girder bridge section.")

    tab_input, tab_field, tab_series = st.tabs(["Inputs", "Temperature field", "Time series"])

    # ------------------------------------------------------------------
    # ---- Inputs tab (compact, single-screen) ----
    # ------------------------------------------------------------------
    with tab_input:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            # ---- Geometry source ----
            r_geo = st.container()
            c1, c2 = r_geo.columns([1, 1])
            with c1:
                use_benchmark = st.checkbox("Use benchmark geometry", value=True)
            with c2:
                uploaded_dxf = st.file_uploader("Or upload DXF", type=["dxf"], key="dxf_uploader",
                                                disabled=use_benchmark)
            if uploaded_dxf is not None:
                use_benchmark = False

            section_polygon = None
            section_for_mesh = None
            geom_label = ""
            if use_benchmark:
                if BENCHMARK_GEO_JSON.exists():
                    section_polygon = load_dwg_geometry(str(BENCHMARK_GEO_JSON))
                    section_for_mesh = section_polygon
                    geom_label = "benchmark"
                    st.success("Loaded benchmark geometry.")
                else:
                    st.error("Benchmark geometry file missing.")
                    use_benchmark = False
            else:
                if uploaded_dxf is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
                        tmp.write(uploaded_dxf.getvalue())
                        tmp_path = tmp.name
                    sec_parsed, err = dxf_to_box_section(tmp_path)
                    os.unlink(tmp_path)
                    if sec_parsed is not None:
                        section_for_mesh = sec_parsed
                        geom_label = f"{sec_parsed.n_cells}-cell box"
                        st.success(f"Loaded DXF geometry.")
                    else:
                        st.error(f"DXF parse failed.")
                        section_for_mesh = BoxSection()
                        geom_label = "default"
                else:
                    section_for_mesh = BoxSection()
                    geom_label = "default"

            show_override = st.checkbox("Edit parametric geometry", value=False)
            if not use_benchmark and show_override:
                r1, r2, r3 = st.columns(3)
                width_m = r1.number_input("Width (m)", 0.5, 25.0, 12.0, 0.1)
                height_m = r2.number_input("Depth (m)", 0.3, 8.0, 3.0, 0.1)
                n_cells = r3.selectbox("Cells", [1, 2, 3], index=2)
                r4, r5, r6 = st.columns(3)
                top_slab_m = r4.number_input("Top slab (m)", 0.05, 1.0, 0.35, 0.05)
                bottom_slab_m = r5.number_input("Bottom slab (m)", 0.05, 1.0, 0.30, 0.05)
                web_m = r6.number_input("Web (m)", 0.05, 1.2, 0.45, 0.05)
                overhang_m = st.number_input("Overhang (m)", 0.0, 4.0, 1.2, 0.1)
                section_for_mesh = BoxSection(width=width_m, height=height_m, top_slab=top_slab_m,
                                              bottom_slab=bottom_slab_m, web_thick=web_m,
                                              n_cells=n_cells, overhang=overhang_m)

            # ---- Materials (compact) ----
            def_mat = BENCHMARK_DEFAULTS if use_benchmark else dict(
                alpha_s=0.45, k=1.80, rho=2400.0, cp=920.0, longwave_mode="none", ws_scale=1.0
            )
            c_m1, c_m2, c_m3, c_m4, c_m5, c_m6 = st.columns(6)
            with c_m1:
                alpha_s = st.number_input("Solar abs. αs", 0.2, 0.9, def_mat["alpha_s"], 0.05)
            with c_m2:
                k = st.number_input("Conductivity k", 1.0, 3.0, def_mat["k"], 0.05,
                                    help="Thermal conductivity (W·m⁻¹·K⁻¹)")
            with c_m3:
                rho = st.number_input("Density ρ", 2000.0, 2800.0, def_mat["rho"], 50.0,
                                      help="Concrete density (kg·m⁻³)")
            with c_m4:
                cp = st.number_input("Specific heat cp", 700.0, 1200.0, def_mat["cp"], 20.0,
                                     help="Specific heat capacity (J·kg⁻¹·K⁻¹)")
            with c_m5:
                longwave_mode = st.selectbox("Long-wave", ["none", "simple"],
                                             index=0 if def_mat["longwave_mode"] == "none" else 1,
                                             help="Long-wave radiation model")
            with c_m6:
                ws_scale = st.number_input("Wind scale", 0.1, 2.0, def_mat["ws_scale"], 0.01,
                                           help="Empirical wind-speed scaling factor")

            # ---- Site / time (compact) ----
            weather_options = ["Demo", "Upload CSV/Excel"]
            if BENCHMARK_BC_CSV.exists():
                weather_options.append("Measured benchmark")

            c_t1, c_t2, c_t3, c_t4, c_t5 = st.columns([1.2, 1.2, 0.8, 0.8, 0.8])
            with c_t1:
                weather_mode = st.selectbox("Weather", weather_options,
                                            index=2 if use_benchmark and len(weather_options) > 2 else 0)
            with c_t2:
                wfile = None
                if weather_mode == "Upload CSV/Excel":
                    wfile = st.file_uploader("Weather file", type=["csv", "xlsx", "xls"], key="weather_uploader",
                                             label_visibility="collapsed")
                else:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

            start_default = BENCHMARK_DEFAULTS["start"] if use_benchmark else datetime(2024, 8, 12, 0, 0)
            end_default = BENCHMARK_DEFAULTS["end"] if use_benchmark else datetime(2024, 8, 12, 23, 0)
            with c_t3:
                start = st.datetime_input("Start", value=start_default)
            with c_t4:
                end = st.datetime_input("End", value=end_default)
            if end <= start:
                st.error("End must be after Start.")
                end = start + timedelta(days=1) - timedelta(hours=1)
            with c_t5:
                dx_mm = st.selectbox("Mesh (mm)", [50, 25, 20, 10], index=1)

            c_l1, c_l2, c_l3, c_n1, c_n2 = st.columns(5)
            with c_l1:
                lat = st.number_input("Lat (°N)", -60.0, 60.0,
                                     BENCHMARK_DEFAULTS["lat"] if use_benchmark else 31.06, 0.01)
            with c_l2:
                lon = st.number_input("Lon (°E)", -180.0, 180.0,
                                     BENCHMARK_DEFAULTS["lon"] if use_benchmark else 107.50, 0.01)
            with c_l3:
                face_normal = st.number_input("Azimuth (°)", 0.0, 360.0,
                                             BENCHMARK_DEFAULTS["face_normal"] if use_benchmark else 140.6, 1.0)
            with c_n1:
                dt_auto = st.checkbox("Auto Δt", value=True)
            with c_n2:
                dt_s = st.number_input("Δt (s)", 1.0, 300.0,
                                       30.0, 1.0, disabled=dt_auto)

        # ---- Build mesh (shared) ----
        cfg = FEMConfig(
            alpha_solar=alpha_s, k=k, rho=rho, cp=cp,
            dx=dx_mm / 1000.0,
            dt=None if dt_auto else dt_s,
            face_normal_deg=face_normal,
            longwave_mode=longwave_mode,
            ws_scale=ws_scale,
        )
        with st.spinner("Meshing..."):
            # Try cache for benchmark first
            if use_benchmark and BENCHMARK_GEO_JSON.exists():
                mesh = cached_mesh("benchmark", cfg.dx)
                if mesh is None:
                    mesh = build_cartesian_mesh(section_for_mesh, cfg.dx)
            else:
                mesh = build_cartesian_mesh(section_for_mesh, cfg.dx)

        with col_right:
            st.metric("Nodes", mesh["coords"].shape[0])
            st.metric("Elements", int(mesh["is_surface"].sum()))
            st.pyplot(compact_preview_figure(mesh["coords"]), use_container_width=True)

        # ---- Weather assembly ----
        hole_air = None
        if weather_mode == "Measured benchmark":
            weather, hole_air = load_benchmark_weather(start, end)
        elif weather_mode == "Upload CSV/Excel":
            if wfile is None:
                st.warning("Upload a weather file or switch to Demo/Measured.")
                n_hours = int((end - start).total_seconds() // 3600) + 1
                weather = make_demo_weather(n_hours, start)
            else:
                weather = parse_uploaded_weather(wfile)
                weather = weather.reindex(pd.date_range(start, end, freq="h"))
                weather = weather.interpolate(method="linear").ffill().bfill()
        else:
            n_hours = int((end - start).total_seconds() // 3600) + 1
            weather = make_demo_weather(n_hours, start)

        times = weather.index.copy()
        bc = weather[["t2m_C", "ws_m_s", "ssrd_W_m2"]].copy()

        if run_clicked:
            with st.spinner("Computing..."):
                cos_inc = compute_cos_incidence(mesh, times, lat, face_normal,
                                                use_raytracing=True,
                                                section_polygon=section_polygon)
                T_all = run_fem(mesh, bc, times, cos_inc, cfg, hole_air=hole_air, verbose=True)
            st.session_state["T_all"] = T_all
            st.session_state["mesh"] = mesh
            st.session_state["times"] = times
            st.session_state["cos_inc"] = cos_inc
            st.session_state["hole_air"] = hole_air
            st.success(f"Done: {len(times)} h, {mesh['coords'].shape[0]} nodes.")

    # ------------------------------------------------------------------
    # ---- Outputs ----
    # ------------------------------------------------------------------
    T_all = st.session_state.get("T_all")
    mesh = st.session_state.get("mesh")
    times = st.session_state.get("times")
    cos_inc = st.session_state.get("cos_inc")

    with tab_field:
        if T_all is not None:
            c_ctrl, c_plot = st.columns([1, 3])
            with c_ctrl:
                hour_idx = st.slider("Hour", 0, len(times) - 1,
                                     min(len(times) // 2, len(times) - 1))
                show_sun = st.checkbox("Show sunlit edges", value=False)
                if show_sun:
                    st.caption("Green = sunlit; orange arrow = Sun direction.")
                st.markdown("**Indicators**")
                st.write(f"Max T: {float(np.max(T_all[hour_idx + 1])):.1f} °C")
                st.write(f"Min T: {float(np.min(T_all[hour_idx + 1])):.1f} °C")
            with c_plot:
                title = f"Temperature field at {times[hour_idx]}"
                sunlit = None
                sun_vec = None
                if show_sun and cos_inc is not None:
                    elev, azim = solar_geometry(times[hour_idx], lat)
                    if elev > 0:
                        sunlit = (cos_inc[hour_idx] > 0.05) & mesh["is_surface"]
                        sun_vec = sun_direction_2d(elev, azim, face_normal)
                        title += " (sunlit)"
                fig_field = plot_temperature_field(mesh, T_all[hour_idx + 1], title=title,
                                                   sunlit=sunlit, sun_vec=sun_vec)
                st.pyplot(fig_field, use_container_width=True)
                buf = io.BytesIO()
                fig_field.savefig(buf, format="png", dpi=300, bbox_inches="tight")
                st.download_button("Download field PNG", buf.getvalue(),
                                   file_name=f"bridge_temp_field_{hour_idx:04d}.png")
        else:
            st.info("Click 🚀 Run FEM (top) to view the temperature field.")

    with tab_series:
        if T_all is not None:
            c_sel, c_plot = st.columns([1, 1])
            with c_sel:
                st.markdown("**Select point**")
                coords = mesh["coords"]
                fig_sel = go.Figure()
                fig_sel.add_trace(go.Scatter(
                    x=coords[:, 0] * 1000,
                    y=coords[:, 1] * 1000,
                    mode="markers",
                    marker=dict(size=4, color="steelblue", opacity=0.85),
                    customdata=np.arange(coords.shape[0]),
                    hovertemplate="node %{customdata}<br>x=%{x:.0f} mm<br>y=%{y:.0f} mm<extra></extra>",
                ))
                fig_sel.update_layout(
                    height=260,
                    title="Click a node",
                    xaxis_title="x (mm)",
                    yaxis_title="y (mm)",
                    margin=dict(l=30, r=10, t=35, b=30),
                    clickmode="event+select",
                )
                st.plotly_chart(fig_sel, key="mesh_select", on_select="rerun", selection_mode="points")

                node_idx = None
                try:
                    pts = st.session_state["mesh_select"]["selection"]["points"]
                    if pts:
                        node_idx = int(pts[0]["point_index"])
                except Exception:
                    node_idx = None

                px_col, py_col = st.columns(2)
                with px_col:
                    px = st.number_input("Probe x (mm)", value=0.0, step=100.0)
                with py_col:
                    py = st.number_input("Probe y (mm)", value=0.0, step=100.0)

                if node_idx is not None:
                    nx_mm, ny_mm = coords[node_idx] * 1000.0
                    st.success(f"Node #{node_idx} ({nx_mm:.0f}, {ny_mm:.0f}) mm")

            with c_plot:
                if node_idx is not None:
                    fig_series = plot_time_series(mesh, T_all, times, node_idx=node_idx)
                else:
                    fig_series = plot_time_series(mesh, T_all, times, point_xy=(px, py))
                st.pyplot(fig_series, use_container_width=True)
                buf2 = io.BytesIO()
                fig_series.savefig(buf2, format="png", dpi=300, bbox_inches="tight")
                st.download_button("Download time-series PNG", buf2.getvalue(),
                                   file_name="bridge_temp_series.png")
        else:
            st.info("Click 🚀 Run FEM (top) to view time series.")


if __name__ == "__main__":
    main()
