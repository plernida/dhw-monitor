"""
DHW Dashboard - Streamlit Web App
---------------------------------
Interactive Degree Heating Weeks (DHW) monitoring for Thailand.

IMPORTANT:
- Current SST is downloaded from NOAA Coral Reef Watch (CRW) ERDDAP online.
- SST is downloaded again whenever Streamlit reruns the script.
- The MMM NetCDF file is used only as the climatological MMM baseline.
- The DHW calculation below is kept in the original form.
"""

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
from datetime import datetime, timedelta

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import requests
import streamlit as st
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# 2. PAGE CONFIGURATION AND BASIC STYLE
# ============================================================

st.set_page_config(
    page_title="DHW Coral Bleaching Monitor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #1f77b4;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

plt.rcParams["font.family"] = "Kanit"


# ============================================================
# 3. COLOR SETTINGS
# ============================================================
# These colors control the appearance of the DHW and SST maps.

colors_rgb = [
    "#C8FAFA",   # 0
    "#FFF000",   # 1
    "#FAAA0A",   # 2
    "#F00000",   # 3
    "#960000",   # 4
    "#A05024",   # 5
    "#F000F0"    # 6+
]

cmap_full = plt.get_cmap("Spectral_r")
colors = cmap_full(np.linspace(0, 0.9, 256))
spectral_slice = LinearSegmentedColormap.from_list(
    "spectral_slice", colors
)


def mpl_to_plotly(cmap, n=256):
    """Convert a Matplotlib colormap to a Plotly colorscale."""
    colorscale = []

    for i in range(n):
        r, g, b, _ = cmap(i / (n - 1))
        colorscale.append([
            i / (n - 1),
            f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        ])

    return colorscale


def create_stepped_colorscale(colors, n_levels=7):
    """
    Create discrete Plotly color bands:
    0, 1, 2, 3, 4, 5 and 6+.
    """
    scale = []

    for i, color in enumerate(colors):
        low = i / n_levels
        high = (i + 1) / n_levels
        scale.append([low, color])
        scale.append([high, color])

    return scale


plotly_colorscale = mpl_to_plotly(spectral_slice, n=21)

cmap_colorscale = create_stepped_colorscale(colors_rgb)

cmap_week = [
    [0.0, "#C8FAFA"],
    [0.499, "#C8FAFA"],
    [0.5, "#FFF000"],
    [1.0, "#FFF000"]
]


# ============================================================
# 4. SIDEBAR / ANALYSIS DATE
# ============================================================
# CRW normally has approximately a 2-day delay.
# Therefore the latest analysis date is set to today - 2 days.

th_tz = pytz.timezone("Asia/Bangkok")
now = datetime.now(th_tz)

target_date = now.date() - timedelta(days=2)

MIN_DATE = datetime(1985, 4, 1).date()
MAX_DATE = target_date

st.sidebar.header("⚙️ Auto Daily Update")

st.sidebar.success(
    f"📅 **Latest Analysis:** {target_date.strftime('%Y-%m-%d')}"
)

st.sidebar.info(
    f"✅ CRW SST 5km: 1985-04-01 → "
    f"{target_date.strftime('%Y-%m-%d')}"
)

analysis_date = st.sidebar.date_input(
    "🎯 Analysis Center Date",
    value=target_date,
    min_value=MIN_DATE,
    max_value=MAX_DATE,
    help="Select center date → auto 30-day backward analysis"
)


# ============================================================
# 5. DATA SOURCE
# ============================================================
# SST is always obtained from the online CRW ERDDAP service.
#
# The MMM NetCDF file is NOT used as current SST.
# It is used only as the historical MMM baseline.

CRW_ERDDAP_BASE = (
    "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km"
)

DAYBACK = 30


# ============================================================
# 6. DOWNLOAD CURRENT SST FROM CRW ONLINE
# ============================================================
# This function intentionally has NO st.cache_data.
# Therefore, every Streamlit script rerun downloads SST again.

def download_latest_sst(enddate, days_back=30):
    """
    Download CRW SST for the selected analysis date.

    Returns
    -------
    sst_stack : xarray.DataArray
        SST with dimensions (lat, lon, time)
    time_list : xarray.DataArray
        Time coordinate
    lat_ref : xarray.DataArray
        Latitude coordinate
    lon_ref : xarray.DataArray
        Longitude coordinate
    """

    thtz = pytz.timezone("Asia/Bangkok")
    now_date = datetime.now(thtz).date()

    # CRW usually lags approximately 2 days.
    latest_available = now_date - timedelta(days=2)

    if enddate > latest_available:
        enddate = latest_available

    start_date = enddate - timedelta(days=days_back - 1)

    start_time = start_date.strftime("%Y-%m-%dT12:00:00Z")
    end_time = enddate.strftime("%Y-%m-%dT12:00:00Z")

    url = (
        f"{CRW_ERDDAP_BASE}.nc?"
        f"CRW_SST"
        f"[({start_time}):1:({end_time})]"
        f"[(0.025):1:(14.075)]"
        f"[(90.025):1:(110.025)]"
    )

    # Download directly from CRW every time this function is called.
    response = requests.get(
        url,
        stream=True,
        timeout=120
    )
    response.raise_for_status()

    # Keep the same working approach as the original code:
    # download the NetCDF response to a temporary local file,
    # then let xarray open it.
    local_file = "latest_sst.nc"

    with open(local_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    ds = xr.open_dataset(local_file)

    # Keep the original variable names used by the DHW code.
    ds = ds.rename({
        "latitude": "lat",
        "longitude": "lon",
        "CRW_SST": "sst"
    })

    ds = ds.sortby("lat")

    # Reorder dimensions to match the original DHW calculation.
    ds = ds.transpose("lat", "lon", "time")

    ds = ds.sel(lon=slice(90, 110))

    sst_stack = ds["sst"]
    lat_ref = ds["lat"]
    lon_ref = ds["lon"]
    time_list = ds["time"]

    return sst_stack, time_list, lat_ref, lon_ref


# ============================================================
# 7. ORIGINAL DHW CALCULATION
# ============================================================
# IMPORTANT:
# The formula in this function is retained from the uploaded
# running code. It has NOT been replaced by a standard NOAA
# DHW formula or another calculation.

def calculate_dhw(TSeries, MMM, threshold=1.0):
    """Calculate Degree Heating Weeks from time series"""
    dhw_weeks = []
    sst_weeks = []
    
    for week in range(6):
        start_idx = (5 - week) * 5
        end_idx = start_idx + 5
        week_mean = np.nanmean(TSeries[:, :, start_idx:end_idx], axis=2)
        sst_weeks.append(week_mean)
        hotspot = week_mean - (MMM + threshold)
        dhw_week = xr.where(hotspot > 0, 1, 0)
        dhw_weeks.append(dhw_week)
    
    # Sum all weeks
    dhw_total = sum(dhw_weeks)
    return dhw_weeks, dhw_total, sst_weeks


# ============================================================
# 8. LOAD COASTLINE
# ============================================================
# Coastline is optional. The application can still calculate
# DHW/SST if the GeoJSON is not available.

@st.cache_data
def load_coastline_geojson(path="coastline.geojson"):
    """Load coastline GeoJSON and convert it to WGS84."""
    if not os.path.exists(path):
        return None

    return gpd.read_file(path).to_crs("EPSG:4326")


def gdf_to_plotly_lines(gdf):
    """Convert GeoPandas geometries to x/y arrays for Plotly."""
    xs, ys = [], []

    for geom in gdf.geometry:
        if geom is None:
            continue

        if geom.geom_type == "LineString":
            x, y = geom.xy
            xs.extend(list(x) + [None])
            ys.extend(list(y) + [None])

        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                x, y = part.xy
                xs.extend(list(x) + [None])
                ys.extend(list(y) + [None])

        elif geom.geom_type == "Polygon":
            x, y = geom.exterior.xy
            xs.extend(list(x) + [None])
            ys.extend(list(y) + [None])

        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                x, y = part.exterior.xy
                xs.extend(list(x) + [None])
                ys.extend(list(y) + [None])

    return xs, ys


COASTLINE_FILE = "geoBoundariesCGAZ_ADM0_resized.geojson"
coast_gdf = load_coastline_geojson(COASTLINE_FILE)


# ============================================================
# 9. DHW ACCUMULATED MAP
# ============================================================

def create_dhw_map(lon, lat, dhw_total, title):
    """Create the accumulated DHW Plotly map."""

    fig = go.Figure(
        data=go.Contour(
            z=dhw_total,
            x=lon,
            y=lat,
            colorscale=cmap_colorscale,
            zmin=0,
            zmax=7,
            contours=dict(
                start=0,
                end=7,
                size=1,
                showlines=False
            ),
            colorbar=dict(
                title="DHW (°C Day)",
                tick0=0,
                dtick=1
            ),
            hovertemplate=(
                "Lon: %{x:.2f}°E<br>"
                "Lat: %{y:.2f}°N<br>"
                "DHW: %{z:.2f}°C Days"
                "<extra></extra>"
            )
        )
    )

    if coast_gdf is not None:
        coast_x, coast_y = gdf_to_plotly_lines(coast_gdf)

        fig.add_trace(
            go.Scatter(
                x=coast_x,
                y=coast_y,
                mode="lines",
                fill="toself",
                fillcolor="rgba(150,150,150,1)",
                line=dict(color="gray", width=2),
                hoverinfo="skip",
                showlegend=False
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        margin=dict(l=40, r=20, t=60, b=40),
        height=800,
        hovermode="closest",
        plot_bgcolor="rgba(240,245,250,1)",
        xaxis=dict(range=[91, 109], constrain="domain"),
        yaxis=dict(range=[1, 14], constrain="domain")
    )

    return fig


# ============================================================
# 10. WEEKLY HOTSPOT MAP
# ============================================================

def create_dhw_weeks(lon, lat, dhw_total, title):
    """Create a binary weekly hotspot Plotly map."""

    fig = go.Figure(
        data=go.Contour(
            z=dhw_total,
            x=lon,
            y=lat,
            colorscale=cmap_week,
            zmin=0,
            zmax=1,
            contours=dict(
                start=0,
                end=2,
                size=1,
                showlines=False
            ),
            colorbar=dict(
                title="DHW (°C Day)",
                tick0=0,
                dtick=1
            ),
            hovertemplate=(
                "Lon: %{x:.2f}°E<br>"
                "Lat: %{y:.2f}°N<br>"
                "DHW: %{z:.2f}°C Days"
                "<extra></extra>"
            )
        )
    )

    if coast_gdf is not None:
        coast_x, coast_y = gdf_to_plotly_lines(coast_gdf)

        fig.add_trace(
            go.Scatter(
                x=coast_x,
                y=coast_y,
                mode="lines",
                fill="toself",
                fillcolor="rgba(150,150,150,1)",
                line=dict(color="gray", width=1),
                hoverinfo="skip",
                showlegend=False
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        margin=dict(l=40, r=20, t=60, b=40),
        height=800,
        hovermode="closest",
        plot_bgcolor="rgba(240,245,250,1)",
        xaxis=dict(range=[91, 109], constrain="domain"),
        yaxis=dict(range=[1, 14], constrain="domain")
    )

    return fig


# ============================================================
# 11. CURRENT SST MAP
# ============================================================

def create_sst_map(lon, lat, sst_data, title):
    """Create the current SST Plotly map."""

    fig = go.Figure(
        data=go.Contour(
            z=sst_data,
            x=lon,
            y=lat,
            colorscale=plotly_colorscale,
            zmin=24,
            zmax=34,
            contours=dict(
                start=24,
                end=34,
                size=0.5,
                showlines=False
            ),
            colorbar=dict(
                title="SST (°C)",
                tickmode="linear",
                tick0=24,
                dtick=0.5
            ),
            hovertemplate=(
                "Lon: %{x:.2f}°E<br>"
                "Lat: %{y:.2f}°N<br>"
                "SST: %{z:.2f}°C"
                "<extra></extra>"
            )
        )
    )

    if coast_gdf is not None:
        coast_x, coast_y = gdf_to_plotly_lines(coast_gdf)

        fig.add_trace(
            go.Scatter(
                x=coast_x,
                y=coast_y,
                mode="lines",
                fill="toself",
                fillcolor="rgba(150,150,150,1)",
                line=dict(color="gray", width=2),
                hoverinfo="skip",
                showlegend=False
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        margin=dict(l=40, r=20, t=60, b=40),
        height=800,
        hovermode="closest",
        plot_bgcolor="rgba(240,245,250,1)",
        xaxis=dict(range=[91, 109], constrain="domain"),
        yaxis=dict(range=[1, 14], constrain="domain")
    )

    return fig


# ============================================================
# 12. MAIN PROCESSING
# ============================================================
# Processing is intentionally performed directly after the
# analysis date is selected.
#
# Every rerun:
#   1. Download CRW SST online
#   2. Load MMM baseline
#   3. Calculate DHW using the original formula
#   4. Generate the three dashboard views

enddate = analysis_date

MMM_FILE = "crw_mmm_sst_thailand_1985-2025.nc"

with st.spinner("Downloading CRW SST and processing DHW analysis..."):

    # --------------------------------------------------------
    # 12.1 Load MMM baseline
    # --------------------------------------------------------
    if not os.path.exists(MMM_FILE):
        st.error(
            f"MMM file not found: {MMM_FILE}"
        )
        st.stop()

    try:
        baseline = xr.open_dataset(MMM_FILE)
    except Exception as e:
        st.error(
            "Unable to open MMM NetCDF file. "
            "Please make sure the uploaded file is a real NetCDF file."
        )
        st.exception(e)
        st.stop()

    MMM = baseline["sst"].sel(
        lon=slice(90, 110),
        lat=slice(14.1, 0)
    )

    # --------------------------------------------------------
    # 12.2 Download SST ONLINE
    # --------------------------------------------------------
    try:
        TSeries, time_list, lat_ref, lon_ref = download_latest_sst(
            enddate,
            days_back=DAYBACK
        )
    except Exception as e:
        st.error(
            "CRW SST download failed. "
            "The dashboard does not use an old/local SST instead."
        )
        st.exception(e)
        st.stop()

    # --------------------------------------------------------
    # 12.3 Calculate DHW
    # --------------------------------------------------------
    # IMPORTANT: Original formula is used here.
    dhw_weeks, dhw_total, sst_weeks = calculate_dhw(
        TSeries,
        MMM
    )

    # --------------------------------------------------------
    # 12.4 Current SST = last downloaded day
    # --------------------------------------------------------
    sst_current = TSeries[:, :, -1]

    # Coordinates used by Plotly maps
    lon = lon_ref.values
    lat = lat_ref.values


# ============================================================
# 13. SUMMARY METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Max DHW",
        f"{float(dhw_total.max().values):.1f} weeks"
    )

with col2:
    st.metric(
        "AVG SST",
        f"{float(np.nanmean(sst_current)):.2f} °C"
    )

with col3:
    alert_area = (
        (dhw_total >= 4).sum()
        / dhw_total.size
        * 100
    )

    st.metric(
        "Alert Area",
        f"{float(alert_area):.1f}%"
    )

with col4:
    bleaching_area = (
        (dhw_total >= 5).sum()
        / dhw_total.size
        * 100
    )

    st.metric(
        "Bleaching Risk",
        f"{float(bleaching_area):.1f}%"
    )


# ============================================================
# 14. DASHBOARD TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Accumulated DHW",
    "🗓️ Weekly Hotspots",
    "🌡️ Current SST"
])


# ============================================================
# 15. TAB 1 — ACCUMULATED DHW
# ============================================================

with tab1:

    st.subheader(
        f"Degree Heating Weeks - {enddate.strftime('%Y-%m-%d')}"
    )

    col_left, col_right = st.columns([80, 20])

    with col_left:

        fig_dhw = create_dhw_map(
            lon=lon,
            lat=lat,
            dhw_total=(
                dhw_total.values
                if hasattr(dhw_total, "values")
                else dhw_total
            ),
            title="Degree Heating Days"
        )

        st.plotly_chart(
            fig_dhw,
            width="stretch",
            key="dhw_map"
        )

    with col_right:

        st.markdown("**📊 DHW Distribution**")

        dhw_flat = dhw_total.values.flatten()

        dhw_counts = (
            pd.Series(dhw_flat)
            .value_counts()
            .sort_index()
            .reindex(range(7), fill_value=0)
        )

        fig_dist = go.Figure(
            data=go.Bar(
                x=dhw_counts.index,
                y=dhw_counts.values,
                marker_color=[
                    "#4270C2",
                    "#D6D6D6",
                    "#EBDEC4",
                    "#E3CCD9",
                    "#C98C59",
                    "#A65959",
                    "#8C4D1A"
                ]
            )
        )

        fig_dist.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            title="Distribution by Level"
        )

        st.plotly_chart(
            fig_dist,
            width="stretch"
        )

        st.markdown("**⚠️ Risk Summary**")

        total_pixels = dhw_total.size

        risk_data = {
            "Alert Level": [
                "Safe (0)",
                "Watch (1-2)",
                "Alert (3-4)",
                "Bleaching (≥5)"
            ],
            "Pixels": [
                int(np.sum(dhw_total == 0)),
                int(np.sum(
                    (dhw_total >= 1) &
                    (dhw_total <= 2)
                )),
                int(np.sum(
                    (dhw_total >= 3) &
                    (dhw_total <= 4)
                )),
                int(np.sum(dhw_total >= 5))
            ],
            "% Area": [
                f"{np.sum(dhw_total == 0) / total_pixels * 100:.1f}%",
                f"{np.sum((dhw_total >= 1) & (dhw_total <= 2)) / total_pixels * 100:.1f}%",
                f"{np.sum((dhw_total >= 3) & (dhw_total <= 4)) / total_pixels * 100:.1f}%",
                f"{np.sum(dhw_total >= 5) / total_pixels * 100:.1f}%"
            ]
        }

        risk_df = pd.DataFrame(risk_data)

        st.dataframe(
            risk_df,
            width="stretch",
            hide_index=True
        )


# ============================================================
# 16. TAB 2 — WEEKLY HOTSPOTS
# ============================================================

with tab2:

    st.subheader("Weekly Hotspot Analysis")

    date_labels = []

    for week in range(6):

        end_day = enddate - timedelta(days=week * 5)
        start_day = end_day - timedelta(days=4)

        date_labels.append(
            f"{start_day.strftime('%d%b')}-"
            f"{end_day.strftime('%d%b')}"
        )

    for row in range(2):

        cols = st.columns(3)

        for col_idx in range(3):

            week_idx = row * 3 + col_idx

            with cols[col_idx]:

                if week_idx < len(dhw_weeks):

                    fig = create_dhw_weeks(
                        lon,
                        lat,
                        dhw_weeks[week_idx],
                        date_labels[week_idx]
                    )

                    fig.update_layout(height=350)

                    st.plotly_chart(
                        fig,
                        width="stretch",
                        key=f"weekly_dhw_{week_idx}"
                    )

                else:
                    st.warning("⚠ No data available")


# ============================================================
# 17. TAB 3 — CURRENT SST
# ============================================================

with tab3:

    st.subheader(
        f"Sea Surface Temperature - "
        f"{enddate.strftime('%Y-%m-%d')}"
    )

    col_left, col_right = st.columns([80, 20])

    with col_left:

        fig_sst = create_sst_map(
            lon=lon,
            lat=lat,
            sst_data=(
                sst_current.values
                if hasattr(sst_current, "values")
                else sst_current
            ),
            title="Current Sea Surface Temperature"
        )

        st.plotly_chart(
            fig_sst,
            width="stretch",
            key="sst_map"
        )

    with col_right:

        st.markdown("**SST Statistics**")

        sst_stats = {
            "Metric": [
                "Mean",
                "Median",
                "Min",
                "Max",
                "Std Dev"
            ],
            "Value (°C)": [
                f"{np.nanmean(sst_current):.2f}",
                f"{np.nanmedian(sst_current):.2f}",
                f"{np.nanmin(sst_current):.2f}",
                f"{np.nanmax(sst_current):.2f}",
                f"{np.nanstd(sst_current):.2f}"
            ]
        }

        st.dataframe(
            pd.DataFrame(sst_stats),
            width="stretch",
            hide_index=True
        )

        fig_hist = go.Figure(
            data=go.Histogram(
                x=sst_current.values.flatten(),
                nbinsx=30,
                marker_color="rgb(55, 83, 109)"
            )
        )

        fig_hist.update_layout(
            title="SST Distribution",
            xaxis_title="Temperature (°C)",
            yaxis_title="Frequency",
            height=300
        )

        st.plotly_chart(
            fig_hist,
            width="stretch",
            key="sst_histogram"
        )
