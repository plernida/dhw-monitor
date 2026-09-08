"""
DHW Dashboard - Streamlit Web App

SST is downloaded from CRW ERDDAP every time the Streamlit script reruns.
The MMM file is used only as the climatological baseline.

IMPORTANT:
- calculate_dhw() is kept unchanged from the original application.
- No pre-generated SST/DHW PNG or static DHW statistics are used for analysis.
"""

import os
import warnings
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

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DHW Coral Bleaching Monitor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { padding: 0rem 1rem; }
    h1 { color: #1f77b4; }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CRW_ERDDAP_BASE = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km"
MMM_FILE = "crw_mmm_sst_thailand_1985-2025.nc"
COASTLINE_FILE = "geoBoundariesCGAZ_ADM0_resized.geojson"

LON_MIN, LON_MAX = 90.025, 110.025
LAT_MIN, LAT_MAX = 0.025, 14.075
DAYS_BACK = 30

# -----------------------------------------------------------------------------
# Colors
# -----------------------------------------------------------------------------
cmap_full = plt.get_cmap("Spectral_r")
colors = cmap_full(np.linspace(0, 0.9, 256))
spectral_slice = LinearSegmentedColormap.from_list(
    "spectral_slice", colors
)

colors_rgb = [
    "#C8FAFA",
    "#FFF000",
    "#FAAA0A",
    "#F00000",
    "#960000",
    "#A05024",
    "#F000F0",
]

cmap_week = [
    [0.0, "#C8FAFA"],
    [0.499, "#C8FAFA"],
    [0.5, "#FFF000"],
    [1.0, "#FFF000"],
]


def mpl_to_plotly(cmap, n=256):
    """Convert a Matplotlib colormap to a Plotly colorscale."""
    result = []
    for i in range(n):
        r, g, b, _ = cmap(i / (n - 1))
        result.append(
            [
                i / (n - 1),
                f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})",
            ]
        )
    return result


plotly_colorscale = mpl_to_plotly(spectral_slice, n=21)


def create_stepped_colorscale(color_list, n_levels=7):
    """Create flat Plotly color bands for DHW levels."""
    scale = []
    for i, color in enumerate(color_list):
        low = i / n_levels
        high = (i + 1) / n_levels
        scale.append([low, color])
        scale.append([high, color])
    return scale


cmap_colorscale = create_stepped_colorscale(colors_rgb)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
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
    f"✅ CRW SST 5km: 1985-04-01 → {target_date.strftime('%Y-%m-%d')}"
)

analysis_date = st.sidebar.date_input(
    "🎯 Analysis Center Date",
    value=target_date,
    min_value=MIN_DATE,
    max_value=MAX_DATE,
    help="Select center date → download 30 days of CRW SST online",
)

st.sidebar.caption(
    "SST is downloaded online from CRW ERDDAP on every Streamlit rerun."
)

# -----------------------------------------------------------------------------
# Online SST download
# -----------------------------------------------------------------------------
def download_latest_sst(enddate, days_back=30):
    """
    Download CRW SST directly from ERDDAP.

    There is intentionally NO @st.cache_data here.
    Therefore every Streamlit rerun performs a new online request.
    """
    latest_available = datetime.now(th_tz).date() - timedelta(days=2)

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

    st.write(f"**SST source:** CRW ERDDAP | {start_date} → {enddate}")

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "ไม่สามารถดาวน์โหลด CRW SST จาก ERDDAP ได้ "
            f"โปรดตรวจสอบ Internet/ERDDAP connection\n\n{exc}"
        ) from exc

    # Use a unique temporary filename so old SST files cannot be accidentally used.
    local_file = "_crw_sst_current_download.nc"
    with open(local_file, "wb") as f:
        f.write(response.content)

    try:
        with xr.open_dataset(local_file) as source_ds:
            ds = source_ds.load()
    except Exception as exc:
        raise RuntimeError(
            "ไฟล์ที่ดาวน์โหลดจาก CRW ไม่สามารถเปิดเป็น NetCDF ได้ "
            f"\nURL: {url}\n\n{exc}"
        ) from exc
    finally:
        if os.path.exists(local_file):
            os.remove(local_file)

    required = {"latitude", "longitude", "CRW_SST"}
    missing = required - set(ds.variables)
    if missing:
        raise RuntimeError(
            f"CRW dataset ไม่มีตัวแปรที่ต้องใช้: {sorted(missing)}"
        )

    ds = ds.rename(
        {
            "latitude": "lat",
            "longitude": "lon",
            "CRW_SST": "sst",
        }
    )

    ds = ds.sortby("lat")
    ds = ds.transpose("lat", "lon", "time")
    ds = ds.sel(lon=slice(90, 110))

    sst_stack = ds["sst"]
    lat_ref = ds["lat"]
    lon_ref = ds["lon"]
    time_list = ds["time"]

    return sst_stack, time_list, lat_ref, lon_ref


# -----------------------------------------------------------------------------
# MMM baseline
# -----------------------------------------------------------------------------
def load_mmm(mmm_file=MMM_FILE):
    """Load only the MMM climatological baseline; current SST never comes from this file."""
    if not os.path.exists(mmm_file):
        raise FileNotFoundError(
            f"ไม่พบไฟล์ MMM: {mmm_file}\n"
            "ต้องมีไฟล์ MMM ใน repository เพื่อใช้เป็น baseline ของ DHW"
        )

    try:
        with xr.open_dataset(mmm_file) as source_ds:
            baseline = source_ds.load()
    except Exception as exc:
        raise RuntimeError(
            f"เปิด MMM NetCDF ไม่ได้: {mmm_file}\n\n{exc}"
        ) from exc

    if "sst" not in baseline:
        raise RuntimeError(
            "MMM file ต้องมีตัวแปรชื่อ 'sst' เพื่อใช้กับ calculate_dhw()."
        )

    MMM = baseline["sst"].sel(
        lon=slice(90, 110),
        lat=slice(0, 14.1),
    ).sortby("lat")

    return MMM


# -----------------------------------------------------------------------------
# DHW calculation — DO NOT CHANGE
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Coordinate/grid alignment
# -----------------------------------------------------------------------------
def align_mmm_to_sst(MMM, lat_ref, lon_ref):
    """Interpolate only the MMM baseline onto the downloaded CRW SST grid."""
    return MMM.interp(lat=lat_ref, lon=lon_ref, method="nearest")


# -----------------------------------------------------------------------------
# Coastline
# -----------------------------------------------------------------------------
def load_coastline_geojson(path=COASTLINE_FILE):
    if not os.path.exists(path):
        return None
    try:
        return gpd.read_file(path).to_crs("EPSG:4326")
    except Exception:
        return None


def gdf_to_plotly_lines(gdf):
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


coast_gdf = load_coastline_geojson()
if coast_gdf is None:
    st.sidebar.warning(
        "⚠️ Coastline GeoJSON not found. Maps will run without coastline overlay."
    )

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def add_coastline(fig):
    if coast_gdf is None:
        return fig

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
            showlegend=False,
        )
    )
    return fig


def create_dhw_map(lon, lat, dhw_total, title):
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
                showlines=False,
            ),
            colorbar=dict(
                title="DHW (°C Day)",
                tick0=0,
                dtick=1,
            ),
            hovertemplate=(
                "Lon: %{x:.2f}°E<br>"
                "Lat: %{y:.2f}°N<br>"
                "DHW: %{z:.2f}°C Days<extra></extra>"
            ),
        )
    )

    fig = add_coastline(fig)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        margin=dict(l=40, r=20, t=60, b=40),
        height=800,
        hovermode="closest",
        plot_bgcolor="rgba(240,245,250,1)",
        xaxis=dict(range=[91, 109], constrain="domain"),
        yaxis=dict(range=[1, 14], constrain="domain"),
    )
    return fig


def create_dhw_weeks(lon, lat, dhw_total, title):
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
                showlines=False,
            ),
            colorbar=dict(
                title="Hotspot",
                tick0=0,
                dtick=1,
            ),
            hovertemplate=(
                "Lon: %{x:.2f}°E<br>"
                "Lat: %{y:.2f}°N<br>"
                "Hotspot: %{z}<extra></extra>"
            ),
        )
    )

    fig = add_coastline(fig)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        margin=dict(l=40, r=20, t=60, b=40),
        height=350,
        hovermode="closest",
        plot_bgcolor="rgba(240,245,250,1)",
        xaxis=dict(range=[91, 109], constrain="domain"),
        yaxis=dict(range=[1, 14], constrain="domain"),
    )
    return fig


def create_sst_map(lon, lat, sst_data, title):
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
                showlines=False,
            ),
            colorbar=dict(
                title="SST (°C)",
                tickmode="linear",
                tick0=24,
                dtick=0.5,
            ),
            hovertemplate=(
                "Lon: %{x:.2f}°E<br>"
                "Lat: %{y:.2f}°N<br>"
                "SST: %{z:.2f}°C<extra></extra>"
            ),
        )
    )

    fig = add_coastline(fig)
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        margin=dict(l=40, r=20, t=60, b=40),
        height=800,
        hovermode="closest",
        plot_bgcolor="rgba(240,245,250,1)",
        xaxis=dict(range=[91, 109], constrain="domain"),
        yaxis=dict(range=[1, 14], constrain="domain"),
    )
    return fig


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------
enddate = analysis_date

with st.spinner("🌊 Downloading current CRW SST online and processing DHW..."):
    try:
        # 1. ALWAYS download current/selected SST from CRW ERDDAP.
        TSeries, time_list, lat_ref, lon_ref = download_latest_sst(
            enddate, days_back=DAYS_BACK
        )

        # 2. Load MMM baseline only.
        MMM = load_mmm()

        # 3. Align MMM to the fresh online SST grid.
        MMM = align_mmm_to_sst(MMM, lat_ref, lon_ref)

        # 4. Calculate DHW using the ORIGINAL formula unchanged.
        dhw_weeks, dhw_total, sst_weeks = calculate_dhw(TSeries, MMM)

        # Current SST is the final day from the freshly downloaded online data.
        sst_current = TSeries[:, :, -1]
        lon = lon_ref.values
        lat = lat_ref.values

    except Exception as exc:
        st.error("❌ ไม่สามารถประมวลผลข้อมูล SST/DHW ได้")
        st.exception(exc)
        st.stop()

# -----------------------------------------------------------------------------
# Statistics — calculated from fresh online SST/DHW
# -----------------------------------------------------------------------------
st.info(
    f"🔄 SST downloaded online for analysis date **{enddate.strftime('%Y-%m-%d')}** "
    f"({DAYS_BACK} days). No cached SST/PNG/statistics are used."
)

col1, col2, col3, col4 = st.columns(4)

max_dhw = float(dhw_total.max().values)
avg_sst = float(np.nanmean(sst_current))
alert_area = float((dhw_total >= 4).sum().values / dhw_total.size * 100)
bleaching_area = float((dhw_total >= 5).sum().values / dhw_total.size * 100)

with col1:
    st.metric("Max DHW", f"{max_dhw:.0f} weeks")
with col2:
    st.metric("Avg SST", f"{avg_sst:.2f} °C")
with col3:
    st.metric("Alert Area", f"{alert_area:.1f}%")
with col4:
    st.metric("Bleaching Risk", f"{bleaching_area:.1f}%")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Accumulated DHW", "🗓️ Weekly Hotspots", "🌡️ Current SST"]
)

with tab1:
    st.subheader(f"Degree Heating Weeks - {enddate.strftime('%Y-%m-%d')}")

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
            title="Degree Heating Days",
        )
        st.plotly_chart(fig_dhw, width="stretch", key="dhw_map")

    with col_right:
        st.markdown("**📊 DHW Distribution**")
        dhw_flat = np.asarray(dhw_total.values).flatten()
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
                    "#8C4D1A",
                ],
            )
        )
        fig_dist.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            title="Distribution by Level",
        )
        st.plotly_chart(fig_dist, width="stretch")

        st.markdown("**⚠️ Risk Summary**")
        total_pixels = dhw_total.size
        risk_data = {
            "Alert Level": [
                "Safe (0)",
                "Watch (1-2)",
                "Alert (3-4)",
                "Bleaching (≥5)",
            ],
            "Pixels": [
                int(np.sum(dhw_total == 0)),
                int(np.sum((dhw_total >= 1) & (dhw_total <= 2))),
                int(np.sum((dhw_total >= 3) & (dhw_total <= 4))),
                int(np.sum(dhw_total >= 5)),
            ],
            "% Area": [
                f"{np.sum(dhw_total == 0) / total_pixels * 100:.1f}%",
                f"{np.sum((dhw_total >= 1) & (dhw_total <= 2)) / total_pixels * 100:.1f}%",
                f"{np.sum((dhw_total >= 3) & (dhw_total <= 4)) / total_pixels * 100:.1f}%",
                f"{np.sum(dhw_total >= 5) / total_pixels * 100:.1f}%",
            ],
        }
        st.dataframe(
            pd.DataFrame(risk_data),
            width="stretch",
            hide_index=True,
        )

with tab2:
    st.subheader("Weekly Hotspot Analysis")

    date_labels = []
    for week in range(6):
        end_day = enddate - timedelta(days=week * 5)
        start_day = end_day - timedelta(days=4)
        date_labels.append(
            f"{start_day.strftime('%d%b')}-{end_day.strftime('%d%b')}"
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
                        date_labels[week_idx],
                    )
                    st.plotly_chart(
                        fig,
                        width="stretch",
                        key=f"dhw_week_{week_idx}",
                    )
                else:
                    st.warning("⚠ No data available")

with tab3:
    st.subheader(
        f"Sea Surface Temperature - {enddate.strftime('%Y-%m-%d')}"
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
            title="Current Sea Surface Temperature",
        )
        st.plotly_chart(fig_sst, width="stretch")

    with col_right:
        st.markdown("**SST Statistics**")
        sst_stats = {
            "Metric": ["Mean", "Median", "Min", "Max", "Std Dev"],
            "Value (°C)": [
                f"{np.nanmean(sst_current):.2f}",
                f"{np.nanmedian(sst_current):.2f}",
                f"{np.nanmin(sst_current):.2f}",
                f"{np.nanmax(sst_current):.2f}",
                f"{np.nanstd(sst_current):.2f}",
            ],
        }
        st.dataframe(
            pd.DataFrame(sst_stats),
            width="stretch",
            hide_index=True,
        )

        fig_hist = go.Figure(
            data=go.Histogram(
                x=np.asarray(sst_current.values).flatten(),
                nbinsx=30,
                marker_color="rgb(55, 83, 109)",
            )
        )
        fig_hist.update_layout(
            title="SST Distribution",
            xaxis_title="Temperature (°C)",
            yaxis_title="Frequency",
            height=300,
        )
        st.plotly_chart(fig_hist, width="stretch")
