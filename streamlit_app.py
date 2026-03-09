"""
DHW Dashboard - Streamlit Web App (Simplified - No External Files Required)
Deploy to Streamlit Community Cloud via GitHub
Interactive online interface for Degree Heating Weeks monitoring
"""

import streamlit as st
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta
import requests
from netCDF4 import Dataset
import tempfile
import os
import json
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
from matplotlib.colors import ListedColormap
from io import BytesIO
import pytz
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

font_path = "static/Kanit-Regular.ttf"

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = ['Kanit','DejaVu Sans','Arial']

#coast_gdf = gpd.read_file("ne_10m_coastline.shp").to_crs('EPSG:4326')  # Ensure CRS is EPSG:4326
#coast_geojson = coast_gdf.__geo_interface__
cmap_full = plt.get_cmap('Spectral_r')#nipy_spectral
slice_start, slice_end = 0, 0.9
colors = cmap_full(np.linspace(slice_start, slice_end, 256))
spectral_slice = LinearSegmentedColormap.from_list('spectral_slice', colors)#'nipy_yellow_red
colors_rgb = [
    '#C8FAFA',    # Blue
    '#FFF000',   # Gray
    '#FAAA0A',   # Beige
    '#F00000',   # Pink
    '#960000',    # Brown
    '#A05024',     # Dark brown
    '#F000F0'      # Dark brown
]
cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors_rgb, N=7)


# Page configuration
st.set_page_config(
    page_title="DHW Coral Bleaching Monitor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# Title and description
st.title("🌊 แผนที่อุณหภูมิน้ำทะเล Degree Heating Weeks (DHW) Coral Bleaching Monitor")
st.markdown("""
ติดตามอุณหภูมิน้ำทะเลที่ส่งผลต่อการฟอกขาวของปะการัง Monitor sea surface temperature anomalies and coral bleaching risk in Thai waters.
Data source: GHRSST satellite observations (90-110°E, 0-14.5°N)
""")

# Sidebar controls
st.sidebar.header("⚙️ Auto Daily Update")
# Auto current date
th_tz = pytz.timezone('Asia/Bangkok')
now = datetime.now(th_tz)
target_date = now.date() - timedelta(days=2)


MIN_DATE = datetime(1985, 1, 1)
MAX_DATE = target_date

st.sidebar.success(f"📅 **Latest Analysis:** {target_date.strftime('%Y-%m-%d')}")
st.sidebar.info("✅ CRW SST 5km: 1985-01-01 → present")

analysis_date = st.sidebar.date_input("🎯 Analysis Center Date",
    value=target_date,
    min_value=MIN_DATE,
    max_value=MAX_DATE,
    help="Select center date → auto 30-day backward analysis")


process_button = st.sidebar.button("🔄 Generate DHW Analysis", type="primary")



# NOAA OISST base URL pattern
#NOAA_BASE_URL = "https://www.ncei.noaa.gov/thredds/fileServer/OisstBase/NetCDF/V2.1/AVHRR/"
#NOAA_NCSS_BASE = "https://www.ncei.noaa.gov/thredds/ncss/grid/OisstBase/NetCDF/V2.1/AVHRR/"
CRW_ERDDAP_BASE = "https://coastwatch.noaa.gov/erddap/griddap/noaacrwsstDaily"
dayback=30
@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_latest_sst(enddate, days_back=30):

    thtz = pytz.timezone('Asia/Bangkok')
    now_date = datetime.now(thtz).date()

    # CRW usually lags ~2 days
    latest_available = now_date - timedelta(days=2)

    if enddate > latest_available:
        enddate = latest_available

    start_date = enddate - timedelta(days=days_back - 1)

    start_time = start_date.strftime('%Y-%m-%dT12:00:00Z')
    end_time = enddate.strftime('%Y-%m-%dT12:00:00Z')

    url = (
        f"{CRW_ERDDAP_BASE}.nc?"
        f"analysed_sst"
        f"[({start_time}):1:({end_time})]"
        f"[(0.025):1:(14.075)]"
        f"[(90.025):1:(110.025)]"
    )

    #print("Downloading:", url)

    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()

    local_file = "latest_sst.nc"

    with open(local_file, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)

    #print("Download complete")

    ds = xr.open_dataset(local_file)

    # convert Kelvin → Celsius
    #ds["analysed_sst"] = ds["analysed_sst"] - 273.15

    # rename to match AVHRR variable naming if needed
    ds = ds.rename({
        "latitude": "lat",
        "longitude": "lon",
        "analysed_sst": "sst"
    })

    # reorder dimensions to match your DHW code
    ds = ds.transpose("lat", "lon", "time")
    ds = ds.sel(lon=slice(90,110))

    sst_stack = ds["sst"].values
    lat_ref = ds["lat"].values
    lon_ref = ds["lon"].values
    time_list = ds["time"].values

    return sst_stack, time_list, lat_ref, lon_ref



# Coordinate data
@st.cache_data
def create_coordinates():
    """Create coordinate grid for Thai region"""
    lon = np.linspace(90.025, 109.975, 400)
    lat = np.linspace(0.025, 14.075, 282)
    LON, LAT = np.meshgrid(lon, lat)
    return LON, LAT, lon, lat


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
#colors_rgb = [
#    (66/255, 112/255, 194/255),    # Blue
#    (214/255, 214/255, 214/255),   # Gray
#    (235/255, 222/255, 196/255),   # Beige
#    (227/255, 204/255, 217/255),   # Pink
#    (201/255, 140/255, 89/255),    # Brown
#    (166/255, 89/255, 89/255),     # Dark brown
#    (140/255, 77/255, 26/255)      # Dark brown
#]

# Create custom colormap (N=256 for smooth gradient)
#cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors_rgb, N=256)

def create_dhw_map(lon, lat, dhw_data, title, levels):
    """Create Plotly contour map for DHW data"""
    if levels == 2:  # Binary (0/1)
        colorscale = [[0, 'white'], [1, 'rgb(102, 204, 204)']]
        colorbar_title = "Hotspot"
        tickvals = [0, 1]
        ticktext = ['No', 'Yes']
    else:  # Multi-level (0-6)
        colorscale = [
            [0, 'rgb(66, 112, 194)'],      # Blue - 0
            [1, 'rgb(214, 214, 214)'],  # Gray - 1
            [2, 'rgb(235, 222, 196)'],  # Beige - 2
            [3, 'rgb(227, 204, 217)'],   # Pink - 3
            [4, 'rgb(201, 140, 89)'],   # Brown - 4
            [5, 'rgb(166, 89, 89)'],    # Dark brown - 5
            [6, 'rgb(140, 77, 26)']        # Very dark - 6
        ]
        colorbar_title = "DHW Level"
        tickvals = list(range(7))
        ticktext = ['0', '1', '2', '3', '4', '5', '6+']
    
    fig = go.Figure(data=go.Contour(
        z=dhw_data,
        x=lon,
        y=lat,
        colorscale=colorscale,
        contours=dict(
            start=0,
            end=levels,
            size=1,
        ),
        colorbar=dict(
            title=colorbar_title,
            tickvals=tickvals,
            ticktext=ticktext
        ),
        hovertemplate='Lon: %{x:.2f}°E<br>Lat: %{y:.2f}°N<br>Value: %{z}<extra></extra>'
    ))
    
    # Add land boundary (simplified Thailand outline)
    # Gulf of Thailand
    #gulf_lon = [99.5, 101, 102, 102.5, 102, 100.5, 99.5, 99.5]
    #gulf_lat = [6, 6.5, 8, 10, 12, 13.5, 12, 6]
    

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title='Longitude (°E)',
        yaxis_title='Latitude (°N)',
        height=500,
        hovermode='closest',
        plot_bgcolor='rgba(240,245,250,1)',
        xaxis=dict(range=[90, 110]),
        yaxis=dict(range=[0, 14.5])
    )

    return fig
def plot_dhw_week(lon, lat, dhw_total, title):
    lon2d, lat2d = np.meshgrid(lon, lat)
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # DHW raster
    im = ax.contourf(
        lon2d, lat2d, dhw_total,
        cmap=ListedColormap(colors_rgb[0:2]), levels=2,
        vmin=0, vmax=1,
        transform=ccrs.PlateCarree()
    )
    ax.set_extent([91, 110, 1, 14])
    #ax.set_xlabel('Longitude (°E)')
    #ax.set_ylabel('Latitude (°N)')
    
        # Coastlines
    #ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.LAND, facecolor='lightgray',zorder=3,edgecolor='black',lw=0.5)
    #cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
    #cbar.set_label('DHW (weeks)', fontsize=12)
    
    ax.set_xticks(np.arange(92,111,2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(2,16,2), crs=ccrs.PlateCarree())
    #ax.coastlines('10m',zorder=3,lw=0.3)
    
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(which='both',labeltop=True, labelright=True,labelleft=True,width=0.8,
                  bottom=True,top=True,right=True,labelsize=15,grid_color='black',grid_linewidth=0.5)
    # Custom legend patches + labels matching your markdown
    legend_elements = [
        mpatches.Patch(color=colors_rgb[0], label='No stress'),
        mpatches.Patch(color=colors_rgb[1], label='Watch')]

    ax.legend(handles=legend_elements,ncol=5,  # Horizontal (5 columns)
           loc='upper center', 
           bbox_to_anchor=(0.5, -0.05),
          fontsize=20, frameon=True, fancybox=True, shadow=True)
    ax.set_title(title, fontsize=20)
    plt.tight_layout()
    #plt.savefig(filename, dpi=150, bbox_inches='tight')
    return fig
    
def plot_cartopy_map(lon, lat, dhw_total, title):

    lon2d, lat2d = np.meshgrid(lon, lat)
      # sample DHW

    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # DHW raster
    im = ax.contourf(
        lon2d, lat2d, dhw_total,
        cmap=cmap, levels=6,
        vmin=0, vmax=6,
        transform=ccrs.PlateCarree()
    )
    ax.set_extent([91, 110, 1, 14])
    #ax.set_xlabel('Longitude (°E)')
    #ax.set_ylabel('Latitude (°N)')
    
        # Coastlines
    #ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.LAND, facecolor='lightgray',zorder=3,edgecolor='black',lw=0.5)
    #cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
    #cbar.set_label('DHW (weeks)', fontsize=12)
    
    ax.set_xticks(np.arange(92,111,2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(2,16,2), crs=ccrs.PlateCarree())
    #ax.coastlines('10m',zorder=3,lw=0.3)
    
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(which='both',labeltop=True, labelright=True,labelleft=True,width=0.8,
                  bottom=True,top=True,right=True,labelsize=10,grid_color='black',grid_linewidth=0.5)
    # Custom legend patches + labels matching your markdown
    ax.annotate(f"Daily  \n{title[7:17]}",xy=(1, 1), xycoords='axes fraction',fontsize=10,fontweight='bold',
                xytext=(-25,-10), textcoords='offset points',
                ha='right', va='top')
    legend_elements = [
        mpatches.Patch(color=colors_rgb[0], label='No stress'),
        mpatches.Patch(color=colors_rgb[1], label='Watch'),
        mpatches.Patch(color=colors_rgb[2], label='Warning'),
        mpatches.Patch(color=colors_rgb[3], label='Alert 1'),
        mpatches.Patch(color=colors_rgb[4], label='Al 2'),
        mpatches.Patch(color=colors_rgb[5], label='Al 3'),
        mpatches.Patch(color=colors_rgb[6], label='Al 4')# Use darkest for 6+
    ]
    ax.legend(handles=legend_elements,ncol=7,  # Horizontal (5 columns)
           loc='upper center', 
           bbox_to_anchor=(0.5, -0.05),
          fontsize=8, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    return fig    

def create_sst_map_mapbox(lon, lat, sstdata, title):
    lon2d, lat2d = np.meshgrid(lon, lat)    
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.contourf(lon2d, lat2d, sstdata,
                     cmap=spectral_slice,levels=np.linspace(24,34,21),
                     extend='neither',
                     transform=ccrs.PlateCarree()
                    )
    #im.set_clim(24, 34)
    ax.set_extent([91, 110, 1, 14])
    #ax.set_xlabel('Longitude (°E)')
    #ax.set_ylabel('Latitude (°N)')
    
        # Coastlines
    #ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.LAND, facecolor='lightgray',zorder=3,edgecolor='black',lw=0.5)
    #cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
    #cbar.set_label('DHW (weeks)', fontsize=12)
    
    ax.set_xticks(np.arange(92,111,2), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(2,16,2), crs=ccrs.PlateCarree())
    #ax.coastlines('10m',zorder=3,lw=0.3)
    
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(which='both',labeltop=True, labelright=True,labelleft=True,width=0.8,
                  bottom=True,top=True,right=True,labelsize=8,grid_color='black',grid_linewidth=0.5)
    cbar=fig.colorbar(im,ax=ax,orientation='horizontal', shrink=0.8, pad=0.05)
    cbar.set_ticks(np.arange(24,34.1,1))
    cbar.set_label('°C',fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    ax.annotate(f"Sea Surface Temperatures \n{title[7:17]}",xy=(1, 1), xycoords='axes fraction',fontsize=15,fontweight='bold',
            xytext=(-25,-10), textcoords='offset points',
            ha='right', va='top')
    #cbar.mappable.set_clim(23, 35)
    plt.tight_layout()


    return fig

def update_bleaching_history(date, value):

    os.makedirs("static", exist_ok=True)
    filepath = "static/bleaching_history.json"

    # โหลด history เดิม
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            history = json.load(f)
    else:
        history = {}

    # บันทึกค่าของวันนี้
    history[date.strftime("%Y-%m-%d")] = float(value)

    # save กลับ
    with open(filepath, "w") as f:
        json.dump(history, f, indent=2)

def get_previous_bleaching(date):

    filepath = "static/bleaching_history.json"

    if not os.path.exists(filepath):
        return None

    with open(filepath, "r") as f:
        history = json.load(f)

    yesterday = (date - timedelta(days=1)).strftime("%Y-%m-%d")

    return history.get(yesterday)
    
# Main processing
#if process_button:
with st.spinner('Processing DHW analysis...'):
    # Use SELECTED date as analysis center

    enddate = analysis_date
    # Download 48 days BACK from analysis_date
    TSeries, time_list, lat_ref, lon_ref = download_latest_sst(enddate, days_back=30)
    # Get coordinates
    LON, LAT, lon, lat = create_coordinates()

    # Check for pre-generated PNGs (from daily Actions)
    datedhw_png = f"static/{enddate.strftime('%Y-%m-%d')}_dhw.png"
    datesst_png = f"static/{enddate.strftime('%Y-%m-%d')}_sst.png"
    
    baseline = xr.open_dataset('crw_mmm_sst_thailand_1985-2025.nc') # read array
    MMM = baseline['sst'].sel(lon=slice(90,110),lat=slice(14.1,0))

    # Calculate DHW
    dhw_weeks, dhw_total, sst_weeks = calculate_dhw(TSeries, MMM)
    #dhw_weeks = xr.DataArray(dhw_weeks, dims=('week', 'lat', 'lon'))
    #sst_weeks = xr.DataArray(sst_weeks, dims=('week', 'lat', 'lon'))
    # Current SST
    sst_current = TSeries[:, :, -1]
    
    # Success message
    st.success("✅ Data processed successfully!")

    
   
    # Display statistics

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Max DHW", f"{(dhw_total.max().values)} weeks")
    with col2:
        st.metric("Avg SST", f"{float(np.nanmean(sst_current)):.2f} °C")
    with col3:
        alert_area = xr.where(dhw_total>=4,1,0).sum() / dhw_total.size * 100
        st.metric("Alert Area", f"{alert_area:.1f}%")
    with col4:
        bleaching_area = xr.where(dhw_total >= 5, 1, 0).sum() / dhw_total.size * 100
        previous_bleaching = get_previous_bleaching(enddate)
        if previous_bleaching is not None:
            delta_bleaching = bleaching_area - previous_bleaching
        else:
            delta_bleaching = 0
        st.metric("Bleaching Risk", f"{bleaching_area:.1f}%", delta=f"{delta_bleaching:.1f}%", delta_color="inverse")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Accumulated DHW", "🗓️ Weekly Hotspots", "🌡️ Current SST"])
    
    with tab1:
        st.subheader(f"Degree Heating Weeks - {enddate.strftime('%Y-%m-%d')}")
   

        # NEW LAYOUT: Portrait map LEFT + distribution/stats RIGHT
        col_left, col_right = st.columns([80, 20])
        
        with col_left:
            if os.path.exists(datedhw_png):
                #st.success(f"✅ Using cached DHW PNG for {enddate.strftime('%Y-%m-%d')}")
                st.image(datedhw_png, caption="", width="stretch")
            else:
                #st.info("⚠️ No cached PNG found. Computing live...")
            # Portrait DHW map (tall)
                fig_dhw = st.pyplot(plot_cartopy_map(
                    lon, lat, dhw_total,
                    f"static/{enddate}_dhw.png"
                ))
            #st.plotly_chart(fig_dhw, width='stretch')
                        
        with col_right:
            # Upper right: DHW Distribution
            st.markdown("**📊 DHW Distribution**")
            dhw_flat = dhw_total.values.flatten()   
            dhw_counts = pd.Series(dhw_flat).value_counts().sort_index().reindex(range(7), fill_value=0)
            
            fig_dist = go.Figure(data=go.Bar(
                x=dhw_counts.index,
                y=dhw_counts.values,
                marker_color=['#4270C2','#D6D6D6','#EBDEC4','#E3CCD9','#C98C59','#A65959','#8C4D1A']
            ))
            fig_dist.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                title="Distribution by Level"
            )
            st.plotly_chart(fig_dist, width='stretch')
            
            # Lower right: Risk Summary
            st.markdown("**⚠️ Risk Summary**")
            total_pixels = dhw_total.size
            risk_data = {
                'Alert Level': ['Safe (0)', 'Watch (1-2)', 'Alert (3-4)', 'Bleaching (≥5)'],
                'Pixels': [
                    int(np.sum(dhw_total == 0)),
                    int(np.sum((dhw_total >= 1) & (dhw_total <= 2))),
                    int(np.sum((dhw_total >= 3) & (dhw_total <= 4))),
                    int(np.sum(dhw_total >= 5))
                ],
                '% Area': [
                    f"{np.sum(dhw_total == 0)/total_pixels*100:.1f}%",
                    f"{np.sum((dhw_total >= 1) & (dhw_total <= 2))/total_pixels*100:.1f}%",
                    f"{np.sum((dhw_total >= 3) & (dhw_total <= 4))/total_pixels*100:.1f}%",
                    f"{np.sum(dhw_total >= 5)/total_pixels*100:.1f}%"
                ]
            }
            risk_df = pd.DataFrame(risk_data)
            st.dataframe(risk_df, width='stretch', hide_index=True)
   
    with tab2:
        st.subheader("Weekly Hotspot Analysis")
        
        date_labels = []
        datestr = enddate.strftime('%Y-%m-%d')


        for week in range(6):
            end_day = enddate - timedelta(days=week*5)
            start_day = end_day - timedelta(days=4)
            date_labels.append(f"{start_day.strftime('%d%b')}-{end_day.strftime('%d%b')}")
        static_paths = [f"static/{datestr}_week_{i+1:02d}.png" for i in range(6)]
        
        

        for row in range(2):
            cols = st.columns(3)
            for col_idx in range(3):
                week_idx = row * 3 + col_idx
    
                with cols[col_idx]:
    
                    # กรณีมีไฟล์ PNG
                    if week_idx < len(static_paths) and os.path.exists(static_paths[week_idx]):
                        st.image(
                            static_paths[week_idx],
                            caption="",#date_labels[week_idx],
                            width="stretch"
                        )
                        
                    # กรณีไม่มีไฟล์ → plot สด
                    elif week_idx < len(dhw_weeks):
                        fig = plot_dhw_week(
                            lon,
                            lat,
                            dhw_weeks[week_idx],
                            date_labels[week_idx]
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                        
    
                    else:
                        st.warning("⚠ No data available")

            
    with tab3:
        st.subheader(f"Sea Surface Temperature - {enddate.strftime('%Y-%m-%d')}")
        col_left, col_right = st.columns([80, 20])
        with col_left:
            if os.path.exists(datesst_png):
                #st.success(f"✅ Using cached SST PNG for {enddate.strftime('%Y-%m-%d')}")
                st.image(datesst_png, caption="", width="stretch")
            else:
                #st.info("⚠️ No cached PNG found. Computing live...")
            # SST map
                fig_sst = st.pyplot(create_sst_map_mapbox(lon, lat, sst_current,
                                        f"static/{enddate}_sst.png"))
            #fig_sst.update_layout(height=800, margin=dict(l=50,r=20, t=50, b=50))
            #st.plotly_chart(fig_sst, width='stretch')
        with col_right:    
        # Temperature statistics and distribution
            st.markdown("**SST Statistics**")
            sst_stats = {
                'Metric': ['Mean', 'Median', 'Min', 'Max', 'Std Dev'],
                'Value (°C)': [
                    f"{np.nanmean(sst_current):.2f}",
                    f"{np.nanmedian(sst_current):.2f}",
                    f"{np.nanmin(sst_current):.2f}",
                    f"{np.nanmax(sst_current):.2f}",
                    f"{np.nanstd(sst_current):.2f}"
                ]
            }
            st.dataframe(pd.DataFrame(sst_stats), width='stretch', hide_index=True)
        
        
            # Temperature distribution
            fig_hist = go.Figure(data=go.Histogram(
                x=sst_current.flatten(),
                nbinsx=30,
                marker_color='rgb(55, 83, 109)'
            ))
            fig_hist.update_layout(
                title="SST Distribution",
                xaxis_title='Temperature (°C)',
                yaxis_title='Frequency',
                height=300
            )
            st.plotly_chart(fig_hist, width='stretch')


