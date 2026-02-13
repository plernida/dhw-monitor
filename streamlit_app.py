"""
DHW Dashboard - Streamlit Web App (Simplified - No External Files Required)
Deploy to Streamlit Community Cloud via GitHub
Interactive online interface for Degree Heating Weeks monitoring
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import requests
from netCDF4 import Dataset
import tempfile
import os
from io import BytesIO
import pytz
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

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
st.title("🌊 Degree Heating Weeks (DHW) Coral Bleaching Monitor")
st.markdown("""
Monitor sea surface temperature anomalies and coral bleaching risk in Thai waters.
Data source: GHRSST satellite observations (90-110°E, 0-14.5°N)
""")

# Sidebar controls
st.sidebar.header("⚙️ Auto Daily Update")

# Auto current date
th_tz = pytz.timezone('Asia/Bangkok')
now = datetime.now(th_tz)
target_date = now.date() - timedelta(days=2)

MIN_DATE = datetime(1981, 9, 1)
MAX_DATE = target_date

st.sidebar.success(f"📅 **Latest Analysis:** {target_date.strftime('%Y-%m-%d')}")
st.sidebar.info("✅ NOAA OISST v2.1: 1981-09-01 → present")

analysis_date = st.sidebar.date_input("🎯 Analysis Center Date",
    value=target_date,
    min_value=MIN_DATE,
    max_value=MAX_DATE,
    help="Select center date → auto 48-day backward analysis")


process_button = st.sidebar.button("🔄 Generate DHW Analysis", type="primary")



# NOAA OISST base URL pattern
#NOAA_BASE_URL = "https://www.ncei.noaa.gov/thredds/fileServer/OisstBase/NetCDF/V2.1/AVHRR/"
NOAA_NCSS_BASE = "https://www.ncei.noaa.gov/thredds/ncss/grid/OisstBase/NetCDF/V2.1/AVHRR/"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_latest_sst(enddate, days_back=48):
    sstdata = []
    th_lon = np.linspace(90, 110, 82)
    th_lat = np.linspace(0, 14.5, 60)

    PRELIM_WINDOW_DAYS = 14 # Relative to NOW (not enddate)
    now_date = datetime.now(pytz.timezone('Asia/Bangkok')).date()
    
    for i in range(days_back):
        target_date = enddate - timedelta(days=i)
        yyyymm = target_date.strftime('%Y%m')
        datestr = target_date.strftime('%Y%m%d')
        iso_date = target_date.strftime('%Y-%m-%d')
        
        # Preliminary: if target_date is within 14 days of CURRENT now_date
        if (now_date - target_date).days <= PRELIM_WINDOW_DAYS:
            filename = f"oisst-avhrr-v02r01.{datestr}_preliminary.nc"
        else:
            filename = f"oisst-avhrr-v02r01.{datestr}.nc"
        
        url = (
            f"{NOAA_NCSS_BASE}{yyyymm}/{filename}?"
            f"var=sst&north=14.500&west=90.000&east=110.000&south=0.000&"
            f"horizStride=1&time_start={iso_date}T12:00:00Z&time_end={iso_date}T12:00:00Z&"
            f"accept=netcdf3"
        )
        
        # Silent download + error handling (as before)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with Dataset('in-memory', mode='r', memory=resp.content) as nc:
                    sst_raw = nc.variables['sst'][0, :, :]  # (lat, lon)
                    subset_lat = nc.variables['lat'][:]
                    subset_lon = nc.variables['lon'][:]
                    
                    # Scale and mask
                    sst_scaled = np.where(sst_raw < 0, np.nan, sst_raw)
                    
                    # Interpolate to exact 60x82 gri
                    from scipy.interpolate import griddata
                    lon_grid, lat_grid = np.meshgrid(subset_lon, subset_lat)
                    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
                    values = sst_scaled.ravel()
                    
                    target_lon, target_lat = np.meshgrid(th_lon, th_lat)
                    sst_interp = griddata(points, values, (target_lon, target_lat), method='linear')
                    # Interp to grid...
                    sstdata.append(sst_interp)
            else:
                sstdata.append(np.full((60, 82), np.nan))
        except:
            sstdata.append(np.full((60, 82), np.nan))
            
    return np.stack(sstdata, axis=2)


# Coordinate data
@st.cache_data
def create_coordinates():
    """Create coordinate grid for Thai region"""
    lon = np.linspace(90, 110, 82)
    lat = np.linspace(0, 14.5, 60)
    LON, LAT = np.meshgrid(lon, lat)
    return LON, LAT, lon, lat


def calculate_dhw(TSeries, baseline, threshold=1.0):
    """Calculate Degree Heating Weeks from time series"""
    dhw_weeks = []
    sst_weeks = []
    
    for week in range(6):
        start_idx = (5 - week) * 5
        end_idx = start_idx + 5
        week_mean = np.nanmean(TSeries[:, :, start_idx:end_idx], axis=2)
        sst_weeks.append(week_mean)
        hotspot = week_mean - (baseline + threshold)
        dhw_week = np.where(hotspot > 0, 1, 0)
        dhw_weeks.append(dhw_week)
    
    # Sum all weeks
    dhw_total = sum(dhw_weeks)
    return dhw_weeks, dhw_total, sst_weeks

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
            [0.17, 'rgb(214, 214, 214)'],  # Gray - 1
            [0.33, 'rgb(235, 222, 196)'],  # Beige - 2
            [0.5, 'rgb(227, 204, 217)'],   # Pink - 3
            [0.67, 'rgb(201, 140, 89)'],   # Brown - 4
            [0.83, 'rgb(166, 89, 89)'],    # Dark brown - 5
            [1, 'rgb(140, 77, 26)']        # Very dark - 6
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

def create_dhw_map_mapbox(lon, lat, dhw_data, title):
    lon2d, lat2d = np.meshgrid(lon, lat)

    fig = go.Figure(go.Densitymapbox(
        lon=lon2d.flatten(),
        lat=lat2d.flatten(),
        z=dhw_data.flatten(),
        radius=15,
        colorscale=[
            [0.0, "rgb(66,112,194)"],
            [0.3, "rgb(214,214,214)"],
            [0.5, "rgb(235,222,196)"],
            [0.7, "rgb(201,140,89)"],
            [1.0, "rgb(140,77,26)"],
        ],
        zmin=0,
        zmax=6,
        colorbar=dict(title="DHW (weeks)")
    ))

    fig.update_layout(
        title=title,
        mapbox=dict(
            style="carto-positron",   # 👈 land + coastlines
            center=dict(lat=7.5, lon=100),
            zoom=4.3
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=800
    )

    return fig
    
def create_sst_map_mapbox(lon, lat, sst_data, title):
    lon2d, lat2d = np.meshgrid(lon, lat)

    fig = go.Figure(go.Densitymapbox(
        lon=lon2d.flatten(),
        lat=lat2d.flatten(),
        z=sst_data.flatten(),
        radius=12,
        colorscale="jet",
        zmin=28,
        zmax=32,
        colorbar=dict(title="SST (°C)")
    ))

    fig.update_layout(
        title=title,
        mapbox=dict(
            style="carto-positron",  # 🌄 topography
            center=dict(lat=7.5, lon=100),
            zoom=4.3
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=600
    )

    return fig

def create_sst_map(lon, lat, sst_data, title):
    """Create Plotly contour map for SST data"""
    fig = go.Figure(data=go.Contour(
        z=sst_data,
        x=lon,
        y=lat,
        colorscale='jet',
        contours=dict(
            start=28,
            end=32,
            size=0.5,
        ),
        colorbar=dict(
            title='SST (°C)',
            tickmode='linear',
            tick0=28,
            dtick=0.5
        ),
        hovertemplate='Lon: %{x:.2f}°E<br>Lat: %{y:.2f}°N<br>SST: %{z:.2f}°C<extra></extra>'
    ))
    
    # Add land
    #gulf_lon = [99.5, 101, 102, 102.5, 102, 100.5, 99.5, 99.5]
    #gulf_lat = [6, 6.5, 8, 10, 12, 13.5, 12, 6]
    

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title='Longitude (°E)',
        yaxis_title='Latitude (°N)',
        height=600,
        hovermode='closest',
        plot_bgcolor='rgba(240,245,250,1)',
        xaxis=dict(range=[90, 110]),
        yaxis=dict(range=[0, 14.5])
    )

    return fig

# Main processing
if process_button:
    with st.spinner('Processing DHW analysis...'):
        # Use SELECTED date as analysis center
        enddate = analysis_date
        
        # Download 48 days BACK from analysis_date
        TSeries = download_latest_sst(enddate, days_back=48)
        # Get coordinates
        LON, LAT, lon, lat = create_coordinates()

        # baseline
        baseline = 29.0 + 0.5 * np.sin((LON - 90) / 20 * np.pi) + 0.3 * np.cos((LAT - 7) / 7 * np.pi)  # Keep your existing baseline logic
        baseline += np.random.normal(0, 0.1, baseline.shape)  # Add noise if desired

        # Calculate DHW
        dhw_weeks, dhw_total, sst_weeks = calculate_dhw(TSeries, baseline)
        
        # Current SST
        sst_current = TSeries[:, :, -1]
        
        # Success message
        st.success("✅ Data processed successfully!")
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Max DHW", f"{int(np.nanmax(dhw_total))} weeks")
        with col2:
            st.metric("Avg SST", f"{np.nanmean(sst_current):.1f}°C")
        with col3:
            alert_area = np.sum(dhw_total >= 4) / dhw_total.size * 100
            st.metric("Alert Area", f"{alert_area:.1f}%")
        with col4:
            bleaching_area = np.sum(dhw_total >= 5) / dhw_total.size * 100
            st.metric("Bleaching Risk", f"{bleaching_area:.1f}%", delta=f"{bleaching_area:.1f}%", delta_color="inverse")
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Accumulated DHW", "🗓️ Weekly Hotspots", "🌡️ Current SST"])
        
        with tab1:
            st.subheader(f"Degree Heating Weeks - {enddate.strftime('%Y-%m-%d')}")
            
            # NEW LAYOUT: Portrait map LEFT + distribution/stats RIGHT
            col_left, col_right = st.columns([60, 40])
            
            with col_left:
                # Portrait DHW map (tall)
                fig_dhw = create_dhw_map_mapbox(
                    lon, lat, dhw_total,
                    "Accumulated DHW (6 weeks)"
                )
                st.plotly_chart(fig_dhw, use_container_width=True)
                            
            with col_right:
                # Upper right: DHW Distribution
                st.markdown("**📊 DHW Distribution**")
                dhw_flat = dhw_total.flatten()
                dhw_counts = pd.Series(dhw_flat).value_counts().sort_index()
                
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
            
            # Alert levels legend (below everything)
            st.markdown("""
            **DHW Alert Levels:**
            - 🔵 **0**: No stress
            - ⚪ **1-2**: Watch (possible stress)
            - 🟡 **3-4**: Alert (bleaching likely)
            - 🟠 **5**: Bleaching Level 1
            - 🔴 **6+**: Severe bleaching expected
            """)
        
        with tab2:
            st.subheader("Weekly Hotspot Analysis")
            
            # Calculate date labels
            date_labels = []
            for week in range(6):
                end_day = target_date - timedelta(days=week*5)
                start_day = end_day - timedelta(days=4)
                date_labels.append(f"{start_day.strftime('%d%b')}-{end_day.strftime('%d%b')}")
            
            # Display in 2 rows
            for row in range(2):
                cols = st.columns(3)
                for col_idx in range(3):
                    week_idx = row * 3 + col_idx
                    with cols[col_idx]:
                        fig = create_dhw_map(lon, lat, dhw_weeks[week_idx],
                                           date_labels[week_idx], 2)
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, width='stretch')
        
        with tab3:
            st.subheader(f"Sea Surface Temperature - {enddate.strftime('%Y-%m-%d')}")
            col_left, col_right = st.columns([60,40])
            with col_left:
                # SST map
                fig_sst = create_sst_map_mapbox(lon, lat, sst_current,
                                        "Current Sea Surface Temperature")
                fig_sst.update_layout(height=800, margin=dict(l=50,r=20, t=50, b=50))
                st.plotly_chart(fig_sst, width='stretch')
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

else:
    # Landing page
    st.info("👈 Click 'Generate DHW Analysis' in the sidebar to begin")
    
    # Instructions
    with st.expander("📖 How to Use This Dashboard"):
        st.markdown("""
        ### Quick Start
        1. Select your analysis date in the sidebar
        2. Click **"Generate DHW Analysis"** button
        3. Explore the three tabs:
           - **Accumulated DHW**: See 6-week cumulative heat stress
           - **Weekly Hotspots**: Week-by-week analysis
           - **Current SST**: Temperature field and statistics
        
        ### Understanding DHW
        **Degree Heating Weeks (DHW)** measures accumulated thermal stress on coral reefs:
        - Each week where SST exceeds the climatological maximum (MMM + 1°C) adds 1 DHW
        - Accumulated over 6 weeks (30 days)
        - Critical thresholds:
          - **0-2 weeks**: Low risk
          - **3-4 weeks**: Bleaching Alert
          - **5-6+ weeks**: Severe bleaching expected
        
        ### Data Source
        - **Region**: Thai waters (90-110°E, 0-14.5°N)
        - **Resolution**: ~0.25° (~25km)
        - **Demo Mode**: Currently showing simulated data for demonstration
        
        ### For Production Use
        Contact the developer to integrate your actual GHRSST NetCDF files.
        """)
    
    # Sample visualization
    with st.expander("🔬 About Coral Bleaching"):
        st.markdown("""
        ### What is Coral Bleaching?
        Coral bleaching occurs when water is too warm, causing corals to expel the symbiotic algae 
        (zooxanthellae) living in their tissues, turning them white.
        
        ### Why DHW Matters
        - **Early Warning**: DHW provides advance notice of bleaching risk
        - **Spatial Coverage**: Identifies regional hotspots
        - **Management Tool**: Helps reef managers plan interventions
        
        ### Temperature Thresholds
        - **MMM (Maximum Monthly Mean)**: Warmest month average
        - **MMM + 1°C**: Bleaching threshold
        - **Sustained exposure**: Multiple weeks above threshold causes severe damage
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>DHW Coral Bleaching Monitor | GHRSST Satellite Data | Built with Streamlit & Plotly</small><br>
    <small>🌊 For coral reef conservation and marine science 🐠</small>
</div>
""", unsafe_allow_html=True)
