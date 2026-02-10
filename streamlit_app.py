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
st.sidebar.header("⚙️ Settings")

# Date input
target_date = st.sidebar.date_input(
    "Analysis Date",
    value=datetime(2019, 4, 20),
    help="Select the end date for DHW calculation"
)

# Simulation mode (since we removed file upload for simplicity)
st.sidebar.info("📝 Demo mode: Using simulated data. Upload real NetCDF files in production version.")

# Processing button
process_button = st.sidebar.button("🔄 Generate DHW Analysis", type="primary")

# NOAA OISST base URL pattern
NOAA_BASE_URL = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation-v2-1/access/oisst-avhrr-only-v2.1/"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_latest_sst(days_back=30):
    """Download latest 30 days of NOAA OISST data"""
    st.info(f"📡 Downloading latest {days_back} days from NOAA...")
    
    end_date = datetime.now().date()
    dates = []
    
    for i in range(days_back):
        date = end_date - timedelta(days=i)
        dates.append(date.strftime('%Y%m%d'))
    
    # Download files (first 5 days for demo, full 30 in production)
    sst_data = []
    for i, date_str in enumerate(dates[:5]):  # Limit to 5 for demo
        try:
            url = f"{NOAA_BASE_URL}{date_str}.nc"
            resp = requests.get(url, timeout=30)
            
            if resp.status_code == 200:
                with Dataset("temp.nc", "w", format="NETCDF4") as nc:
                    nc.createDimension("lat", 720)
                    nc.createDimension("lon", 1440)
                    nc.createDimension("time", 1)
                    lat = nc.createVariable("lat", "f4", ("lat",))
                    lon = nc.createVariable("lon", "f4", ("lon",))
                    sst_var = nc.createVariable("sst", "f4", ("time", "lat", "lon"))
                    
                    # Simulate SST data extraction (replace with actual parsing)
                    lon[:] = np.linspace(-179.875, 179.875, 1440)
                    lat[:] = np.linspace(89.875, -89.875, 720)
                    sst_var[0,:,:] = np.random.uniform(20, 30, (720, 1440))
                
                # Read the simulated data
                with Dataset("temp.nc") as nc:
                    sst = nc.variables['sst'][0,:,:]
                os.remove("temp.nc")
                sst_data.append(sst)
                st.success(f"✅ Downloaded {date_str}")
            else:
                st.warning(f"❌ {date_str} not found")
        except:
            st.warning(f"⚠️ Error downloading {date_str}")
    
    # Stack into time series
    TSeries = np.stack(sst_data, axis=2)
    return TSeries


# Coordinate data
@st.cache_data
def create_coordinates():
    """Create coordinate grid for Thai region"""
    lon = np.linspace(90, 110, 82)
    lat = np.linspace(0, 14.5, 60)
    LON, LAT = np.meshgrid(lon, lat)
    return LON, LAT, lon, lat

@st.cache_data
def generate_demo_data(seed=42):
    """Generate realistic demo SST data"""
    np.random.seed(seed)
    thlon, thlat, lon, lat = create_coordinates()
    
    # Create realistic baseline (warmer in Gulf, cooler in Andaman)
    baseline = 29.0 + 0.5 * np.sin((thlon - 90) / 20 * np.pi) + 0.3 * np.cos((thlat - 7) / 7 * np.pi)
    baseline += np.random.normal(0, 0.1, baseline.shape)
    
    # Generate time series with warming trend
    n_days = 30
    TSeries = np.zeros((60, 82, n_days))
    
    for day in range(n_days):
        # Add warming trend toward present
        warming = 0.5 * (day / n_days)
        # Add spatial variability (hotspots near coast)
        hotspot_mask = ((thlon > 99) & (thlon < 102) & (thlat > 7) & (thlat < 10)).astype(float)
        daily_sst = baseline + warming + 0.8 * hotspot_mask + np.random.normal(0, 0.2, baseline.shape)
        TSeries[:, :, day] = daily_sst
    
    return TSeries, baseline, thlon, thlat

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
    gulf_lon = [99.5, 101, 102, 102.5, 102, 100.5, 99.5, 99.5]
    gulf_lat = [6, 6.5, 8, 10, 12, 13.5, 12, 6]
    
    fig.add_trace(go.Scatter(
        x=gulf_lon, y=gulf_lat,
        fill='toself',
        fillcolor='rgba(180, 180, 180, 0.8)',
        line=dict(color='rgba(100, 100, 100, 1)', width=1),
        hoverinfo='skip',
        showlegend=False
    ))
    
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
    gulf_lon = [99.5, 101, 102, 102.5, 102, 100.5, 99.5, 99.5]
    gulf_lat = [6, 6.5, 8, 10, 12, 13.5, 12, 6]
    
    fig.add_trace(go.Scatter(
        x=gulf_lon, y=gulf_lat,
        fill='toself',
        fillcolor='rgba(180, 180, 180, 0.8)',
        line=dict(color='rgba(100, 100, 100, 1)', width=1),
        hoverinfo='skip',
        showlegend=False
    ))
    
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
    with st.spinner("Processing data..."):
        # Generate demo data
        TSeries, baseline, thlon, thlat = generate_demo_data()
        
        # Get coordinates
        LON, LAT, lon, lat = create_coordinates()
        
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
            st.subheader(f"Degree Heating Weeks - {target_date.strftime('%Y-%m-%d')}")
            
            # Main DHW map
            fig_dhw = create_dhw_map(lon, lat, dhw_total, 
                                     "Accumulated Degree Heating Weeks (6 weeks)", 6)
            
            # Add DHW level annotations
            st.markdown("""
            **DHW Alert Levels:**
            - 🔵 **0**: Below threshold (No stress)
            - ⚪ **1-2**: Watching (Possible stress)
            - 🟡 **3-4**: Alert Level 1-2 (Bleaching likely)
            - 🟠 **5**: Bleaching Level 1 (Significant bleaching expected)
            - 🔴 **6+**: Bleaching Level 2 (Severe/widespread bleaching expected)
            """)
            
            st.plotly_chart(fig_dhw, use_container_width=True)
            
            # DHW distribution
            col_a, col_b = st.columns(2)
            with col_a:
                dhw_flat = dhw_total.flatten()
                dhw_counts = pd.Series(dhw_flat).value_counts().sort_index()
                
                fig_dist = go.Figure(data=go.Bar(
                    x=dhw_counts.index,
                    y=dhw_counts.values,
                    marker_color=['#4270C2', '#D6D6D6', '#EBDEC4', '#E3CCD9', '#C98C59', '#A65959', '#8C4D1A'][:len(dhw_counts)]
                ))
                fig_dist.update_layout(
                    title="DHW Distribution",
                    xaxis_title="DHW Level",
                    yaxis_title="Number of Pixels",
                    height=300
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col_b:
                # Risk summary
                total_pixels = dhw_total.size
                risk_data = {
                    'Level': ['Safe (0)', 'Watch (1-2)', 'Alert (3-4)', 'Bleaching (5-6)'],
                    'Pixels': [
                        np.sum(dhw_total == 0),
                        np.sum((dhw_total >= 1) & (dhw_total <= 2)),
                        np.sum((dhw_total >= 3) & (dhw_total <= 4)),
                        np.sum(dhw_total >= 5)
                    ]
                }
                risk_df = pd.DataFrame(risk_data)
                risk_df['Percentage'] = (risk_df['Pixels'] / total_pixels * 100).round(1)
                
                st.markdown("**Risk Summary**")
                st.dataframe(risk_df, use_container_width=True, hide_index=True)
        
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
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader(f"Sea Surface Temperature - {target_date.strftime('%Y-%m-%d')}")
            
            # SST map
            fig_sst = create_sst_map(lon, lat, sst_current,
                                    "Current Sea Surface Temperature")
            st.plotly_chart(fig_sst, use_container_width=True)
            
            # Temperature statistics and distribution
            col_left, col_right = st.columns(2)
            
            with col_left:
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
                st.dataframe(pd.DataFrame(sst_stats), use_container_width=True, hide_index=True)
            
            with col_right:
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
                st.plotly_chart(fig_hist, use_container_width=True)

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
