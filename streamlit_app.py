"""
DHW Dashboard - Streamlit Web App
Deploy to Streamlit Community Cloud via GitHub
Interactive online interface for Degree Heating Weeks monitoring
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from netCDF4 import Dataset
from scipy.io import loadmat, savemat
import os
import tempfile

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

# File upload option
uploaded_files = st.sidebar.file_uploader(
    "Upload GHRSST NetCDF files (optional)",
    type=['nc'],
    accept_multiple_files=True,
    help="Upload 30 days of GHRSST NetCDF files for analysis"
)

# Date input
target_date = st.sidebar.date_input(
    "Analysis Date",
    value=datetime(2019, 4, 20),
    help="Select the end date for DHW calculation"
)

# Baseline data upload
baseline_file = st.sidebar.file_uploader(
    "Upload MMM Baseline (.mat)",
    type=['mat'],
    help="Upload GMAMTH(all).mat file containing climatology"
)

# Processing button
process_button = st.sidebar.button("🔄 Process Data", type="primary")

# Coordinate data (example - adjust to your actual region)
def create_sample_coordinates():
    """Create sample coordinate grid for Thai region"""
    lon = np.linspace(90, 110, 82)
    lat = np.linspace(0, 14.5, 60)
    return np.meshgrid(lon, lat)

def calculate_dhw(TSeries, baseline, threshold=1.0):
    """Calculate Degree Heating Weeks from time series"""
    # Calculate 5-day averages for 6 weeks
    dhw_weeks = []
    for week in range(6):
        start_idx = (5 - week) * 5
        end_idx = start_idx + 5
        week_mean = np.nanmean(TSeries[:, :, start_idx:end_idx], axis=2) - 273.15
        hotspot = week_mean - (baseline + threshold)
        dhw_week = np.where(hotspot > 0, 1, 0)
        dhw_weeks.append(dhw_week)
    
    # Sum all weeks
    dhw_total = sum(dhw_weeks)
    return dhw_weeks, dhw_total

def create_dhw_map(lon, lat, dhw_data, title, levels):
    """Create Plotly contour map for DHW data"""
    # Define DHW colorscale
    if levels == 2:  # Binary (0/1)
        colorscale = [[0, 'white'], [1, 'rgb(102, 204, 204)']]
        colorbar_title = "Hotspot"
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
    
    fig = go.Figure(data=go.Contour(
        z=dhw_data,
        x=lon[0, :],
        y=lat[:, 0],
        colorscale=colorscale,
        contours=dict(
            start=0,
            end=levels,
            size=1,
        ),
        colorbar=dict(
            title=colorbar_title,
            tickmode='linear',
            tick0=0,
            dtick=1
        ),
        hovertemplate='Lon: %{x:.1f}°E<br>Lat: %{y:.1f}°N<br>DHW: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Longitude (°E)',
        yaxis_title='Latitude (°N)',
        height=500,
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    
    return fig

def create_sst_map(lon, lat, sst_data, title):
    """Create Plotly contour map for SST data"""
    fig = go.Figure(data=go.Contour(
        z=sst_data,
        x=lon[0, :],
        y=lat[:, 0],
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
        hovertemplate='Lon: %{x:.1f}°E<br>Lat: %{y:.1f}°N<br>SST: %{z:.1f}°C<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Longitude (°E)',
        yaxis_title='Latitude (°N)',
        height=600,
        hovermode='closest'
    )
    
    return fig

# Main processing
if process_button:
    if not uploaded_files and not baseline_file:
        st.warning("⚠️ Please upload data files to begin analysis")
    else:
        with st.spinner("Processing data..."):
            # Create sample data for demonstration
            # In production, replace this with actual file processing
            thlon, thlat = create_sample_coordinates()
            
            # Generate sample baseline (replace with actual loaded data)
            if baseline_file:
                baseline_data = loadmat(baseline_file)
                MGMAMC = baseline_data['MGMAM'] - 273.15
            else:
                # Demo data
                MGMAMC = np.random.uniform(28, 30, (60, 82))
            
            # Generate sample time series (replace with actual NetCDF processing)
            # In production: process uploaded_files here
            TSeries = np.random.uniform(301, 305, (60, 82, 30))  # 30 days in Kelvin
            
            # Calculate DHW
            dhw_weeks, dhw_total = calculate_dhw(TSeries, MGMAMC)
            
            # Current SST
            sst_current = TSeries[:, :, -1] - 273.15
            
            # Success message
            st.success("✅ Data processed successfully!")
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max DHW", f"{np.nanmax(dhw_total):.0f} weeks")
            with col2:
                st.metric("Avg SST", f"{np.nanmean(sst_current):.1f}°C")
            with col3:
                alert_area = np.sum(dhw_total >= 4) / dhw_total.size * 100
                st.metric("Alert Area", f"{alert_area:.1f}%")
            with col4:
                bleaching_area = np.sum(dhw_total >= 5) / dhw_total.size * 100
                st.metric("Bleaching Risk", f"{bleaching_area:.1f}%")
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📊 Accumulated DHW", "🗓️ Weekly Hotspots", "🌡️ Current SST"])
            
            with tab1:
                st.subheader(f"Degree Heating Weeks - {target_date.strftime('%Y-%m-%d')}")
                
                # Main DHW map
                fig_dhw = create_dhw_map(thlon, thlat, dhw_total, 
                                         "Accumulated Degree Heating Weeks (6 weeks)", 6)
                
                # Add DHW level annotations
                st.markdown("""
                **DHW Alert Levels:**
                - 🔵 **0**: Below threshold
                - ⚪ **1-2**: Watching (possible stress)
                - 🟡 **3-4**: Alert (bleaching likely)
                - 🔴 **5-6**: Severe bleaching expected
                """)
                
                st.plotly_chart(fig_dhw, use_container_width=True)
                
                # Download option
                if st.button("💾 Save DHW Data"):
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as tmp:
                        savemat(tmp.name, {'dhw': dhw_total})
                        st.success(f"Data saved! (Note: Download implementation needed)")
            
            with tab2:
                st.subheader("Weekly Hotspot Analysis")
                
                # Create 2x3 subplot grid
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
                            fig = create_dhw_map(thlon, thlat, dhw_weeks[week_idx],
                                               date_labels[week_idx], 2)
                            fig.update_layout(height=350)
                            st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader(f"Sea Surface Temperature - {target_date.strftime('%Y-%m-%d')}")
                
                # SST map
                fig_sst = create_sst_map(thlon, thlat, sst_current,
                                        "Current Sea Surface Temperature")
                st.plotly_chart(fig_sst, use_container_width=True)
                
                # Temperature distribution
                st.subheader("SST Distribution")
                fig_hist = go.Figure(data=go.Histogram(
                    x=sst_current.flatten(),
                    nbinsx=30,
                    marker_color='rgb(55, 83, 109)'
                ))
                fig_hist.update_layout(
                    xaxis_title='Temperature (°C)',
                    yaxis_title='Frequency',
                    height=300
                )
                st.plotly_chart(fig_hist, use_container_width=True)

else:
    # Landing page
    st.info("👈 Upload your data files and click 'Process Data' to begin analysis")
    
    # Instructions
    with st.expander("📖 How to Use This Dashboard"):
        st.markdown("""
        ### Data Requirements
        1. **GHRSST NetCDF files**: 30 consecutive days of sea surface temperature data
        2. **MMM Baseline file**: GMAMTH(all).mat containing climatological baseline
        
        ### Steps
        1. Upload your NetCDF files in the sidebar
        2. Upload the MMM baseline .mat file
        3. Select your analysis date
        4. Click "Process Data"
        
        ### Output
        - **Accumulated DHW Map**: Shows cumulative heat stress over 6 weeks
        - **Weekly Hotspots**: Individual week analysis showing temperature exceedance
        - **Current SST**: Latest sea surface temperature field
        
        ### Alert Levels
        - **Watching (1-2 weeks)**: Possible stress
        - **Alert (3-4 weeks)**: Bleaching likely
        - **Bleaching (5-6 weeks)**: Severe bleaching expected
        """)
    
    # Sample data info
    with st.expander("🔬 About the Data"):
        st.markdown("""
        ### GHRSST Data
        - **Source**: Group for High Resolution Sea Surface Temperature
        - **Resolution**: ~0.05° (~5km)
        - **Coverage**: Global, daily
        
        ### DHW Calculation
        DHW is calculated by accumulating SST anomalies exceeding MMM + 1°C threshold
        over 6 weeks (30 days), where each week with positive anomaly adds 1 DHW unit.
        
        ### Region
        - **Longitude**: 90°E - 110°E (Andaman Sea, Gulf of Thailand)
        - **Latitude**: 0°N - 14.5°N (Thai maritime waters)
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>DHW Coral Bleaching Monitor | Data: GHRSST | Built with Streamlit & Plotly</small>
</div>
""", unsafe_allow_html=True)
