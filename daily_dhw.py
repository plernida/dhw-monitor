import numpy as np
import requests
from netCDF4 import Dataset
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
import os
from io import BytesIO
import xarray as xr
import warnings
warnings.filterwarnings('ignore')

# Your NOAA config
NOAANCSSBASE = "https://www.ncei.noaa.gov/thredds/ncss/grid/OisstBaseNetCDFv2.1/AVHRR"
baseline = xr.open_dataset('mmm_sst_iowp_1981-2020.nc') # read array
MMM = baseline['sst'].sel(lon=slice(90,110.3),lat=slice(0,14.7)) # Add noise if desired

def download_latest_sst(enddate, daysback=30):
    # Exact from your app[file:74]
    sstdata = []
    timelist = []
    latref = lonref = None
    PRELIMWINDOWDAYS = 14
    thtz = pytz.timezone('Asia/Bangkok')
    nowdate = datetime.now(thtz).date()
    for i in range(days_back):
        target_date = enddate - timedelta(days=i)
        yyyymm = target_date.strftime('%Y%m')
        datestr = target_date.strftime('%Y%m%d')
        iso_date = target_date.strftime('%Y-%m-%d')

        age_days = (now_date - target_date).days
        # Preliminary: if target_date is within 14 days of CURRENT now_date
        if 0 <= age_days <= PRELIM_WINDOW_DAYS:
        
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
                    sst_raw = nc.variables['sst'][0, :, :]
                    sst_raw = np.squeeze(sst_raw)# (lat, lon)
                    subset_lat = nc.variables['lat'][:]
                    subset_lon = nc.variables['lon'][:]
                    
                    # Scale and mask
                    
                    
                # Keep reference grid from first successful file
                if lat_ref is None:
                    lat_ref = subset_lat
                    lon_ref = subset_lon
                else:
                    # Optional: check that all days use same grid
                    assert np.array_equal(lat_ref, subset_lat)
                    assert np.array_equal(lon_ref, subset_lon)
                    
                # Scale and mask
                sst_scaled = np.where(sst_raw < -100, np.nan, sst_raw)
                
                sstdata.append(sst_scaled)
               
            else:
                
            # same shape as sst_raw: (nlat, nlon)
                if lat_ref is not None and lon_ref is not None:
                    sstdata.append(np.full((len(lat_ref), len(lon_ref)), np.nan))
                else:
                    sstdata.append(None)
        except Exception:
            if lat_ref is not None and lon_ref is not None:
                sstdata.append(np.full((len(lat_ref), len(lon_ref)), np.nan))
            else:
                sstdata.append(None)
        time_list.append(target_date)
    if lat_ref is None or lon_ref is None:
        raise RuntimeError("No successful downloads; cannot build SST array.")   
    for idx, v in enumerate(sstdata):
        if v is None:
            sstdata[idx] = np.full((len(lat_ref), len(lon_ref)), np.nan)
    sst_stack = np.stack(sstdata, axis=2)
    return sst_stack, time_list, lat_ref, lon_ref

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

def plot_dhw_map(lon, lat, dhw_total, filename):
    LON, LAT = np.meshgrid(lon, lat)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.contourf(LON, LAT, dhw_total, levels=7, cmap='RdYlBu_r', vmin=0, vmax=6)
    ax.set_xlim(90, 110)
    ax.set_ylim(0, 14.5)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title('Daily DHW Total (Thai Waters)')
    plt.colorbar(im, ax=ax, label='DHW (C-weeks)')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# Main daily run
thtz = pytz.timezone('Asia/Bangkok')
today = datetime.now(thtz).date() - timedelta(days=2)  # Your app's target
sststack, timelist, lat, lon = download_latest_sst(today)
dhwweeks, dhwtotal, _ = calculate_dhw(sststack, MMM)
sstcurrent = sststack[:, :, -1]

# Produce PNGs
os.makedirs('static', exist_ok=True)
plot_dhw_map(lon_ref, latref, dhw_total, 'static/latest_dhw.png')
#plt.figure(figsize=(12, 8))
#plt.contourf(np.meshgrid(lon, lat), sst_current, cmap='jet', vmin=25, vmax=32)
#plt.colorbar(label='SST (°C)')
#plt.title('Latest SST')
#plt.savefig('static/latest_sst.png', dpi=150, bbox_inches='tight')
#plt.close()

print(f"Generated PNGs for {today}")
