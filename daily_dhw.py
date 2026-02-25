import numpy as np
import requests
from netCDF4 import Dataset
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
import os
import xarray as xr
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Your NOAA config
NOAANCSSBASE = "https://www.ncei.noaa.gov/thredds/ncss/grid/OisstBaseNetCDFv2.1/AVHRR"
baseline = xr.open_dataset('mmm_sst_iowp_1981-2020.nc')
MMM = baseline['sst'].sel(lon=slice(90,110.3),lat=slice(0,14.7))  # Load your 1981-2020 climatology (59x81); or hardcode if small

def download_latest_sst(enddate, daysback=30):
    # Exact from your app[file:74]
    sstdata = []
    timelist = []
    latref = lonref = None
    PRELIMWINDOWDAYS = 14
    thtz = pytz.timezone('Asia/Bangkok')
    nowdate = datetime.now(thtz).date()
    for i in range(daysback):
        targetdate = enddate - timedelta(days=i)
        yyyymm = targetdate.strftime('%Y/%m')
        datestr = targetdate.strftime('%Y%m%d')
        isodate = targetdate.strftime('%Y-%m-%d')
        agedays = (nowdate - targetdate).days
        if 0 <= agedays < PRELIMWINDOWDAYS:
            filename = f"oisst-avhrr-v02r01.{datestr}_preliminary.nc"
        else:
            filename = f"oisst-avhrr-v02r01.{datestr}.nc"
        url = (
            f"{NOAA_NCSS_BASE}{yyyymm}/{filename}?"
            f"var=sst&north=14.500&west=90.000&east=110.000&south=0.000&"
            f"horizStride=1&time_start={iso_date}T12:00:00Z&time_end={iso_date}T12:00:00Z&"
            f"accept=netcdf3"
        )
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with Dataset('in-memory', mode='r', memory=resp.content) as nc:
                    sstraw = nc.variables['sst'][0, :,:]
                    sstraw = np.squeeze(sstraw)
                    subsetlat = nc.variables['lat'][:]
                    subsetlon = nc.variables['lon'][:]
                if latref is None:
                    latref, lonref = subsetlat, subsetlon
                else:
                    assert np.array_equal(latref, subsetlat)
                    assert np.array_equal(lonref, subsetlon)
                sstscaled = np.where(sstraw < -100, np.nan, sstraw)
                sstdata.append(sstscaled)
            else:
            # same shape as sst_raw: (nlat, nlon)
                if latref is not None and lonref is not None:
                    sstdata.append(np.full((len(latref), len(lonref)), np.nan))
                else:
                    sstdata.append(None)
        except Exception:
            if lat_ref is not None and lon_ref is not None:
                sstdata.append(np.full((len(lat_ref), len(lon_ref)), np.nan))
            else:
                sstdata.append(None)
        timelist.append(targetdate)
    if lat_ref is None or lon_ref is None:
        raise RuntimeError("No successful downloads; cannot build SST array.")         
    # Fill NaNs
    for idx, v in enumerate(sstdata):
        if v is None:
            sstdata[idx] = np.full((len(latref), len(lonref)), np.nan)
    sststack = np.stack(sstdata, axis=2)
    return sststack, timelist, latref, lonref

def calculate_dhw(TSeries, MMM, threshold=1.0):
    # Exact from your app[file:74]
    dhwweeks = []
    sstweeks = []
    for week in range(6):
        startidx = (5 - week) * 5
        endidx = startidx + 5
        weekmean = np.nanmean(TSeries[:, :, startidx:endidx], axis=2)
        sstweeks.append(weekmean)
        hotspot = weekmean - (MMM + threshold)
        dhwweek = xr.where(hotspot>0, 1, 0)
        dhwweeks.append(dhwweek)
    dhwtotal = sum(dhwweeks, axis=0)
    return dhwweeks, dhwtotal, sstweeks

def plot_dhw_map(lon, lat, dhwtotal, filename):
    LON, LAT = np.meshgrid(lon, lat)
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.contourf(LON, LAT, dhwtotal, levels=7, cmap='RdYlBu_r', vmin=0, vmax=6)
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
plot_dhw_map(lon, lat, dhwtotal, 'static/latest_dhw.png')
plt.figure(figsize=(12, 8))
plt.contourf(np.meshgrid(lon, lat), sstcurrent, cmap='jet', vmin=25, vmax=32)
plt.colorbar(label='SST (°C)')
plt.title('Latest SST')
plt.savefig('static/latest_sst.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Generated PNGs for {today}")
