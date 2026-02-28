import numpy as np
import requests
from netCDF4 import Dataset
from datetime import datetime, timedelta
import pytz
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LinearSegmentedColormap
import os
import matplotlib.font_manager as fm
from io import BytesIO
import xarray as xr
import warnings
warnings.filterwarnings('ignore')
if not any('Kanit' in f.name for f in fm.fontManager.ttflist):
    os.system("wget -q -O kanit.ttf https://github.com/google/fonts/raw/main/ofl/kanit/Kanit-Regular.ttf")
    fm.fontManager.addfont('kanit.ttf')
plt.rcParams['font.family'] = 'Kanit'

cmap_full = plt.get_cmap('nipy_spectral')
slice_start, slice_end = 0.45, 0.9
colors = cmap_full(np.linspace(slice_start, slice_end, 256))
nipy_yellow_red = LinearSegmentedColormap.from_list('nipy_yellow_red', colors)

colors_rgb = [
    '#C8FAFA',    # Blue
    '#FFF000',   # Gray
    '#FAAA0A',   # Beige
    '#F00000',   # Pink
    '#960000',    # Brown
    '#A05024',     # Dark brown
    '#F000F0'      # Dark brown
]

# Create custom colormap (N=256 for smooth gradient)
cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors_rgb, N=256)
# Your NOAA config
NOAA_NCSS_BASE = "https://www.ncei.noaa.gov/thredds/ncss/grid/OisstBase/NetCDF/V2.1/AVHRR/"
baseline = xr.open_dataset('mmm_sst_iowp_1981-2020.nc') # read array
MMM = baseline['sst'].sel(lon=slice(90,110.3),lat=slice(0,14.7)) # Add noise if desired

def download_latest_sst(enddate, days_back=30):
    # Exact from your app[file:74]
    sstdata = []
    time_list = []
    lat_ref = lon_ref = None
    PRELIM_WINDOW_DAYS = 14
    thtz = pytz.timezone('Asia/Bangkok')
    now_date = datetime.now(thtz).date()
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
    lon2d, lat2d = np.meshgrid(lon, lat)
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
                  bottom=True,top=True,right=True,labelsize=6,grid_color='black',grid_linewidth=0.5)
    # Custom legend patches + labels matching your markdown
    legend_elements = [
        mpatches.Patch(color=colors_rgb[0], label='No stress'),
        mpatches.Patch(color=colors_rgb[1], label='Watch'),
        mpatches.Patch(color=colors_rgb[2], label='Warning'),
        mpatches.Patch(color=colors_rgb[3], label='Alert 1'),
        mpatches.Patch(color=colors_rgb[4], label='Alert 2')  # Use darkest for 6+
    ]
    ax.legend(handles=legend_elements,ncol=5,  # Horizontal (5 columns)
           loc='upper center', 
           bbox_to_anchor=(0.5, -0.05),
          fontsize=6, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def create_sst_map_mapbox(lon, lat, sstdata, filename):
    lon2d, lat2d = np.meshgrid(lon, lat)    
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    im = ax.contourf(lon2d, lat2d, sstdata,
                     cmap=nipy_yellow_red,levels=np.linespace(24,34,9),
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
                  bottom=True,top=True,right=True,labelsize=6,grid_color='black',grid_linewidth=0.5)
    cbar=fig.colorbar(im,ax=ax,orientation='horizontal', shrink=0.8, pad=0.05)

    #cbar.mappable.set_clim(23, 35)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    plt.close()
    

# Main daily run
thtz = pytz.timezone('Asia/Bangkok')
today = datetime.now(thtz).date() - timedelta(days=2)  # Your app's target
sst_stack, time_list, lat, lon = download_latest_sst(today)
dhw_weeks, dhw_total, _ = calculate_dhw(sst_stack, MMM)
sst_current = sst_stack[:, :, -1]

# Produce PNGs
os.makedirs('static', exist_ok=True)
plot_dhw_map(lon, lat, dhw_total, f"static/{today}_dhw.png")
create_sst_map_mapbox(lon,lat,sst_current,f"static/{today}_sst.png")
#plt.figure(figsize=(12, 8))
#plt.contourf(np.meshgrid(lon, lat), sst_current, cmap='jet', vmin=25, vmax=32)
#plt.colorbar(label='SST (°C)')
#plt.title('Latest SST')
#plt.savefig('static/latest_sst.png', dpi=150, bbox_inches='tight')
#plt.close()


