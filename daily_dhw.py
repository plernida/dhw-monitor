import numpy as np
import json
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
from matplotlib.colors import ListedColormap
from io import BytesIO
import xarray as xr
import warnings
warnings.filterwarnings('ignore')
if not any('Kanit' in f.name for f in fm.fontManager.ttflist):
    os.system("wget -q -O kanit.ttf https://github.com/google/fonts/raw/main/ofl/kanit/Kanit-Regular.ttf")
    fm.fontManager.addfont('kanit.ttf')
plt.rcParams['font.family'] = 'Kanit'

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

# Create custom colormap (N=256 for smooth gradient)
cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors_rgb, N=256)
# Your NOAA config
CRW_ERDDAP_BASE = "https://coastwatch.noaa.gov/erddap/griddap/noaacrwsstDaily"
baseline = xr.open_dataset('crw_mmm_sst_thailand_1985-2025.nc') # read array
MMM = baseline['sst'].sel(lon=slice(90,110),lat=slice(14.1,0))

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
    sst_stack = ds["sst"]
    lat_ref = ds["lat"].values
    lon_ref = ds["lon"].values
    time_list = ds["time"].values

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

def plot_dhw_week(lon, lat, dhw_week, title, filename):
    lon2d, lat2d = np.meshgrid(lon, lat)
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    # DHW raster
    im = ax.contourf(
        lon2d, lat2d, dhw_week,
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
    legend_elements = [
        mpatches.Patch(color=colors_rgb[0], label='No stress'),
        mpatches.Patch(color=colors_rgb[1], label='Watch'),
    ]
    ax.legend(handles=legend_elements,ncol=2,  # Horizontal (5 columns)
           loc='upper center', 
           bbox_to_anchor=(0.5, -0.05),
          fontsize=20, frameon=True, fancybox=True, shadow=True)
    ax.set_title(title, fontsize=20)
    #plt.colorbar(im, ax=ax, shrink=0.8, pad=0.1)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
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
                  bottom=True,top=True,right=True,labelsize=8,grid_color='black',grid_linewidth=0.5)
    # Custom legend patches + labels matching your markdown
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
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
def create_sst_map_mapbox(lon, lat, sstdata, filename):
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
    #cbar.mappable.set_clim(23, 35)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    plt.close()
    
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




thtz = pytz.timezone('Asia/Bangkok')
today = datetime.now(thtz).date() - timedelta(days=3)

try:
    sst_stack, time_list, lat, lon = download_latest_sst(today)
    print("SST downloaded successfully")
    
    dhw_weeks, dhw_total, sst_weeks = calculate_dhw(sst_stack, MMM)
    print("DHW calculated")
    bleaching_area = xr.where(dhw_total >= 5, 1, 0).sum() / dhw_total.size * 100
    update_bleaching_history(today, bleaching_area)
    
    sst_current = sst_stack[:, :, -1]
    
    os.makedirs('static', exist_ok=True)
    
    # Safe NetCDF overwrite
    for nc_file in ['dhw_total.nc', 'sst_current.nc']:
        nc_path = f'static/{nc_file}'
        if os.path.exists(nc_path):
            os.remove(nc_path)

    dhw_total.name = "dhw"
    sst_current.name = "sst"
    dhw_total.to_netcdf('static/dhw_total.nc',engine='scipy')
    sst_current.to_netcdf('static/sst_current.nc',engine='scipy')
    print("NetCDF files saved")
    
    stats = {
        'date': today.strftime('%Y-%m-%d'),
        'max_dhw': float(dhw_total.max()),
        'avg_sst': round(float(np.nanmean(sst_current)), 2),
        'alert_area': round(float((dhw_total >= 4).sum() / dhw_total.size * 100), 1),
        'bleaching_area': round(float((dhw_total >= 5).sum() / dhw_total.size * 100), 2)
    }
    with open('static/dhw_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("dhw_stats.json saved")

    
    # plotting code...

    # Produce PNGs
    plot_dhw_map(lon, lat, dhw_total, f"static/{today.strftime('%Y-%m-%d')}_dhw.png")
    create_sst_map_mapbox(lon, lat, sst_current, f"static/{today.strftime('%Y-%m-%d')}_sst.png")
    
    date_labels = []
    for week in range(6):
        end_day = today - timedelta(days=week*5)
        start_day = end_day - timedelta(days=4)
        date_labels.append(f"{start_day.strftime('%d%b')}-{end_day.strftime('%d%b')}")
    
    for week_idx in range(6):
        plot_dhw_week(lon, lat, dhw_weeks[week_idx], date_labels[week_idx], 
                     f"static/{today.strftime('%Y-%m-%d')}_week_{week_idx+1:02d}.png")
    
    bleaching_area = (xr.where(dhw_total >= 5, 1, 0).sum() / dhw_total.size * 100).item()
    update_bleaching_history(today, bleaching_area)
    
    print("All files generated successfully!")
    
except Exception as e:
    print(f"Error in main run: {e}")
    import traceback
    traceback.print_exc()
    
    #plot_dhw_map(lon, lat, dhw_total, f"static/{today}_dhw.png")
    #create_sst_map_mapbox(lon,lat,sst_current,f"static/{today}_sst.png")
    #date_labels = []
    #for week in range(6):
    #    end_day = today - timedelta(days=week*5)
    #    start_day = end_day - timedelta(days=4)
    #    date_labels.append(f"{start_day.strftime('%d%b')}-{end_day.strftime('%d%b')}")
    
    #for week_idx in range(6):
    #    plot_dhw_week(lon, lat, dhw_weeks[week_idx],date_labels[week_idx], f"static/{today}_week_{week_idx+1:02d}.png")




#plt.figure(figsize=(12, 8))
#plt.contourf(np.meshgrid(lon, lat), sst_current, cmap='jet', vmin=25, vmax=32)
#plt.colorbar(label='SST (°C)')
#plt.title('Latest SST')
#plt.savefig('static/latest_sst.png', dpi=150, bbox_inches='tight')
#plt.close()


