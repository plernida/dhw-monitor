import streamlit as st
import xarray as xr
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import requests
font_paths = {
    'Kanit-Regular.ttf': 'https://raw.githubusercontent.com/google/fonts/main/ofl/kanit/Kanit-Regular.ttf',
    'Kanit-Bold.ttf': 'https://raw.githubusercontent.com/google/fonts/main/ofl/kanit/Kanit-Bold.ttf'
}
for fname, url in font_paths.items():
    if not os.path.exists(fname):
        urllib.request.urlretrieve(url, fname)
    fm.fontManager.addfont(fname)

plt.rcParams['font.family'] = 'Kanit'
st.title('NOAA SST Viewer')

# Latest: Serve from repo (fast, no-wait)
if os.path.exists('static/latest_sst.png'):
    st.image('static/latest_sst.png', caption='Latest Daily SST (auto-updated via GitHub Actions)', use_column_width=True)

# Historical: User selects date, generate PNG
date = st.date_input('Select historical date', datetime.now() - timedelta(days=2))
if st.button('Generate Historical SST'):
    ymd = date.strftime('%Y/%m/%d.nc')
    url = f"https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/highres/netCDF/AVHRR_only/v2.1/{ymd}"
    with st.spinner('Fetching & plotting...'):
        ds = xr.open_url(url)
        sst = ds['sst'].isel(time=0) - 273.15
        fig, ax = plt.subplots(figsize=(12, 8))
        sst.plot(ax=ax, vmin=-2, vmax=32, cmap='RdBu_r')
        st.pyplot(fig)

