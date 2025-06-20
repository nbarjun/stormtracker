from src import stormtracker
import xarray as xr
import sys
import time

log_file = open("stormtracker.log","w")
sys.stdout = log_file

# Filename for wind data
windfile = 'wind10.grib2'
stime = time.time()
# Download Latest IFS wind data
response = stormtracker.download_latest_ifsdata(windfile)
print('Successfully Downloaded Latest IFS Data in {} seconds'.format(time.time()-stime))

stime = time.time()
# Open wind data
ifsdata = xr.open_dataset(windfile,engine='cfgrib')
# Preprocess wind fields
wind10 = stormtracker.preprocess_ifsdata(ifsdata)

# Extract Storms
trackedStorms = stormtracker.detect_storms_scafet(wind10)
print('Successfully Extracted Storms in {} seconds'.format(time.time()-stime))
trackedStorms = trackedStorms.to_pandas()
trackedStorms = trackedStorms[trackedStorms['timestamp']==0]

stime = time.time()
# Plot wind data
w10 = wind10.isel(time=0)
w10['ws'] = (w10['u10']**2 + w10['v10']**2)**0.5
figname = 'wind_streamline.png'
streamplot = stormtracker.plot_wind_streamlines(w10,figname)
# Make the html figure
webname = 'latest_storms.html'
stormtracker.make_webpage(webname, trackedStorms, figname)
print('Successfully Made the Stormtracker Map in {} seconds'.format(time.time()-stime))
log_file.close()
