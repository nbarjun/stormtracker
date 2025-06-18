import src as storm
import xarray as xr
import sys

log_file = open("stormtracker.log","w")
sys.stdout = log_file

# Filename for wind data
windfile = 'wind10.grib2'

# Download Latest IFS wind data
response = storm.utilities.download_latest_ifsdata(windfile)
print('Successfully Downloaded Latest IFS Data')

# Open wind data
ifsdata = xr.open_dataset(windfile,engine='cfgrib')
# Preprocess wind fields
wind10 = storm.utilities.preprocess_ifsdata(ifsdata)

# Extract Storms
print('Successfully Extracted Storms')

# Plot wind data
w10 = wind10.isel(time=0)
w10['ws'] = (w10['u10']**2 + w10['v10']**2)**0.5
figname = 'wind_streamline.png'
streamplot = storm.utilities.plot_wind_streamlines(w10,figname)
# Make the html figure
webname = 'latest_storms.html'
storm.utilities.make_webpage(webname, figname)
print('Successfully Made the Stormtracker Map')
log_file.close()
