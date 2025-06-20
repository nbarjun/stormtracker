# Modules required for computation
import xarray as xr
import numpy as np
from tqdm import tqdm
import cfgrib
import time
from ecmwf.opendata import Client
import metpy.calc as mpcalc
from . import scafet as storm

# Modules required for plotting
import folium
from folium.plugins import Fullscreen
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as colors

def download_latest_ifsdata(fname):
    """
    Downloads the latest IFS (Integrated Forecasting System) forecast data from ECMWF.

    Parameters:
    - fname: str
        Target filename where the downloaded GRIB data will be saved.

    Returns:
    - data: the response object from the data retrieval client.
    """
    # Initialize the client for ECMWF IFS model data
    client = Client(
        source="ecmwf",          # Data source is ECMWF
        model="ifs",             # Use the Integrated Forecasting System (IFS) model
        resol="0p25",            # Horizontal resolution of 0.25 degrees
        preserve_request_order=False,  # Reorder requests for performance (optional)
        infer_stream_keyword=True      # Let client infer the correct stream based on other parameters
    )

    # Retrieve forecast data for 10-meter U and V wind components
    data = client.retrieve(
        time=0,                     # Base time of the forecast (00 UTC)
        type="fc",                  # Forecast type
        step=[0, 12, 24, 36, 48],   # Forecast steps in hours
        param=["10u", "10v"],       # Parameters: 10-meter zonal (u) and meridional (v) winds
        target=fname                # File to save the retrieved data
    )

    return data

def rotate180(ds):
    """
    Rotates longitude coordinates of an xarray Dataset from [-180, 180] to [0, 360] and sorts them.

    Parameters:
    - ds: xarray.Dataset
        Input dataset with longitude coordinates possibly in the [-180, 180] range.

    Returns:
    - xarray.Dataset
        Dataset with longitudes converted to [0, 360] range and sorted.
    """
    
    # Extract longitude values from the dataset
    lon = ds['lon'].values

    # Convert negative longitudes (e.g., -170) to their equivalent in [0, 360] (e.g., 190)
    lon = xr.where(lon < 0, lon + 360, lon)

    # Extract latitude values (not used here, but extracted — can be removed if unused)
    lat = ds['lat'].values

    # Assign the updated longitudes back to the dataset
    ds = ds.assign_coords({'lon': lon})

    # Sort the dataset along the longitude dimension to ensure longitudes are in increasing order
    return ds.sortby('lon')

def preprocess_ifsdata(ds):
    """
    Preprocesses IFS forecast data to prepare it for analysis.
    
    Steps:
    - Rename latitude and longitude dimensions to 'lat' and 'lon'
    - Flip latitude to go from north to south (if needed)
    - Rotate longitudes from [-180, 180] to [0, 360]
    - Rename time-related dimensions for clarity
    - Load the dataset into memory

    Parameters:
    - ds: xarray.Dataset
        Raw IFS dataset

    Returns:
    - ifsdata_rotated: xarray.Dataset
        Preprocessed and standardized dataset
    """
    # Rename longitude and latitude to standard names
    ifsdata = ds.rename({'longitude': 'lon', 'latitude': 'lat'})

    # Reverse the latitude ordering (from south to north → north to south)
    ifsdata = ifsdata.reindex(lat=list(reversed(ifsdata.lat)))

    # Rotate longitudes from [-180, 180] to [0, 360] and sort
    ifsdata_rotated = rotate180(ifsdata)

    # Rename 'time' to 'forecast-time' (actual forecast issue time)
    ifsdata_rotated = ifsdata_rotated.rename({'time': 'forecast-time'})

    # Rename 'step' to 'time' (lead time becomes the main time coordinate)
    ifsdata_rotated = ifsdata_rotated.rename({'step': 'time'}).load()

    return ifsdata_rotated

def plot_wind_streamlines(wind,fout):
    """
    Plots wind streamlines with alpha transparency and custom coloring.

    Parameters:
    - wind: xarray.Dataset with variables 'u10', 'v10', 'ws', and coordinates 'lon', 'lat'

    Returns:
    - fig: The resulting matplotlib figure object
    """

    # Choose base colormap and modify its alpha transparency
    cmap = plt.cm.inferno
    my_cmap = cmap(np.arange(cmap.N))                    # Get RGBA values
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N) ** 0.25   # Set transparency (alpha)
    my_cmap = colors.ListedColormap(my_cmap)             # Create new colormap

    # Customize plot aesthetics
    plt.rcParams.update({
        'font.size': 6,
        'axes.edgecolor': 'w',
        'ytick.color': 'w',
        'xtick.color': 'w'
    })

    # Create figure with Mercator projection
    fig, ax = plt.subplots(figsize=(12, 4), dpi=600,
                        subplot_kw={'projection': ccrs.Mercator(central_longitude=0.0,
                        min_latitude=-82,max_latitude=82)})

    # Plot streamlines
    slplot = wind.plot.streamplot(
        x='lon', y='lat',
        u='u10', v='v10',
        vmax=20, vmin=1,
        hue='ws',
        density=4,
        cmap=my_cmap,
        transform=ccrs.PlateCarree(),
        add_guide=False,
        linewidth=0.5,
        arrowsize=0.2,
        arrowstyle='Fancy',
        ax=ax
    )

    # Custom colorbar positioning
    cb_pos = ax.get_position()
    pos_cax= fig.add_axes([cb_pos.x0+cb_pos.width/2,cb_pos.y0+.07,\
                           cb_pos.width/2.5,cb_pos.height/30])
    cb=plt.colorbar(slplot, cax=pos_cax, orientation='horizontal')
    cb.ax.set_visible=False
    cb.ax.set_title(r'$m/s$',color='w',x=1.1,y=-0.5)
    # Remove axis and set title
    ax.set_title('')
    ax.axis('off')

    # Save output
    plt.savefig(fout, bbox_inches='tight',\
                pad_inches=0, transparent=True)
    return fig

def detect_storms_scafet(wind):
    grid_area = xr.open_dataset('resources/grid_area_era5.nc')
            # .sel(latitude=latslice,longitude=lonslice)
    grid_area = grid_area.rename({'longitude':'lon'})
    grid_area = grid_area.rename({'latitude':'lat'})
    grid_area = grid_area.reindex\
        (lat=list(reversed(grid_area.lat)))
    land_mask = xr.open_dataset('resources/land_sea_mask_era5.nc')

    relVor = mpcalc.vorticity(wind['u10'], wind['v10'])
    rv = relVor.metpy.dequantify().to_dataset(name='rv')
    cyc = rv*np.sign(rv.lat)

    smooth_scale = 2e6
    angle_threshold = 45
    shape_index = [0.625,1]
    min_length = 20e3
    min_area = 2e11
    min_duration = 6
    max_distance_per_tstep = 1000e3
    shape_eccentricity = [0.0,1.0]
    lat_mask = [-0,0]
    lon_mask = [360,0]
    
    properties = storm.properties.object_properties2D(grid_area,\
                        land_mask,min_length,min_area,\
                        smooth_scale,angle_threshold,min_duration,max_distance_per_tstep,\
                        shape_index,shape_eccentricity,lon_mask,lat_mask)

    stime = time.time()
    sdetect = storm.shape_analysis.shapeDetector(cyc)
    vor = sdetect.apply_smoother(cyc,properties)
    print('Finished smoothing in {} seconds'.format(time.time()-stime))

    vor = vor.rename({'rv':'mag'})
    cyc = cyc.rename({'rv':'mag'})
    stime = time.time()
    # Select only positive values of cyclonic vorticity
    cyc_sm = vor.where((vor.mag>0)).fillna(0)
    # Detect Ridges
    ridges = sdetect.apply_shape_detection(cyc_sm,properties)
    print('Finished shape extraction in {} seconds'.format(time.time()-stime))

    # Use unsmoothed vorticity as primary field 
    cyc_us = cyc.where((cyc.mag>0)).fillna(0)
    # Define the secondary field as wind speed
    ws = np.sqrt(wind['u10']**2+wind['v10']**2)
    props_mag = xr.concat([ws.expand_dims('Channel'),\
                cyc_us.mag.expand_dims('Channel')], dim='Channel')
    props_mag = props_mag.to_dataset(name='mag')
    
    stime = time.time()
    cycfilter = storm.filtering.filterObjects(ridges)
    filtered = cycfilter.apply_filter(ridges,\
                props_mag,['max_intensity','mean_intensity'],
                [10,2e-5],properties,'ridges')
    print('Finished Filtering in {} seconds'.format(time.time()-stime))

    object_masks = filtered[1]
    object_properties = filtered[0]
    
    stime = time.time()
    # Tracking
    properties.obj['Min_Duration']=0
    properties.obj['Max_Distance']=1000
    
    # Tracking based on centroid of each object
    latlon = ['wclat','wclon']
    
    tracker = storm.tracking.Tracker(latlon,properties)
    tracked = tracker.apply_tracking(object_properties,object_masks)
    print('Finished Tracking in {} seconds'.format(time.time()-stime))

    object_masks = filtered[1]
    object_properties = filtered[0]
    
    stime = time.time()
    # Tracking
    properties.obj['Min_Duration']=0
    properties.obj['Max_Distance']=1000
    
    # Tracking based on centroid of each object
    latlon = ['wclat','wclon']
    
    tracker = storm.tracking.Tracker(latlon,properties)
    tracked = tracker.apply_tracking(object_properties,object_masks)
    print('Finished Tracking in {} seconds'.format(time.time()-stime))

    return tracked[0]
    
def make_webpage(webname, tracked_storms, image_name):
    """
    Creates an interactive folium map and overlays a raster image (e.g., wind or storm data).
    
    Parameters:
    - webname: str, name of the output HTML file
    - image_name: str or numpy array, path to the image or image array to overlay
    
    Returns:
    - None (saves a webpage as HTML)
    """
    # Create a base map figure
    f = folium.Figure(width=1080, height=300)
    # Initialize the map with dark tiles and centered on equator
    m = folium.Map(
        location=(0, 0),
        zoom_start=3,
        min_zoom=1,
        max_zoom=9,
        no_wrap=True,
        crs='EPSG3857',
        tiles=folium.TileLayer("CartoDB.DarkMatter",no_wrap=True),
    ).add_to(f)

    Fullscreen(position="topright",
            title="Expand me",
            title_cancel="Exit me",
            force_separate_button=True,
            ).add_to(m)
    
    # Overlay image (e.g., wind velocity or forecast plot)
    folium.raster_layers.ImageOverlay(
        image=image_name,
        bounds=[[-82, -180], [82, 180]],  # Global extent
        opacity=1,
        name="Overlay",
        interactive=False,
        cross_origin=False,
        zindex=1,
        alt="Overlay Image"
    ).add_to(m)

    icon_size = 15
    circle_scale = 1e6
    for lat,lon,w,v,a in zip(tracked_storms['wclat'].values,tracked_storms['wclon'].values,\
                           tracked_storms['max_intensity-1'].values,tracked_storms['max_intensity-2'].values,\
                      tracked_storms['Obj_Area']):
        if lon>180:
            lon = lon-360
        sprop = '<p style="color:black; font-size:100%; font-family:Helvetica">Max Wind Speed:{:.2f}\
                m/s<br>Max Vorticity:{:.2f}\u271510<sup>-4</sup></p>'.format(w,v*1e4)
        popup = folium.Popup(sprop, min_width=100,max_width=300)
    
        icon = folium.CustomIcon(
            icon_image='storm.png',  # Use your own hurricane icon path
            icon_size=(icon_size, icon_size),
            icon_anchor=(icon_size // 2, icon_size // 2))
        folium.Marker(
            location=[lat, lon],icon=icon,
            popup=popup).add_to(m)
        # Add a circle showing cyclone size
        folium.Circle(
                location=[lat, lon],
                radius=a / circle_scale,  # Convert km to meters
                color="blue",
                weight=2,
                fill=False,
                fill_color="none",
                fill_opacity=0.15).add_to(m)

    # Save the interactive map as HTML
    m.save(webname)

    return None