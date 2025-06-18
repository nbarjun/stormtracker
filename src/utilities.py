# Modules required for computation
import xarray as xr
import numpy as np
from tqdm import tqdm
import cfgrib
from ecmwf.opendata import Client

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
    
def make_webpage(webname, image_name):
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

    # Save the interactive map as HTML
    m.save(webname, title='StormTracker')

    return None