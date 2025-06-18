import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import interp1d
import metpy as metpy
from skimage.morphology import remove_small_holes
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
from scipy.interpolate import NearestNDInterpolator as nn_interp
import metpy.calc as mpcalc
import sys

# import properties as props
# import shape_analysis as sa
# import filtering as filt
# import tracking as track


class object_properties2D:    
    def wrapped(self,data):
        nlons = len(data.lon)//1 #No.of longitudes to be added to right side
        lf_nlons = 0
        #Wrap around in longitudes
        wraped = np.pad(data.values,pad_width=[(0,0),(lf_nlons,nlons)],mode='wrap')
        wrap_lons = np.concatenate([data.lon.values,abs(data.lon.values[:nlons])+360])
#         pad_lats = np.concatenate([[-91],data.lat.values,[91]])

        xwrap = xr.DataArray(wraped,coords={'lon':wrap_lons,'lat':data.lat.values},\
                             dims=['lat','lon'])

        return xwrap

    def check_coverage(self,data):
        if data.lon[1]+data.lon[-1] == 360:
            return True
        else:
            return False
    
    def remove_hole_lats(self,data,lat):
        return remove_small_holes(data,lat)

    def remove_hole_lons(self,data,lon):
        return remove_small_holes(data,lon[0])

    def remove_holes(self,data):
        min_lons= xr.apply_ufunc(self.obj['Min_Lon_Points'],data.lon)
        min_lat = xr.apply_ufunc(self.obj['Min_Lat_Points'],data.lat)
        min_lats= np.tile(min_lat.values,(len(min_lons.lon.values),1))
        min_lats= xr.DataArray(min_lats, coords=[min_lons.lon,min_lat.lat])

        removed_lons = xr.apply_ufunc(self.remove_hole_lons,data,\
                             min_lats,input_core_dims=[['lon'],['lon']],\
                            output_core_dims=[['lon']],vectorize=True)
        removed_lats = xr.apply_ufunc(self.remove_hole_lats,data,\
                             np.mean(min_lons),input_core_dims=[['lat'],[]],\
                                 output_core_dims=[['lat']],vectorize=True)
        removed = xr.ufuncs.logical_and(removed_lons,removed_lats)

        return removed

    #Function to get coastline information from landfraction
    def get_coastline_info(self,land):  
        #Removing small water bodies
        island = land.islnd.fillna(0)
#         land_filled = self.remove_holes(island.astype(bool))
        #Get Coastlines
        coastlines = find_boundaries(island)

        coast_info = land.copy()
        coast_info['coastlines'] = xr.DataArray(coastlines,dims=land.dims,coords=land.coords)
        coast_info['coastlines'] = coast_info['coastlines'].where(coast_info['coastlines']>0)

        xgrad = island.differentiate('lon')
        ygrad = island.differentiate('lat')

        xgrad = xgrad.where(xgrad<=0,1)
        ygrad = ygrad.where(ygrad<=0,1)
        xgrad = xgrad.where(xgrad>=0,-1)
        ygrad = ygrad.where(ygrad>=0,-1)
        xgrad = xgrad.where(coast_info['coastlines']==1)
        ygrad = ygrad.where(coast_info['coastlines']==1)

        coast_info['orientation'] = xr.apply_ufunc(np.arctan2,xgrad,ygrad,dask='parallelized')*180/np.pi
        return coast_info
    
    def mpcalc_grid_delta(self,grid_area):
        #Calculate the grid deltas
        grid_delta = metpy.xarray.grid_deltas_from_dataarray(grid_area.metpy.parse_cf()['cell_area'])
        #Calcualte the sigma of the gaussian filter along each longitude; Constant value is used
        grid_lat  = np.concatenate((np.array(grid_delta[1]),np.array(grid_delta[1][-1:,:])),axis=0)

        #Calculate the sigma of the gaussinan filter along each latitude; Values change with lats
        grid_lon =np.concatenate((np.array(grid_delta[0]),np.array(grid_delta[0][:,0:1])),axis=1)

        distance = xr.Dataset({'xdistance':(['lat','lon'],grid_lon),\
                    'ydistance':(['lat','lon'],grid_lat)},\
                        coords={'lat':(['lat'],grid_area.lat.values),
                         'lon':(['lon'],grid_area.lon.values)})

        distance.xdistance.attrs['units'] = 'm'
        distance.ydistance.attrs['units'] = 'm'
        return distance

    def calc_sigma(self,grid_area):
        #Calculate the sigma of the gaussinan filter along each longitude; Values constant
        sigma_lat = np.rint(self.obj['Smooth_Scale']/np.squeeze(self.grid['ydistance']))
        sigma_lat = np.where(sigma_lat>len(grid_area.lat)//1,len(grid_area.lat)//1,sigma_lat)
        sigma_lat = np.where(sigma_lat<2,2,sigma_lat)/(2*np.pi)

        #Calculate the sigma of the gaussinan filter along each latitude; Values change with lats
        sigma_lon = np.rint(self.obj['Smooth_Scale']/np.squeeze(self.grid['xdistance']))
        sigma_lon = np.where(sigma_lon>len(grid_area.lon)//1,len(grid_area.lon)//1,sigma_lon)
        sigma_lon = np.where(sigma_lon<2,2,sigma_lon)/(2*np.pi)

        sigmas = xr.Dataset({'sigma_lat':(['lat','lon'],sigma_lat),\
                    'sigma_lon':(['lat','lon'],sigma_lon)},\
                        coords={'lat':(['lat'],grid_area.lat.values),
                         'lon':(['lon'],grid_area.lon.values)})
        return sigmas

    def plot_properties(self,option):
        if option == 'grid':
            fig,ax = plt.subplots(1,3,figsize=(12,5),dpi=100,subplot_kw={'projection': \
                                        ccrs.EckertIII(central_longitude=0.0)})
            properties=['xdistance','ydistance','grid_area']

            for a,p in zip(ax,properties):
                if p == 'grid_area':
                    self.grid[p]['cell_area'].plot(ax=a,transform=ccrs.PlateCarree(),\
                                    cbar_kwargs={'shrink':.25,'aspect':15})
                else:
                    self.grid[p].plot(ax=a,transform=ccrs.PlateCarree(),\
                                          cbar_kwargs={'shrink':.25,'aspect':15,'format':'%.1e'})

                a.coastlines()
                a.gridlines(linestyle='--',alpha=0.4)
                a.set_title(p)
        elif option == 'smooth':
            fig,ax = plt.subplots(1,2,figsize=(12,5),dpi=100,subplot_kw={'projection': \
                                    ccrs.EckertIII(central_longitude=0.0)})
            properties=['sigma_lon','sigma_lat']
            for a,p in zip(ax,properties):
                self.smooth[p].plot(ax=a,transform=ccrs.PlateCarree(),\
                    cbar_kwargs={'shrink':.25,'aspect':15,'label':r'$\sigma$'})
                a.coastlines()
                a.gridlines(linestyle='--',alpha=0.4)
                a.set_title(p)
                
        elif option == 'land':
            fig,ax = plt.subplots(1,3,figsize=(12,5),dpi=100,subplot_kw={'projection': \
                                        ccrs.EckertIII(central_longitude=0.0)})
            properties=['islnd','coastlines','orientation']
            cmaps = ['binary','binary','hsv']
            for a,p,c in zip(ax,properties,cmaps):
                c=self.land[p].plot(ax=a,transform=ccrs.PlateCarree(),cmap=c,\
                    add_colorbar=False)
                a.gridlines(linestyle='--',alpha=0.4)
                a.set_title(p)
            cb_pos = ax[-1].get_position()
            pos_cax= fig.add_axes([cb_pos.x0+cb_pos.width+0.01,cb_pos.y0,\
                                   cb_pos.width/30,cb_pos.height])
            cb=plt.colorbar(c, cax=pos_cax, orientation='vertical')
            cb.set_label(r'$degree$')
                
    def print_properties(self):        
        obj_value_unit = ['m','degree',\
                          'unitless','m',r'm$^2$', 'counts','m',\
                          'counts','counts','degrees','degrees',\
                          'unitless','0-1','degrees','degrees']
        
        obj = {'Object Properties':list(self.obj.keys()),'Values':list(self.obj.values()),\
              'Units':obj_value_unit}
        
        obj = pd.DataFrame(obj)
        return obj
    
    def __init__(self,grid_area,grid_land,length,area,smooth,theta_t,
                 min_duration,max_distance,shapei,eccentricity,lon_mask,lat_mask):
        min_length = 2e6 if length==None else length
        min_area = 1e10 if area==None else area
        cell_area = self.wrapped(grid_area.cell_area)
        
#         min_area_pixel = interp1d(cell_area.lat.values,\
#                (min_area/cell_area.isel(lon=0)))
        min_area_lat = interp1d(cell_area.lat.values,\
               ((min_area/4)**.5/cell_area.isel(lon=0)**.5))
        min_area_lon = interp1d(cell_area.lon.values,\
               ((min_area/4)**.5/cell_area.isel(lat=0)**.5))
        
        map_lats = interp1d(range(len(cell_area.lat.values)),\
               cell_area.lat.values,fill_value=(-90.,90.),bounds_error=False)
        map_lons = interp1d(range(len(cell_area.lon.values)),\
               cell_area.lon.values)

        smooth = self.min_length//3 if smooth==None else smooth
        theta = 45 if theta_t==None else theta_t
        shapei = 0.375 if shapei==None else shapei
        ecc = [0.5,1.] if eccentricity==None else eccentricity
        
        #Check if the given data is global
        isglobal = self.check_coverage(grid_area)
        
        obj = {'Smooth_Scale':smooth,'Angle_Coherence':theta,'Shape_Index':shapei,\
              'Min_Length':min_length,'Min_Area':min_area,'Min_Duration':min_duration,\
              'Max_Distance':max_distance,'Min_Lon_Points':min_area_lon,'Min_Lat_Points':min_area_lat,\
              'Map_Lons':map_lons,'Map_Lats':map_lats,\
              'Eccentricity':ecc,'isglobal':isglobal,'Lon_Mask':lon_mask,'Lat_Mask':lat_mask}
        self.obj = obj
        
        grid_deltas = self.mpcalc_grid_delta(grid_area.metpy.parse_cf())
        xdistance = grid_deltas.xdistance
        ydistance = grid_deltas.ydistance
        
        grid = {'xdistance':xdistance,'ydistance':ydistance,'grid_area':grid_area}
        self.grid = grid
        
        sigmas = self.calc_sigma(grid_area)         
        smooth = {'sigma_lon':sigmas.sigma_lon,'sigma_lat':sigmas.sigma_lat}
        self.smooth = smooth
        
        self.land = self.get_coastline_info(grid_land)
