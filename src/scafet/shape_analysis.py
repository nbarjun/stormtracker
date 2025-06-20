import numpy as np
import xarray as xr
import pandas as pd
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import interp1d
import metpy.calc as mpcalc
import sys

# import properties as props
# import shape_analysis as sa
# import filtering as filt
# import tracking as track


class shapeDetector:    
    def apply_smoother(self,data,properties):
        if self.ndims == 2:
            return smoother2D(data,properties)
        elif self.ndims == 3:
            return smoother3D(data,properties)
        
    def apply_shape_detection(self,data,properties):
        return shapeDetection(data,properties)
            
    def __init__(self,data,ndims=None):
        self.ndims = len(list(data.dims))-1 if ndims==None else ndims
        self.vector = True if len(data.data_vars)==2 else False

def gfilter_lons(data,sigma=5):
#     print(np.shape(data))
    return gaussian_filter1d(data,sigma[0],mode='wrap')

def gfilter_lats(data,sigma=3):
#     print(np.shape(data))
    return gaussian_filter1d(data,sigma,mode='nearest')

def non_homogenous_filter(var,slat,slon):
    lonfilter = xr.apply_ufunc(gfilter_lons,var,slon,\
                input_core_dims=[['lon'],['lon']],output_core_dims=[['lon']],\
                vectorize=True, dask='parallelized')

    filtered = xr.apply_ufunc(gfilter_lats,lonfilter,np.mean(slat),\
                input_core_dims=[['lat'],[]],output_core_dims=[['lat']],\
                vectorize=True,dask='parallelized')

#     filtered = xr.DataArray(np.swapaxes(filtered,1,2),dims=['time','lat','lon'],\
#                             coords={'time':var.time,'lat':var.lat,'lon':var.lon})
    return filtered

def calc_magnitude(a, b):
    func = lambda x, y: np.sqrt(x ** 2 + y ** 2)
    with xr.set_options(keep_attrs=True):
        mag = xr.apply_ufunc(func, a, b, dask='allowed')
    return mag

def angleBw(x1,y1,x2,y2):
    ang = np.arccos(((x1*x2)+(y1*y2))/(calc_magnitude(x1,y1)*calc_magnitude(x2,y2)))
    ang = ang*180/np.pi
    return ang

def smoother2D(var,properties):
    #Filter IVTX and IVTY using variying sigma
    for v,i in zip(list(var.data_vars),range(len(list(var.data_vars)))):

        with xr.set_options(keep_attrs=True):
            vn = non_homogenous_filter(var[v],\
                    properties.smooth['sigma_lat'],properties.smooth['sigma_lon'])
            if i==0:
                smoothed = vn.to_dataset(name=v)
                smoothed[v] = smoothed[v].transpose(*var[v].dims)
            else:
                smoothed[v] = vn.transpose(*var[v].dims)
    return smoothed


def shapeDetection(magnitude,props):
    d_dlon = mpcalc.first_derivative(magnitude.metpy.parse_cf()['mag'], axis='lon')
    d_dlat = mpcalc.first_derivative(magnitude.metpy.parse_cf()['mag'], axis='lat')
    
    d2_d2lon = mpcalc.second_derivative(magnitude.metpy.parse_cf()['mag'], axis='lon')
    d2_d2lat = mpcalc.second_derivative(magnitude.metpy.parse_cf()['mag'], axis='lat')

    d2_dlon_dlat = mpcalc.first_derivative(d_dlon, axis='lat')
    d2_dlat_dlon = mpcalc.first_derivative(d_dlat, axis='lon')
    
    #Arranging it for Matric Calculation
    r1= xr.concat([d2_d2lon,d2_dlon_dlat],dim='C1').expand_dims(dim='C2')                  
    r2= xr.concat([d2_dlat_dlon,d2_d2lat],dim='C1').expand_dims(dim='C2')                  
    H_elems = xr.concat([r1,r2],dim='C2')
    ##------------------------------------------------------------------------   
    Hessian = H_elems.transpose('time','lon','lat','C1','C2')
    #Calcualtion of eigen vectors and eigen values for the symmetric array
    eigvals = xr.apply_ufunc(np.linalg.eigh,Hessian,\
            input_core_dims=[['lat','lon','C1','C2']],\
            output_core_dims=[['lat','lon','e'],['lat','lon','n','e']],\
            dask='parallelized',vectorize=True)
    
    eigval = eigvals[0].assign_coords(e=['min','max'])
    eigvec = eigvals[1].assign_coords(n=['x','y'])
    eigvec = eigvec.assign_coords(e=['min','max'])
    
    #Transport along the ridge direction
    Ar = (magnitude.mag*eigvec.sel(e='max',n='x')*-1 +\
        magnitude.mag*eigvec.sel(e='max',n='y')*-1)/np.sqrt(eigvec.sel(e='max',n='x')**2+\
                            eigvec.sel(e='max',n='y')**2)
    
    #Transport along the ridge direction
    gAr = d_dlon*eigvec.sel(e='max',n='x')*-1 +\
        d_dlat*eigvec.sel(e='max',n='y')*-1
    
    gradient = d_dlon+d_dlat
    shapei = (2/np.pi)*np.arctan((eigval.sel(e='min')+eigval.sel(e='max'))/\
                                 (eigval.sel(e='min')-eigval.sel(e='max')))
    eigs = shapei.to_dataset(name='sindex')
    eigs['Ar'] = Ar/magnitude.mag
    eigs['gAr'] = gAr/magnitude.mag
    ridges = magnitude.mag.where((eigs['sindex']>props.obj['Shape_Index'][0])&\
                             (eigs['sindex']<=props.obj['Shape_Index'][1]) & \
                                 (abs(eigs['Ar'])>0.))
    eigs['ridges'] = ridges*0+1
    eigs['gradient'] = gradient
    
    
    cores = (abs(np.sign(gradient).differentiate('lat'))+\
                     abs(np.sign(gradient).differentiate('lon')))
    eigs['core'] = cores.where((cores>0) & (shapei>props.obj['Shape_Index'][0]) &\
                              (shapei>props.obj['Shape_Index'][0]))*0+1

    return eigs