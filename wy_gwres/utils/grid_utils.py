# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 10:27:19 2017

Collection of functions to aid Modflow grid creation and manipulation from
spatial datasets

@author: kbefus
"""
from __future__ import print_function
import netCDF4
from osgeo import gdal, osr, ogr
from scipy.interpolate import RegularGridInterpolator,griddata
from scipy.stats import binned_statistic_2d
import numpy as np
import rasterio
import affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
#import cartopy.crs as ccrs

from .proj_utils import xrot,yrot,projectXY

# ---------- Global variables --------------
grid_type_dict = {'active': 1.,
                  'inactive': 0.,
                  'noflow_boundary':-1.,
                  'nearshore':-2.,
                  'coastline': -5.,
                  'river': -10.,
                  'waterbody': -15.}

grid_type_dict={'noflow':-1.,
                'active':1,
               'inactive':0,
               'reservoir':3,
               'river': -10.}
# ------- Raster i/o ------------
def gdal_error_handler(err_class, err_num, err_msg):
    '''
    Capture gdal error and report if needed
    
    Source:
    http://pcjericks.github.io/py-gdalogr-cookbook/gdal_general.html#install-gdal-ogr-error-handler
    '''
    errtype = {
            gdal.CE_None:'None',
            gdal.CE_Debug:'Debug',
            gdal.CE_Warning:'Warning',
            gdal.CE_Failure:'Failure',
            gdal.CE_Fatal:'Fatal'
    }
    err_msg = err_msg.replace('\n',' ')
    err_class = errtype.get(err_class, 'None')
    print('Error Number: %s' % (err_num))
    print('Error Type: %s' % (err_class))
    print('Error Message: %s' % (err_msg))

def load_grid_prj(fname=None,gt_out=False):
    
    indataset = gdal.Open(fname)
    proj_wkt = indataset.GetProjectionRef()
    if gt_out:
        gt=indataset.GetGeoTransform()
        indataset = None 
        return proj_wkt,gt
    else:
        indataset=None
        return proj_wkt
    
def get_rast_info(rast):
    indataset = gdal.Open(rast)
    nrows = indataset.RasterYSize
    ncols = indataset.RasterXSize
    gt=indataset.GetGeoTransform()
    indataset=None
    return nrows,ncols,gt

def xy_from_affine(tform=None,nx=None,ny=None):
    X,Y = np.meshgrid(np.arange(nx)+0.5,np.arange(ny)+0.5)*tform
    return X,Y

def read_griddata(in_fname,band=0,in_extent=None):
    with rasterio.open(in_fname) as src:
        data = src.read()[band]
        data[data==src.nodata]=np.nan
        ny,nx = data.shape
        X,Y = xy_from_affine(src.transform,nx,ny)
        
    if in_extent is not None:
        locate_extent_x = ((X>=in_extent[0]) & (X<=in_extent[2])).nonzero()[1]
        locate_extent_y = ((Y>=in_extent[1]) & (Y<=in_extent[3])).nonzero()[0]
        minx,maxx = locate_extent_x.min(),locate_extent_x.max()+1
        miny,maxy = locate_extent_y.min(),locate_extent_y.max()+1
        
        X,Y,data = X[miny:maxy,minx:maxx],Y[miny:maxy,minx:maxx],data[miny:maxy,minx:maxx]
    
    return X,Y,data


def write_gdaltif(fname,X,Y,Vals,rot_xy=0.,
                  proj_wkt=None,set_proj=True,
                  nan_val = -9999.,dxdy=[None,None],
                  geodata=None):
    '''Write geotiff to file from numpy arrays.
    
    Inputs
    ----------------
    fname: str
           full path and extension of the raster file to save.
         
    X: np.ndarray
           Array of X coordinates of array Vals.

    Y: np.ndarray
           Array of Y coordinates of array Vals.
    
    Vals: np.ndarray
           Array of the raster values to save as a geotiff.
           
    rot_xy: float
           The rotation of the model grid in radians. Automatically calculated
           if rot_xy=0 using grid_utils.grid_rot.
           
    proj_wkt: wkt str, osr.SpatialReference
           Well-known text or spatial reference object that specifies the 
           coordinate system of the X,Y grids to provide projection information
           that will be saved in the geotiff.
           
    set_proj: bool
           True to save the geotiff with coordinate system information provided
           by proj_wkt. If proj_wkt=None and set_proj=True, assumes that proj_wkt='NAD83'.       
           
    nan_val: float
           Not a number value to store in the geotiff.
           
    dxdy: list
           List giving [dx,dy]. If dxdy=[None,None], dxdy is automatically
           calculated from the X,Y arrays with grid_utils.calc_dxdy.  
         
    geodata: list
           List defining raster geographic information in gdal format
            [top left x, w-e pixel resolution, rotation, 
             top left y, rotation, n-s pixel resolution] 
           
    Outputs
    ----------------
    
    None
    
    Example
    --------------
    
    >>> # Assumes X,Y, data_array already stored in memory
    >>> filename = os.path.join(r'C:/research_dir','grid.tif')
    >>> output_dict = {'fname':filename,'proj_wkt':'NAD83','X':X,'Y':Y,'Vals':data_array}
    >>> grid_utils.read_griddata(**output_dict)
    
    '''
    # Create gtif
    # install error handler
    gdal.PushErrorHandler(gdal_error_handler)
    
    driver = gdal.GetDriverByName("GTiff")
    ds_out = driver.Create(fname, Vals.shape[1], Vals.shape[0], 1, gdal.GDT_Float32,options = [ 'COMPRESS=LZW' ] )
    

        
    # [top left x, w-e pixel resolution, rotation, 
    #  top left y, rotation, n-s pixel resolution]   
    if geodata is None:
        [X,Y,Vals],geodata = make_geodata(X,Y,Vals,rot_xy=rot_xy,
                                            dxdy=dxdy)  

    ds_out.SetGeoTransform(geodata)
    
    Vals_out = Vals.copy()
    Vals_out[np.isnan(Vals_out)]=nan_val

    if set_proj:
        # set the reference info
        srs = osr.SpatialReference()
        if proj_wkt is None:
            srs.SetWellKnownGeogCS("NAD83")
        else:
            srs.ImportFromWkt(proj_wkt)
            if len(srs.ExportToWkt())==0:
                srs.ImportFromProj4(proj_wkt)
            
        ds_out.SetProjection(srs.ExportToWkt())
        
    # write the band    
    outband = ds_out.GetRasterBand(1)
    outband.SetNoDataValue(nan_val)
    outband.WriteArray(Vals_out)
    outband.FlushCache()
    ds_out,outband = None, None

def make_geodata(X,Y,Vals,rot_xy=0.,
                 dxdy=[None,None]):
    # Check grid rotation
    if rot_xy==0.:
        [X,Y],Vals,rot_xy = grid_rot(XY=[X,Y],val=Vals)
    
    if dxdy[0] is None:
        # Calculate dx,dy taking into account rotation
        dx,dy = calc_dxdy(XY=[X,Y],rot_xy=rot_xy)
    else:
        dx,dy = dxdy
    
    # need to move origin from cell center to top left node of grid cell
    geodata = [X[0,0]-np.cos(-rot_xy)*dx/2.+np.sin(-rot_xy)*dx/2.,
               np.cos(-rot_xy)*dx,
               -np.sin(-rot_xy)*dx,
               Y[0,0]-np.cos(-rot_xy)*dy/2.-np.sin(-rot_xy)*dy/2.,
               np.sin(-rot_xy)*dy,
               np.cos(-rot_xy)*dy]  
    return [X,Y,Vals],geodata

def save_nc(fname=None,out_data_dict=None,out_desc=None):
    '''Save netcdf4 file.
    '''
    
    nc_out = netCDF4.Dataset(fname,'w', format='NETCDF4')
    if 'out_desc' not in list(out_data_dict.keys()):
        nc_out.description = r'No description'
    else:
        nc_out.description = out_data_dict['out_desc']
    
    # Assign dimensions
    for dim in out_data_dict['dims']['dim_order']:
        nc_out.createDimension(dim,out_data_dict['dims'][dim]['data'].size)
        dim_var = nc_out.createVariable(dim,'f8',(dim,),zlib=True)
        dim_var.setncatts(out_data_dict['dims'][dim]['attr'])
        nc_out.variables[dim][:] = out_data_dict['dims'][dim]['data']
    
    # Assign data arrays
    for ikey in out_data_dict['vars']:
        data_var = nc_out.createVariable(ikey,'f8',out_data_dict[ikey]['dims'],zlib=True)
        data_var.setncatts(out_data_dict[ikey]['attr'])        
        nc_out.variables[ikey][:]=  out_data_dict[ikey]['data']
    
    nc_out.close()
        
def load_nc(fname=None):
    '''Load netcdf4 file.
    '''
    out_dict = {}
    f = netCDF4.Dataset(fname)
    for var_name in list(f.variables.keys()):
        out_dict.update({var_name:{'data':f.variables[var_name][:],
                                    'long_name':f.variables[var_name].long_name,
                                    'var_desc':f.variables[var_name].var_desc,
                                    'units':f.variables[var_name].units}})
    
    return out_dict

def save_txtgrid(fname=None,data=None,delimiter=',',header=None):
    with open(fname,'w') as f_out:
        if (header is not None):
            f_out.write(header)
        for data_line in data:
            f_out.write('{}\n'.format(delimiter.join(data_line.astype('|S'))))
        f_out.close()

def read_txtgrid(fname=None,delimiter=',',comment='#'):
    with open(fname,'r') as f_in:
        load_data = []
        header_info = []
        for iline in f_in:
            if iline[0] in [comment]:
                header_info.append(iline.strip('\n'))
            else:
                pieces = iline.split(delimiter)
                try:
                    idata = [int(piece) for piece in pieces]
                except:
                    try:
                        idata = [float(piece) for piece in pieces]
                    except:
                        idata = pieces
                load_data.append(idata)
                
        f_in.close()
    return load_data,header_info

# --------- Grid transformations/information --------------
def grid_rot(XY=None,val=None):
    '''Rotate matrixes to set origin at top left row,col=0,0.'''
    x,y=XY
    h=val.copy()
    # Reorient matrixes based on xy orientation
    xmin_inds = np.unravel_index(np.argmin(x),x.shape)
    ymax_inds = np.unravel_index(np.argmax(y),y.shape)
    
    if xmin_inds[1]==x.shape[0]-1:
        # Flip column axis
        x,y = x[:,::-1],y[:,::-1]
        if len(h.shape)==3:
            h = h[:,:,::-1]
        else:
            h = h[:,::-1]

    if ymax_inds[0]!=0:
        # Flip row axis
        x,y = x[::-1,:],y[::-1,:]
        if len(h.shape)==3:
            h = h[:,::-1,:]
        else:
            h = h[::-1,:]

    # Calculate grid rotation
    grid_rot = np.arctan2(y[0,1]-y[0,0],x[0,1]-x[0,0])
    
    return [x,y],h,grid_rot

def calc_dxdy(XY=None,rot_xy=0,ndec=5):
    '''Calculate spatial discretization with rotation.'''
    tempX,tempY = XY
    if rot_xy==0:
        dx=tempX[0,1]-tempX[0,0]
        dy=tempY[1,0]-tempY[0,0]
    else:
        x0,y0 = tempX[0,0],tempY[0,0]
        newX = xrot(tempX-x0,tempY-y0,-rot_xy)
        newY = yrot(tempX-x0,tempY-y0,-rot_xy)
        dx = np.round(newX[0,1]-newX[0,0],decimals=ndec)
        dy = np.round(newY[1,0]-newY[0,0],decimals=ndec)
    return dx,dy

def load_and_griddata(fname,new_xy,in_extent=None,mask=None,
                      interp_method = 'linear',ideal_cell_size=None):
    '''
    Load raster dataset (fname) and re-grid to new raster cells specified by new_xy
    
    interp_method: 'linear': uses bilinear grid interpoloation
                   'median': uses median filter, when cell_spacing_orig << cell_spacing_new
                   other: can assign bindata2d function (e.g., np.median, np.std, np.mean) 
                   
    '''                      
    X_temp,Y_temp,Grid_val = read_griddata(fname,in_extent=in_extent)
    
    # Decimate grid to lower resolution from ultrahigh res dataset
    if ideal_cell_size is not None:
        X_temp,Y_temp,Grid_val = decimate_raster(X_temp,Y_temp,Grid_val,
                                                 ideal_cell_size=ideal_cell_size)
    if interp_method.lower() in ('linear','bilinear'):
        if (mask is not None):
            new_xy[0] = np.ma.masked_array(new_xy[0],mask=mask)
            new_xy[1] = np.ma.masked_array(new_xy[1],mask=mask)   
                 
        out_grid = subsection_griddata([X_temp,Y_temp],Grid_val,new_xy)
    elif interp_method in ('median'):
        out_grid = bindata2d([X_temp,Y_temp],Grid_val,new_xy)
    else:
        out_grid = bindata2d([X_temp,Y_temp],Grid_val,new_xy,stat_func=interp_method)
    return out_grid

def bindata2d(XY_orig,Z_orig,XY_new,stat_func=np.median):
    X,Y = XY_new    
    dx,dy = X[0,1]-X[0,0],Y[1,0]-Y[0,0]
    xbins = np.hstack([X[0,0]-dx/2.,X[0,:]+dx/2.])
    ybins = np.hstack([Y[0,0]-dy/2.,Y[:,0]+dy/2.])
    nan_inds = np.isnan(Z_orig)
    Z_new = binned_statistic_2d(XY_orig[1][~nan_inds],XY_orig[0][~nan_inds],values=Z_orig[~nan_inds], statistic=stat_func, bins=[ybins,xbins])
    count_mask = Z_new.statistic==0.
    Z_new.statistic[count_mask] = np.nan
    return Z_new.statistic

def subsection_griddata(orig_xy,orig_val,new_xy,nsections=20.,min_ndxy = 25.,
                        active_method='linear'):
    
    # Unpack inputs
    X_temp,Y_temp = orig_xy
    
    if isinstance(nsections,float) or isinstance(nsections,int):
        nsections = [np.float(nsections),np.float(nsections)] # convert to list
        
    
    # Set up subsection indexes
    ny,nx = new_xy[0].shape
    sections_dy,sections_dx = np.ceil(ny/nsections[1]),np.ceil(nx/nsections[0])
    
    # Want at least min_ndxy number of points per dimension in a subsection
    if sections_dy < min_ndxy:
        sections_dy = min_ndxy
    
    if sections_dx < min_ndxy:
        sections_dx = min_ndxy
        
    sstart_y,sstart_x = np.arange(0,ny,sections_dy,dtype=np.int),np.arange(0,nx,sections_dx,dtype=np.int)
    send_y,send_x = np.roll(sstart_y,-1),np.roll(sstart_x,-1)
    send_y[-1],send_x[-1] = ny,nx
    
    # Initiate output and loop
    try:
        val_mask = new_xy[0].mask # already masked
    except:
        # Make mask
        val_mask = np.isnan(new_xy[0]) # probably none unless previously set to nan
    
    new_val = val_mask*np.nan*np.zeros_like(new_xy[0])
        
    icount = -1
    
    # Loop
    for irow,(rowstart,rowend) in enumerate(zip(sstart_y,send_y)):
        for icol,(colstart,colend) in enumerate(zip(sstart_x,send_x)):
            icount+=1
            in_x = new_xy[0][rowstart:rowend,colstart:colend]
            in_y = new_xy[1][rowstart:rowend,colstart:colend]
            
            if in_y[~val_mask[rowstart:rowend,colstart:colend]].shape[0]==0 or \
                len(in_y)==0 or len(in_x)==0:
#                print '{},{} have no active cells'.format(irow,icol)
                continue
            else:
                
                temp_extent = [in_x.min(),in_x.max(),in_y.min(),in_y.max()]
                buffer0 = 3*np.abs(np.diff(X_temp,axis=1).mean()+1j*np.diff(Y_temp,axis=0).mean()) # mean diagonal for cells
                inpts = (X_temp<=temp_extent[1]+buffer0) & (X_temp>=temp_extent[0]-buffer0) \
                        & (Y_temp<=temp_extent[3]+buffer0) & (Y_temp>=temp_extent[2]-buffer0)
                if len(inpts.nonzero()[0])>0:
                    if len(np.unique(Y_temp[inpts]))>1 and len(np.unique(X_temp[inpts]))>1:
                        new_val_temp = griddata(np.c_[X_temp[inpts],Y_temp[inpts]],orig_val[inpts],(in_x,in_y),method=active_method)
                        new_val[rowstart:rowend,colstart:colend] = new_val_temp.copy()
                    else:
                        # Only one line of unique values, assign nan
#                        new_val_temp = griddata(np.c_[X_temp[inpts],Y_temp[inpts]],orig_val[inpts],(in_x,in_y),method='nearest')
                        new_val[rowstart:rowend,colstart:colend] = np.nan
    
    return new_val
#%% ----------- General array utilities ----------------------
#
#def define_mask(cc_XY,active_indexes=None):
#    bool_out = np.zeros_like(cc_XY[0],dtype=bool)
#    bool_out[active_indexes[0],active_indexes[1]] = True
#    return bool_out

def define_mask(shp=None,rast_template=None,
              bands=[1],options=['ALL_TOUCHED=TRUE'],burn_values=[1],
              trans_arg1=None,trans_arg2=None,nan_val=0,out_bool=True):
    if isinstance(shp,str):
        shp_ds = ogr.Open(shp)
        shp_layer = shp_ds.GetLayer()
    elif hasattr(shp,'geom_type'):
        # Convert from pyshp to ogr
        shp_layer = ogr.CreateGeometryFromWkt(shp.to_wkt())
    else:
        shp_layer = shp
    
    if isinstance(rast_template,str):
        nrows,ncols,gd = get_rast_info(rast_template)
    else:
        nrows,ncols,gd = rast_template
    
    dst_ds = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1 ,gdal.GDT_Int32)
    dst_rb = dst_ds.GetRasterBand(1)
    dst_rb.Fill(nan_val) #initialise raster with nans
    dst_rb.SetNoDataValue(nan_val)
    dst_ds.SetGeoTransform(gd)

    _ = gdal.RasterizeLayer(dst_ds,bands,shp_layer,
                        trans_arg1,trans_arg2,burn_values=burn_values,
                        options=options)
    dst_ds.FlushCache()
    
    mask_arr=dst_ds.GetRasterBand(1).ReadAsArray()
    
    if out_bool:
        return mask_arr.astype(bool)
    else:
        return mask_arr

def clean_ibound(ibound,min_area=None,check_inactive=False):
    '''
    Removes isolated active cells from the IBOUND array.
    
    Assumes only active and inactive ibound conditions (i.e., no constant heads).
    
    Source: modified after Wesley Zell, PyModflow.pygrid.grid_util.clean_ibound, Jul 17 2013
    '''
    from scipy.ndimage import measurements
    # Distinguish disconnected clusters of active cells in the IBOUND array.
    cluster_ibound = ibound.copy()
    cluster_ibound[ibound != 0] = 1
    
    
    array_of_cluster_idx,num = measurements.label(cluster_ibound)
    
    # Identify the cluster with the most active cells; this is the main active area
    areas = measurements.sum(cluster_ibound,array_of_cluster_idx,\
                             index=np.arange(array_of_cluster_idx.max()+1))
    
    clean_ibound_array = np.zeros_like(ibound)                         
    if (min_area is None):
        # Use only largest area
        cluster_idx = np.argmax(areas)
        # Activate all cells that belong to primary clusters
        clean_ibound_array[array_of_cluster_idx == cluster_idx] = 1
    else:
        cluster_idx = (areas >= min_area).nonzero()[0]
        
        # Activate all cells that belong to primary clusters
        for idx_active in cluster_idx:
            clean_ibound_array[array_of_cluster_idx==idx_active] = 1

    if check_inactive:
        # Identify inactive clusters surrounded by active cells
        
        cluster_ibound2 = 1-clean_ibound_array.copy() # Flip values
        clean_ibound_array2 = clean_ibound(cluster_ibound2,min_area=min_area)
        clean_ibound_array[clean_ibound_array2==1] = 0
        clean_ibound_array[clean_ibound_array2==0] = 1
        
        
    return clean_ibound_array  

def unique_rows(a,sort=True,return_inverse=False):
    '''
    Find unique rows and return indexes of unique rows
    '''
    a = np.ascontiguousarray(a)
    unique_a,uind,uinv = np.unique(a.view([('', a.dtype)]*a.shape[1]),return_index=True,return_inverse=True)
    if sort:    
        uord = [(uind==utemp).nonzero()[0][0] for utemp in np.sort(uind)]
        outorder = uind[uord]
    else:
        outorder = uind
    if return_inverse:
        return unique_a,uind,uinv
    else:
        return outorder

def remove_lrc(lrc_array=None,remove_lrc_array=None):
    '''Remove cellid's for cells outside domain.'''
#    max_ind = lrc_array.shape[0]
    joined_array= np.vstack([remove_lrc_array,lrc_array])
#    unq_inds = unique_rows(joined_array)
    all_cells,uinds,uinv = unique_rows(joined_array,sort=False,return_inverse=True)
    uord = [(uinds==utemp).nonzero()[0][0] for utemp in np.sort(uinds)]
#    outorder = uinds[uord]
    
    # Find repeated indexes
    repeated_inds = uord[remove_lrc_array.shape[0]:]
    keep_inds = uinds[repeated_inds]-remove_lrc_array.shape[0]
#    keep_inds = [uind for uind in uinds if uind not in repeated_inds]
#    keep_inds = [uind for uind in out_inds if uind < max_ind]
#    keep_inds = unq_inds[unq_inds<max_ind]
    return keep_inds

def raster_edge(cell_types=None,search_val=-2,invalid_val=0,
                size=3,zsize=20,min_area=None,bool_array=None):
    '''Define a raster edge with additional boolean array.'''
    from scipy.ndimage.filters import minimum_filter
    from scipy.ndimage import measurements
    edge_bool=minimum_filter(np.abs(cell_types),size=size,mode='nearest') == invalid_val
    
    #Z2 = -np.ma.masked_invalid(Z) # mask and convert depth
    
    # Find where min_Z is the invalid value but original was search_val
    if search_val == -2:
        # Find cells far from land
        offshore_bool = minimum_filter(-cell_types,size=zsize,mode='nearest') >= invalid_val
        bool_out = edge_bool & (cell_types==search_val) & offshore_bool
    elif search_val == 1:
        bool_out = edge_bool & (cell_types==search_val)
    else:
        # find any boundary
        bool_out = edge_bool & (cell_types != invalid_val)

    if bool_array is not None:
        bool_out = bool_out & bool_array
    # Select longest continuous selection
        
    # Distinguish disconnected clusters of active cells in the IBOUND array.
    cluster_array = bool_out.copy().astype(np.int)
    
    array_of_cluster_idx,num = measurements.label(cluster_array)
    
    # Identify the cluster with the most active cells; this is the main active area
    areas = measurements.sum(cluster_array,array_of_cluster_idx,\
                             index=np.arange(array_of_cluster_idx.max()+1))
    
    clean_bool_array = np.zeros_like(bool_out)                         
    if (min_area is None):
        # Use only largest area
        cluster_idx = np.argmax(areas)
        # Activate all cells that belong to primary clusters
        clean_bool_array[array_of_cluster_idx == cluster_idx] = 1
    else:
        cluster_idx = (areas >= min_area).nonzero()[0]
        # Activate all cells that belong to primary clusters
        for idx_active in cluster_idx:
            clean_bool_array[array_of_cluster_idx==idx_active] = 1    

    return clean_bool_array.astype(bool)            
    
def calc_dist(xy1,xy2,**kwargs):
    '''Calculate euclidean distance.'''
    from scipy.spatial import distance
    
    dist_mat = distance.cdist(xy1,xy2,**kwargs)
    return dist_mat

def fill_mask(in_array,fill_value=np.nan):
    if hasattr(in_array,'mask'):
        out_array = np.ma.filled(in_array.copy(),fill_value)
    else:
        out_array = in_array.copy()
        
    if ~np.isnan(fill_value):
        out_array[np.isnan(out_array)]=fill_value
        
    return out_array

def fill_nan(array=None,fill_value=0.):
    if isinstance(array,np.ndarray):
        array[np.isnan(array)] = fill_value
    return array

def remove_nan_rc(X,Y,Z,return_mask=False):
    '''Remove columns and rows with only null values.
    '''
    # Conslidate masks
    mask = np.isnan(Z)
    if hasattr(X,'mask'):
        X2 = np.ma.getdata(X.copy())
    else:
        X2 = X.copy()
    X2[mask] = np.nan
    if hasattr(Y,'mask'):
       Y2 = np.ma.getdata(Y.copy())
    else:
       Y2 = Y.copy()
    
#    Y2[mask] = np.nan
    X2 = np.ma.masked_array(X2,mask=mask)
    Y2 = np.ma.masked_array(Y2,mask=mask)
    xleft = np.nanmin(X2,axis=1)
#    xleft[np.isnan(xleft)] = np.nanmax(xleft)
    xright = np.nanmax(X2,axis=1)
#    xright[np.isnan(xright)] = np.nanmin(xright)
    ytop = np.nanmax(Y2,axis=0)
#    ytop[np.isnan(ytop)] = np.nanmin(ytop)
    ybot = np.nanmin(Y2,axis=0)
#    ybot[np.isnan(ybot)] = np.nanmax(ybot)
    
    # Find first and last indices to keep
    x0=(xleft.min()==X2[np.argmin(xleft),:]).nonzero()[0][0]
    x1=(xright.max()==X2[np.argmax(xright),:]).nonzero()[0][0]
    y0=(ybot.min()==Y2[:,np.argmin(ybot)]).nonzero()[0][0]
    y1=(ytop.max()==Y2[:,np.argmax(ytop)]).nonzero()[0][0]    
    X2,Y2 = [],[]
    
    x0,x1 = np.sort([x0,x1])
    y0,y1 = np.sort([y0,y1])
    if return_mask:
        mask_out = np.zeros_like(X,dtype=bool)
        mask_out[y0:y1+1,x0:x1+1] = 1
        return X.copy()[y0:y1+1,x0:x1+1],Y.copy()[y0:y1+1,x0:x1+1],Z.copy()[y0:y1+1,x0:x1+1],mask_out
    else:
        return X.copy()[y0:y1+1,x0:x1+1],Y.copy()[y0:y1+1,x0:x1+1],Z.copy()[y0:y1+1,x0:x1+1]

def get_extent(XY):
    return [np.nanmin(XY[0]),np.nanmin(XY[1]),np.nanmax(XY[0]),np.nanmax(XY[1])]

def reduce_extent(in_extent,inx,iny, buffer_size=0,fill_mask_bool=True):
    '''
    Select portions of x,y cooridnates within in_extent
    
    in_extent = [minx,miny,maxx,maxy]
    
    '''
    if fill_mask_bool:
        inx = fill_mask(inx,np.inf)
        iny = fill_mask(iny,np.inf)
        
    inpts = (inx>=in_extent[0]-buffer_size) & (inx<=in_extent[2]+buffer_size) &\
            (iny>=in_extent[1]-buffer_size) & (iny<=in_extent[3]+buffer_size) 
    if len(inpts) > 0:
        return inx[inpts],iny[inpts],inpts
    else:
        return [],[],[]
    
def shrink_domain(dicts_obj,XY=None,inactive_cell_type=grid_type_dict['inactive']):
    '''Remove rows and columns with only inactive cells.
    '''
    cell_types_array = dicts_obj.cell_types.copy()
    cell_types_array[cell_types_array==inactive_cell_type] = np.nan # set inactive cells to nan
    if (XY is None):
        x=np.arange(0,cell_types_array.shape[1],dtype=float)
        y=np.arange(0,cell_types_array.shape[0],dtype=float)
        XY = np.meshgrid(x,y)
        
    newX,newY,new_cell_types,old2new_mask = remove_nan_rc(XY[0],XY[1],cell_types_array,return_mask=True)
    dicts_obj.dis_obj.nrow,dicts_obj.dis_obj.ncol = newX.shape
    new_cell_types[np.isnan(new_cell_types)] = inactive_cell_type # reset nan's to inactive
    dicts_obj.cell_types = new_cell_types
    return old2new_mask 

def decimate_raster(X_temp,Y_temp,Grid_val,ideal_cell_size=None,ndecimate_in=None):
    if ndecimate_in is not None and ndecimate_in>1:
        X_temp = X_temp[::ndecimate_in,::ndecimate_in]
        Y_temp = Y_temp[::ndecimate_in,::ndecimate_in]
        Grid_val = Grid_val[::ndecimate_in,::ndecimate_in]
    
    max_grid_dxy = np.max([np.abs(np.diff(X_temp,axis=1).mean()),np.abs(np.diff(Y_temp,axis=0).mean())])
    if ideal_cell_size is not None:
        cell_size_ratio = ideal_cell_size/max_grid_dxy
        if cell_size_ratio > 5:
            rows,cols = X_temp.shape
            ntimes_gt_ideal = 3
            ndecimate = np.int(np.floor(cell_size_ratio)/ntimes_gt_ideal)
            X2 = X_temp[::ndecimate,::ndecimate]
            Y2 = Y_temp[::ndecimate,::ndecimate]
            Grid_val2 = Grid_val[::ndecimate,::ndecimate]
            # check to see if last entries of arrays are the same (i.e., don't lose boundary values)
            xfix_switch = False
            if X2[0,-1] != X_temp[0,-1]:
                # Add last column into array
                X2 = np.hstack([X2,X_temp[::ndecimate,-1].reshape((-1,1))])
                Y2 = np.hstack([Y2,Y_temp[::ndecimate,-1].reshape((-1,1))])
                Grid_val2 = np.hstack([Grid_val2,Grid_val[::ndecimate,-1].reshape((-1,1))])
                xfix_switch = True
            
            if Y2[-1,0] != Y_temp[-1,0]:
                # Add last row to array
                if xfix_switch:
                    print(X2.shape,np.hstack([X_temp[-1,::ndecimate],X_temp[-1,-1]]).reshape((1,-1)).shape)
                    X2 = np.vstack([X2,np.hstack([X_temp[-1,::ndecimate],X_temp[-1,-1]]).reshape((1,-1))])
                    Y2 = np.vstack([Y2,np.hstack([Y_temp[-1,::ndecimate],Y_temp[-1,-1]]).reshape((1,-1))])
                    Grid_val2 = np.vstack([Grid_val2,np.hstack([Grid_val[-1,::ndecimate],Grid_val[-1,-1]]).reshape((1,-1))])
                else:
                    X2 = np.vstack([X2,X_temp[-1,::ndecimate].reshape((1,-1))])
                    Y2 = np.vstack([Y2,Y_temp[-1,::ndecimate].reshape((1,-1))])
                    Grid_val2 = np.vstack([Grid_val2,Grid_val[-1,::ndecimate].reshape((1,-1))])
            
            X_temp,Y_temp,Grid_val = X2,Y2,Grid_val2
    
        
    return X_temp, Y_temp, Grid_val

# -------------- Modflow grid creation -----------------
def make_zbot(ztop,grid_dis,delv,zthick_elev_min=None):
    '''Make cell bottom elevation array.
    '''
    nlay,nrow,ncol = grid_dis
    if ~isinstance(delv,(list,np.ndarray)):
        delv_array = delv*np.ones(nlay) # m
        delv_cumulative=np.cumsum(delv_array)    
    else:
        delv_cumulative=np.cumsum(delv)

    zbot = [] # for multiple layers
    for ilay in np.arange(nlay):
        zbot.append(ztop-delv_cumulative[ilay])
    
    zbot = np.array(zbot)
    zbot = zbot.reshape((nlay,nrow,ncol))# mainly for nlay=1
    
    if zthick_elev_min is not None:
        zbot = adj_zbot(zbot,ztop,zthick_elev_min) # ensure no negative thickness cells
    
    return zbot

def adj_zbot(zbot,ztop,zthick_elev_min=None):
    '''Adjust cell bottom elevations.
    '''
    if zthick_elev_min is not None: 
        if zthick_elev_min > 0:
            # Minimum thickness
            zbot[-1,:,:] = np.minimum(ztop-zthick_elev_min,zbot[-1,:,:])
        else:
            # Minimum elevation
            zbot[-1,:,:] = np.minimum(zthick_elev_min,zbot[-1,:,:]) # ensure no negative thickness cells
        
    return zbot