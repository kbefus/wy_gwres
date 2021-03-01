# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 08:44:02 2017

@author: kbefus
"""
from __future__ import print_function
import numpy as np
from osgeo import gdal,osr
from flopy.utils.reference import SpatialReference as fusr


lenuni = {'undefined': 0,
         'feet': 1,
         'meters': 2,
         'centimeters': 3}

itmuni = {0: "undefined", 1: "seconds", 2: "minutes", 3: "hours",
          4: "days",
          5: "years"}

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

def define_UTM_zone(input_extent,use_negative=True):
    ''' Find UTM zone of domain center.
    
    Inputs
    --------
    input_extent: list,np.ndarray
        input bounds of feature for defining UTM zone, [xmin,ymin,xmax,ymax]
    '''
    meanx = np.mean(input_extent[::2])
    if meanx < 0. and use_negative:
        utm_edges=np.arange(-180.,186.,6.)
    else:
        utm_edges=np.arange(0.,366.,6.)
    utm_zones=np.arange(1,utm_edges.shape[0])
    west_boundary = utm_edges[:-1]
    east_boundary = utm_edges[1:]
    UTM_ind = ((meanx>=west_boundary) & (meanx<=east_boundary)).nonzero()[0][0]
    output_zone = utm_zones[UTM_ind]
    return output_zone

def xrot(x,y,theta):
    return x*np.cos(theta)-y*np.sin(theta)
    
def yrot(x,y,theta):
    return x*np.sin(theta)+y*np.cos(theta)

def rot_shift(XY,xyshift=[0,0],rad_angle=0, reverse=False):
    if reverse:
        # Shift and then rotate
        rotshift_x = xrot(XY[0]+xyshift[0],XY[1]+xyshift[1],-rad_angle)
        rotshift_y = yrot(XY[0]+xyshift[0],XY[1]+xyshift[1],-rad_angle)
    else:
        # Rotate and then shift
        rotshift_x = xrot(XY[0],XY[1],rad_angle)-xyshift[0]
        rotshift_y = yrot(XY[0],XY[1],rad_angle)-xyshift[1]
    
    return rotshift_x,rotshift_y

def projectXY(xy_source, inproj=None, outproj="NAD83"):
    '''
    Convert coordinates of raster source
    
    '''
    # install error handler
    gdal.PushErrorHandler(gdal_error_handler)
    
    if hasattr(outproj,'ImportFromWkt'):
        dest_proj=outproj
    else:
        dest_proj = osr.SpatialReference()
        if isinstance(outproj,(float,int)):
            dest_proj.ImportFromEPSG(outproj)
        elif '+' in outproj:
            dest_proj.ImportFromProj4(outproj)
        elif 'PROJCS' in outproj or 'GEOGCS' in outproj:
            dest_proj.ImportFromWkt(outproj)
        elif hasattr(outproj,'ImportFromWkt'):
            dest_proj = outproj # already an osr sr
        else:
            # Assume outproj is geographic sr
            dest_proj.SetWellKnownGeogCS(outproj)
    
    if hasattr(inproj,'ImportFromEPSG'):
        src_proj=inproj
    else:
        src_proj = osr.SpatialReference()
        if isinstance(inproj,(float,int)):
            src_proj.ImportFromEPSG(inproj)
        elif '+' in inproj:
            src_proj.ImportFromProj4(inproj)
        elif 'PROJCS' in inproj or 'GEOGCS' in inproj:
            src_proj.ImportFromWkt(inproj)
        elif hasattr(inproj,'ImportFromWkt'):
            src_proj = inproj # already an osr sr
        else:
            # Assume outproj is geographic sr
            src_proj.SetWellKnownGeogCS(inproj)
        
    # the ct object takes and returns pairs of x,y, not 2d grids
    # so the the grid needs to be reshaped (flattened) and back.
    ct = osr.CoordinateTransformation(src_proj, dest_proj)
    
    if len(xy_source.shape) == 3:
        shape = xy_source[0,:,:].shape
        size = xy_source[0,:,:].size
        xy_source = xy_source.reshape(2, size).T
        xy_target = np.array(ct.TransformPoints(xy_source))
        xx = xy_target[:,0].reshape(shape)
        yy = xy_target[:,1].reshape(shape)
        return xx, yy
    else:
        transposed = False
        if xy_source.shape[0]<xy_source.shape[1]:
            xy_source = xy_source.T # want n x 2 matrix
            transposed = True        

        xy_target = np.array(ct.TransformPoints(xy_source))[:,:2] # only want first 2 columns
    
        if transposed:
            xy_target = xy_target.T
            
        return xy_target

def osr_transform(XY=None,mf_model=None,xyul_m=None,
                          xyul=[0,0],rotation=0.,
                          proj_in=None,proj_out="NAD83",param_opts=None):
    '''Use GDAL/OSR to transform from model to projected coordinates.
    
    Note: xyul is not the xyshift from defining the domain, but the upper left 
            [x,y] pt of the model in a projected (e.g., UTM) coordinate system.
    '''
    
    if xyul_m is None:
        # Load node cooridnates
        y,x,z = mf_model.dis.get_node_coordinates()
        
        # Select upper left node coordinates
        xul_m,yul_m = x[0],y[0]
    else:
        xul_m,yul_m = xyul_m
    
    X_out = xrot(XY[0]-xul_m,XY[1]-yul_m,rotation)+xyul[0]
    Y_out = yrot(XY[0]-xul_m,XY[1]-yul_m,rotation)+xyul[1]
    
    xy_source = np.array([X_out,Y_out])
    
    shape=xy_source[0,:,:].shape
    size=xy_source[0,:,:].size

    if (proj_out is not None) and (proj_in is not None):
        # Define coordinate systems
        src_proj = osr.SpatialReference()
        dest_proj = osr.SpatialReference()
        
        if isinstance(proj_in,(int,float)):
            src_proj.ImportFromEPSG(proj_in)
        elif hasattr(proj_in,'ImportFromWkt'):
            src_proj=proj_in
        elif '+' in proj_in:
            src_proj.ImportFromWkt(proj_in)
        else:
            src_proj.SetWellKnownGeogCS(proj_in) # assume geographic coordinates
            
        if isinstance(proj_out,(int,float)):
            dest_proj.ImportFromEPSG(proj_out)
        elif hasattr(proj_out,'ImportFromWkt'):
            dest_proj=proj_out
        elif '+' in proj_out:
            dest_proj.ImportFromWkt(proj_out)
        else:
            dest_proj.SetWellKnownGeogCS(proj_out) # assume geographic coordinates
        
        # Check for changes to coordinate systems
        if param_opts is not None:
            if 'proj_in' in list(param_opts.keys()):
                if 'Params' in param_opts['proj_in'].keys():
                    for ikey in param_opts['proj_in']['Params'].keys():
                        src_proj.SetProjParm(ikey,param_opts['proj_in']['Params'][ikey])
                if 'LinearUnits' in param_opts['proj_in'].keys():
                    ikey = list(param_opts['proj_in']['LinearUnits'].keys())[0]
                    src_proj.SetLinearUnits(ikey,param_opts['proj_in']['LinearUnits'][ikey])
                
            if 'proj_out' in param_opts.keys():
                if 'Params' in param_opts['proj_out'].keys():
                    for ikey in param_opts['proj_out']['Params'].keys():
                        dest_proj.SetProjParm(ikey,param_opts['proj_out']['Params'][ikey])
                if 'LinearUnits' in param_opts['proj_out'].keys():
                    ikey = list(param_opts['proj_out']['LinearUnits'].keys())[0]
                    dest_proj.SetLinearUnits(ikey,param_opts['proj_out']['LinearUnits'][ikey])

        # Project to new coordinate system
        # the ct object takes and returns pairs of x,y, not 2d grids
        # so the the grid needs to be reshaped (flattened) and back.
        ct = osr.CoordinateTransformation(src_proj, dest_proj)
        xy_target = np.array(ct.TransformPoints(xy_source.reshape(2, size).T))
        X_out = xy_target[:,0].reshape(shape)
        Y_out = xy_target[:,1].reshape(shape)
    
    return X_out, Y_out

def modelcoord_transform(XY=None,mf_model=None,xyul_m=None, xyul=[0,0],rotation=0.,
                         proj_in=None,proj_out=None):
    '''Convert from model coordinates to other coordinate system.
    
    
    Note: xyul is not the xyshift from defining the domain, but the upper left 
            [x,y] pt of the model in a projected (e.g., UTM) coordinate system.
    
    '''
    if xyul_m is None:
        # Load node cooridnates
        y,x,z = mf_model.dis.get_node_coordinates()
        
        # Select upper left node coordinates
        xul_m,yul_m = x[0],y[0]
    else:
        xul_m,yul_m = xyul_m
    
    X_out = xrot(XY[0]-xul_m,XY[1]-yul_m,rotation)+xyul[0]
    Y_out = yrot(XY[0]-xul_m,XY[1]-yul_m,rotation)+xyul[1]

    if (proj_out is not None) and (proj_in is not None):
        # Project to new coordinate system
        xy_temp = proj_out.transform_points(proj_in,X_out,Y_out)
        if len(xy_temp.shape)==3:
            X_out,Y_out = xy_temp[:,:,0].reshape(X_out.shape),xy_temp[:,:,1].reshape(X_out.shape)
        else:
            X_out,Y_out = xy_temp[:,0].reshape(X_out.shape),xy_temp[:,1].reshape(X_out.shape)
    
    return X_out, Y_out    

def nodes_to_cc(XYnodes=None,grid_transform=None):
    '''
    
    Returns:
        Variables:  [cc_x,cc_y], [cc_x_proj,cc_y_proj],     [cc_x_latlong, cc_y_latlong]
        Dimensions: [[ny-1,nx-1],[ny-1,nx-1]]      , [[ny-1,nx-1],[ny-1,nx-1]], [[ny-1,nx-1],[ny-1,nx-1]]
        where ny,nx = Xcorners.shape
        
    Dependencies:
        Packages: Numpy, osgeo/gdal
        Functions: xrot, yrot
    '''
    Xcorners,Ycorners = XYnodes
    from_proj = grid_transform['proj']
    xyshift = grid_transform['xyshift']
    rot_angle = grid_transform['rotation']
    # Calculate cell centers from nodes (grid corners)
    cc_x = Xcorners[:-1,:-1]+np.diff(Xcorners[:-1,:],axis=1)/2.
    cc_y = Ycorners[:-1,:-1]+np.diff(Ycorners[:,:-1],axis=0)/2.
    
    # Unrotate cell_centers to get cell centers in projected coordinate system
    cc_x_proj = xrot(cc_x+xyshift[0],cc_y+xyshift[1],-rot_angle)
    cc_y_proj = yrot(cc_x+xyshift[0],cc_y+xyshift[1],-rot_angle)
    
    # Unproject cell centers to geographic coordinate system (NAD83)
    if from_proj is not None:
        cc_x_latlong,cc_y_latlong = projectXY(np.array([cc_x_proj,cc_y_proj]),from_proj)
    else:
        cc_x_latlong, cc_y_latlong = [],[]
    return [cc_x,cc_y],[cc_x_proj,cc_y_proj],[cc_x_latlong, cc_y_latlong]

def sim_to_flopysr(gt=None,sim_obj=None):
    '''Convert usgs.model.reference file to flopy spatial reference.
    '''
    sr_dict = {'delr':sim_obj.dis.delr,
               'delc':sim_obj.dis.delc,
               'lenuni':[ival.lower() for ival in list(lenuni.keys())].index(sim_obj.dis.length_units),
               'xul':gt['xul'],
               'yul':gt['yul'],
               'rotation':np.rad2deg(gt['rotation']),
               'prj':gt['proj'].ExportToWkt(),
               }
    sr = fusr(**sr_dict)

    return sr

def read_prj(fname=None):
    out_txt = []
    with open(fname,'r') as prj:
        for iline in prj:
            out_txt.append(iline)
    return out_txt[0]
    