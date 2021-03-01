# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 08:34:41 2017

Collection of functions to aid Modflow operations related to shapefiles

@author: kbefus
"""
from __future__ import print_function
import os
import shutil
from osgeo import osr
import numpy as np
import shapefile
from shapely.geometry import Polygon,MultiPoint,LineString,MultiPolygon
from shapely.geometry import box as shpbox
from shapely.prepared import prep
from shapely.affinity import rotate
from shapely.ops import cascaded_union
from shapely import speedups

import pandas as pd


speedups.enable()
from wy_gwres.utils import grid_utils
from wy_gwres.utils.proj_utils import xrot,yrot,projectXY,modelcoord_transform,osr_transform

#%% ---------- Shapefile i/o -----------------
def shp_to_polygon(in_polygon):
    '''Convert a shapefile-like item to a shapely polygon.
    
    Inputs
    --------
    
    in_polygon: str, pyshp obj, or shapely obj
        input polygon that can be one of the above formats that will be
        converted to a list of shapely polygons.
        
    Returns
    --------
    
    out_polygon: list
        list of shapely polygons.
        
    nshapes: int
        Number of shapes in list.
    
    '''
    if isinstance(in_polygon,str):
        shp1 = shapefile.Reader(in_polygon)
        in_polygon=[]
        for shape in shp1.shapes():
            parts = shape.parts
            points = shape.points
            if len(parts) > 2:
                for part1,part2 in zip(parts[:-1],parts[1:]):
                    in_polygon.append(Polygon(points[part1:part2]))
                else:
                    in_polygon.append(Polygon(points[part2:]))
            else:
                in_polygon.append(Polygon(points))
#        in_polygon = [Polygon(shape.points) for shape in shp1.shapes() #old way, can't handle holes
    
    if hasattr(in_polygon,'bbox'):
        if len(in_polygon.parts)>1:
            pts = in_polygon.points
            parts = np.hstack([in_polygon.parts,len(pts)])
            out_polygon = [Polygon(pts[parts[ipt]:parts[ipt+1]]) for ipt in range(len(parts)-1)]
        else:
            out_polygon = [Polygon(in_polygon.points)]
            
        nshapes = len(out_polygon) 
    else:
        if isinstance(in_polygon,list):
            nshapes = len(in_polygon)
            out_polygon=in_polygon
        elif isinstance(in_polygon,np.ndarray):
            nshapes = len(in_polygon)
            out_polygon=in_polygon
        else:
            if hasattr(in_polygon,'geom_type') and in_polygon.is_valid:
                if in_polygon.geom_type is 'MultiPolygon':
                    out_polygon = in_polygon.geoms
                    nshapes = len(out_polygon)
                else:
                    out_polygon = [in_polygon] # put single shape into a list
                    nshapes = len(out_polygon)
    return out_polygon,nshapes

def shp_to_line(in_polyline):
    if isinstance(in_polyline,str):
        shp1 = shapefile.Reader(in_polyline)
        in_poly=[]
        for shape in shp1.shapes():
            parts = shape.parts
            points = shape.points
            if len(parts) > 2:
                for part1,part2 in zip(parts[:-1],parts[1:]):
                    in_poly.append(LineString(points[part1:part2]))
                else:
                    in_poly.append(LineString(points[part2:]))
            else:
                in_poly.append(LineString(points))
        
    else:
        in_poly = in_polyline
    return in_poly

def write_model_bound_shp(xycorners,data=None,out_fname=None,field_dict=None,
                          col_name_order=None):
    
    poly_out = Polygon(xycorners)
    write_shp(polys=[poly_out],data=[data],out_fname=out_fname,field_dict=field_dict,
              col_name_order=col_name_order,write_prj_file=True)
    
            
def write_shp(polys=None,data=None,out_fname=None,
              field_dict=None,inproj='WGS84',write_prj_file=False,
              col_name_order=None):
    '''Write shapefile.
    '''
    w = shapefile.Writer()
    
    # Add new fields
    if (col_name_order is None):
        for f in field_dict:
            w.field(f, **field_dict[f])
    else:
        for f in col_name_order:
            w.field(f, **field_dict[f])

    for i,shp in enumerate(polys):
        if not hasattr(shp,'geom_type'):
            raise ValueError('Input features are not shapley geometries')
        elif shp.geom_type == 'Polygon':
            xtemp,ytemp = shp.exterior.xy
            main_poly = np.array(zip(xtemp,ytemp)).tolist()
            out_parts = []
            out_parts.append(main_poly)
            # check for interior holes
            if len(shp.interiors)>0:
                internal_polys = [np.array(inpoly.xy).T.tolist() for inpoly in shp.interiors]
                out_parts.extend(internal_polys)
            
            w.poly(parts = out_parts)
        elif shp.geom_type == 'MultiPolygon':
            all_parts = []
            for shp_temp in shp:
                out_parts = []
                xtemp,ytemp = shp_temp.exterior.xy
                out_parts.append(np.array(zip(xtemp,ytemp)).tolist())
                # check for interior holes
                if len(shp_temp.interiors)>0:
                    internal_polys = [np.array(inpoly.xy).T.tolist() for inpoly in shp_temp.interiors]
                    out_parts.extend(internal_polys)
                    
                all_parts.extend(out_parts)
            w.poly(parts = all_parts)
        else:
            raise ValueError('Currently only implemented for polygons')
    
    w.records.extend(data)    
    w.save(out_fname)
    
    
    if write_prj_file:
        prj_name = os.path.basename(out_fname).split('.')[0]
        dir_name = os.path.dirname(out_fname)

        write_prj(inproj,os.path.join(dir_name,prj_name))

def write_prj(inproj,fname):
    '''Write projection file for shapefile.
    '''
    shp_dir,shp_name =  os.path.split(fname)
    
    osr_proj = osr.SpatialReference()
    if isinstance(inproj,(float,int)):
        osr_proj.ImportFromEPSG(inproj)
    elif '+' in inproj:
        osr_proj.ImportFromProj4(inproj)
    elif 'PROJCS' in inproj or 'GEOGCS' in inproj:
        osr_proj.ImportFromWkt(inproj)
    elif hasattr(inproj,'ImportFromWkt'):
        osr_proj = inproj # already an osr sr
    else:
        # Assume outproj is geographic sr
        osr_proj.SetWellKnownGeogCS(inproj)
        
    osr_wkt = osr_proj.MorphToESRI().ExportToWkt()
    prj_file = open("{}.prj".format(os.path.join(shp_dir,shp_name.split('.')[0])), "w")
    prj_file.write(osr_wkt)
    prj_file.close()

def copy_prj(projshp=None,unprojshp=None):
    
    prjshp_dir,prjshp_name =  os.path.split(projshp)
    prjname,_ = os.path.splitext(prjshp_name)
    prjfname = os.path.join(prjshp_dir,'{}.prj'.format(prjname))    
    
    unprjshp_dir,unprjshp_name =  os.path.split(unprojshp)
    unprjname,_ = os.path.splitext(unprjshp_name)
    newprjfname = os.path.join(unprjshp_dir,'{}.prj'.format(unprjname)) 
    
    if os.path.isfile(prjfname):
        
        shutil.copy(prjfname,newprjfname)
    else:
        print("{} does not have a .prj file, choose another shapefile.".format(prjname))




#%% ------------ Shapefile conversion ---------------------
def shp_to_grid(shp,cell_spacing=None,pts_to_decimate=1,
                in_proj = None,out_proj=None,
                active_feature = None):
    '''Construct regular grid from polygon feature.
    
    Calculates smallest bounding rectangle for active_feature in shapefile fname.
    Converts shapefile to UTM prior to rotating and shifting. Output grid is
    in a Modflow-like coordinate system with the origin at row,col=0,0.
    
    
    '''
        
    if ~hasattr(shp,'geom_type'):
        # shp is not a shapely feature
        shp,npolys = shp_to_polygon(shp)
        if len(shp) > 1:
            if active_feature is None:
                areas = [Polygon(shape.points).area for shape in shp]
                feature_index = np.argmax(areas) # use feature with largest area
            else:
                feature_index = active_feature # use chosen feature
                
            shp = shp[feature_index]
        else:
            shp=shp[0]
            
    shp_pts = np.array(shp.exterior.xy).T
    if in_proj is not None and out_proj is not None:
        if in_proj != out_proj:
            # Reproject shp coordinates to out_proj
            proj_dict = {'xy_source':shp_pts,
                         'inproj':in_proj,'outproj':out_proj}
            shp_pts = projectXY(**proj_dict)
    elif in_proj is not None:
        out_proj = in_proj
        
    shp_pts_dec = shp_pts[::pts_to_decimate]
    
    # Calculate convex hull of feauture
    shp_dec = Polygon(list(zip(shp_pts_dec[:,0],shp_pts_dec[:,1])))
    shp_hull = shp_dec.convex_hull
    shp_hull_xy = np.array(shp_hull.exterior.coords.xy).T
    ncoords = shp_hull_xy.shape[0]
    
    rect_area = lambda bounds: (bounds[2]-bounds[0])*(bounds[3]-bounds[1])
    
    out_bounds = []
    angle_area_array = []
    rot_shps = []
    for icoord in range(ncoords-1):
        dx,dy = np.diff(shp_hull_xy[icoord:icoord+2,:],axis=0)[0]
        angle_theta = np.arctan2(dy,dx) # positive angle = counter-clockwise
        rot_shp = rotate(shp_hull,angle_theta,origin=(0.,0.),use_radians=True)
        out_bounds.append(rot_shp.bounds)
        angle_area_array.append([angle_theta,rect_area(rot_shp.bounds)])
        rot_shps.append(rot_shp)
        
    angle_area_array = np.array(angle_area_array)
    min_area_index = np.argmin(angle_area_array[:,1])
    min_angle = angle_area_array[min_area_index,0]    # radians
    
    # Constrain minimum angle in 1st and 4th quandrants to keep from flipping unnecessarily
    if min_angle < np.pi and min_angle > np.pi/2.:
        min_angle = np.pi+min_angle # 2nd quadrant to 4th
    elif min_angle >= np.pi and min_angle <= 3.*np.pi/2.:
        min_angle = min_angle-np.pi # 3rd to 1st
    
    shp_dec_rot = rotate(shp_dec,min_angle,origin=(0.,0.),use_radians=True)
    shp_dec_rot_xy = np.array(shp_dec_rot.exterior.coords.xy).T
    domain_extent = shp_dec_rot.bounds
    
    bounds1=np.array(domain_extent).reshape((2,2))
    xshift,yshift = bounds1[0,:] # set bottom left corner to 0,0
    rect_bounds = np.vstack((bounds1,np.hstack((bounds1[:,0],np.roll(bounds1[:,1],1,axis=0))).reshape((2,2)).T))
    rect_bounds = rect_bounds[np.argsort(rect_bounds[:,0]),:]

    # Apply x and y shifts to translate domain to origin    
    rect_bounds[:,0] = rect_bounds[:,0]-xshift
    rect_bounds[:,1] = rect_bounds[:,1]-yshift    
    shp_dec_rot_xy[:,0]=shp_dec_rot_xy[:,0]-xshift
    shp_dec_rot_xy[:,1]=shp_dec_rot_xy[:,1]-yshift
    shp_dec_rot_trans = Polygon(shp_dec_rot_xy.tolist())
    
    max_x,max_y = np.max(rect_bounds, axis=0)
    x_vect,y_vect = np.arange(-cell_spacing, max_x + 2.*cell_spacing, cell_spacing), np.arange(-cell_spacing, max_y + 2.*cell_spacing, cell_spacing)
    
    # Node locations
    X_nodes,Y_nodes = np.meshgrid(x_vect, y_vect[::-1]) # y decreases down increasing rows, x increases by columns to right
    # Return X,Y of nodes, rotated polygon, projection,rotation, and translation info
    gtransform = {'proj':out_proj,
                  'xyshift':[xshift,yshift],
                  'rotation':min_angle}
    return [X_nodes,Y_nodes],shp_dec_rot_trans,gtransform

def shp_to_patchcollection(in_polys=None,in_shp=None,radius=500.):
    '''Shapefile plotting.
    
    Source: modified from flopy.plot.plotutil.shapefile_to_patch_collection
    '''

    from matplotlib.patches import Polygon as mPolygon
    from matplotlib.patches import Circle as mCircle
    from matplotlib.patches import Path as mPath
    from matplotlib.patches import PathPatch as mPathPatch
    from matplotlib.collections import PatchCollection
    
    in_polygons = []
    # If shapefile path input:
    if (in_shp is not None):
        in_polygons,_ = shp_to_polygon(in_shp)
    
    if (in_polys is not None):
        in_polygons.extend(in_polys)
    
    blanks = []
    ptchs = []
    for poly in in_polys:
        st = poly.geom_type.lower()
        if st in ['point']:
            #points
            for p in poly.coords:
                ptchs.append(mCircle( (p[0], p[1]), radius=radius))
        elif st in ['linestring','linearring']:
            #line
            vertices = np.array(poly.coords)
            path = mPath(vertices)
            ptchs.append(mPathPatch(path, fill=False))
        elif st in ['polygon']:
            #polygons
            pts = np.array(poly.exterior.xy).T
            ptchs.append(mPolygon(pts))
            blanks.extend([mPolygon(np.array(poly1.xy).T) for poly1 in poly.interiors])
        elif st in ['multipolygon']:
            for ipoly in poly.geoms:
                ptchs.append(mPolygon(np.array(ipoly.exterior.xy).T))
                blanks.extend([mPolygon(np.array(ipoly1.xy).T) for ipoly1 in ipoly.interiors])
                
    pc = PatchCollection(ptchs)
    if len(blanks) > 0:
        bpc = PatchCollection(blanks)
    else:
        bpc = None
        
    return pc,bpc

def poly_bound_to_extent(in_polys):
    bound_list=[]
    for in_poly in in_polys:
        if in_poly.geom_type in ['Polygon']:
            bound_list.append(in_poly.bounds)
        else:
            for poly2 in in_poly:
                bound_list.append(poly2.bounds)

    bound_array = np.array(bound_list)
    # find maximum limits and rearrange to [xmin,xmax,ymin,ymax]
    max_bounds = np.array([bound_array[:,0].min(),bound_array[:,2].max(),
                           bound_array[:,1].min(),bound_array[:,3].max()])
    return max_bounds

#%% ------------ Shapefile transformations ----------------

def buffer_invalid(poly_in,buffer_size=0.):
    if not poly_in.is_valid:
        poly_in = poly_in.buffer(buffer_size)
    return poly_in

def sort_xy_cw(x=None,y=None):  
    '''Sort xy coordinates counterclockwise.
    
    Note: Only works when centroid is located within the polygon
            defined by the x,y coordinates.
            
    Source: http://stackoverflow.com/a/13935419
    '''
    
    # Find centroids
    centroidx = np.mean(x)
    centroidy = np.mean(y)
    
    # Calculate angles from centroid
    angles_from_centroid = np.arctan2(y-centroidy,x-centroidx)
    
    sort_order = np.argsort(angles_from_centroid)
    
    return x[sort_order],y[sort_order]

def proj_polys(polys=None,proj_in=None,proj_out=None,model_coords=False,proj_kwargs=None):
    '''Project list of polygons to new projection.
    
    Inputs
    -------------
    polys: list, array
        Iterable containing polygons to be projected
    
    proj_in: osr.SpatialReference or other projection information
        Starting projection of original polygon
        
    proj_out: osr.SpatialReference or other projection information
        Target projection of output polygon
        
    model_coords: bool
        sets whether or not to do a model transformation or just coordinate
        transformation
        if True, must supply proj_kwargs, which contains proj_in and proj_out
        if False, must supply proj_in and proj_out
    
    reverse_bool: bool
        sets order of transformations:
        if True, then rotation and shift before reprojection (i.e., model to utm)
        if False, rotation and shift after reprojection (i.e., utm to model)
    '''               

    # Load all xy pairs into one matrix, keep track of indexes
    new_shape_inds = []
    new_shape_internal_inds = []
    all_xy = []
    all_xy_internal=[]
    istart,iend = 0,0
    istart_internal,iend_internal = 0,0
    
    projected_polys = []
    for poly in polys:
        if hasattr(poly,'area'):
            if poly.geom_type in ['Polygon']:
                shp_xy = np.array(poly.exterior.xy).T
                all_xy.append(shp_xy)
                iend += int(shp_xy.shape[0])
                new_shape_inds.append([istart,iend])
                istart += int(shp_xy.shape[0])
                
                # For internal features separately
                if len(poly.interiors) > 0:
                    int_xy = []
                    int_inds = []
                    for poly1 in poly.interiors:
                        shp_int_xy = np.array(poly1.xy).T.reshape((-1,2))
                        int_xy.append(shp_int_xy)
                        iend_internal += int(shp_int_xy.shape[0])
                        int_inds.append([istart_internal,iend_internal])
                        istart_internal += int(shp_int_xy.shape[0])
                    
                    new_shape_internal_inds.append(int_inds)    
                    all_xy_internal.append(np.vstack(int_xy))
                else:
                    new_shape_internal_inds.append([[None,None]])    

                
            elif poly.geom_type in ['MultiPolygon']:
                    # Loop through polygon parts
                    multi_inds = []
                    poly_parts = []
                    multi_shape_internal_inds = []
                    multi_xy_internal = []
                    for ipoly in poly:
                        shp_xy_temp = np.array(ipoly.exterior.xy).T
                        poly_parts.append(shp_xy_temp)
                        iend += int(shp_xy_temp.shape[0])
                        multi_inds.append([istart,iend])
                        istart += int(shp_xy_temp.shape[0])
                        
                        # For internal features separately
                        if len(ipoly.interiors) > 0:
                            int_xy = []
                            int_inds = []
                            for poly1 in ipoly.interiors:
                                shp_int_xy = np.array(poly1.xy).T.reshape((-1,2))
                                int_xy.append(shp_int_xy)
                                iend_internal += int(shp_int_xy.shape[0])
                                int_inds.append([istart_internal,iend_internal])
                                istart_internal += int(shp_int_xy.shape[0])
                            
                            multi_shape_internal_inds.append(int_inds)    
                            multi_xy_internal.append(np.vstack(int_xy))
                        else:
                            multi_shape_internal_inds.append([[None,None]])
                    
                    new_shape_internal_inds.append(multi_shape_internal_inds)
                    if len(multi_xy_internal)>0:
                        all_xy_internal.append(np.vstack(multi_xy_internal))
                    
                    shp_xy = np.vstack(poly_parts)    
                    all_xy.append(shp_xy)
                    
                    new_shape_inds.append(multi_inds)
            
    # Project data once
    all_xy_array = np.vstack(all_xy)
    if model_coords:
        shp_xy_proj = np.array(modelcoord_transform(XY=[all_xy_array[:,0],all_xy_array[:,1]],**proj_kwargs)).T
    else:
        shp_xy_proj = np.array(osr_transform(XY=[all_xy_array[:,0],all_xy_array[:,1]],
                                                 proj_in=proj_in,
                                                 proj_out=proj_out)).T
        
    shp_xy_proj = shp_xy_proj.squeeze()
    
    if len(all_xy_internal) > 0:
        all_xy_internal_array = np.vstack(all_xy_internal)
        if model_coords:
            shp_xy_proj_internal = np.array(modelcoord_transform(XY=[all_xy_internal_array[:,0],all_xy_internal_array[:,1]],**proj_kwargs)).T
        else:
            shp_xy_proj_internal = np.array(osr_transform(XY=[all_xy_internal_array[:,0],all_xy_internal_array[:,1]],
                                                 proj_in=proj_in,
                                                 proj_out=proj_out)).T
    # Parse xy pairs back into polygons     
    for ipoly_startend,iinternal_se in zip(new_shape_inds,new_shape_internal_inds):
        if isinstance(ipoly_startend[0],int) or isinstance(ipoly_startend[0],float):
            # Only one shape to make
            istart,iend = ipoly_startend
            # remake internal features
            internal_polys = []
            for istart_int,iend_int in iinternal_se:
                if istart_int is not None:
                    internal_polys.append(Polygon(shp_xy_proj_internal[istart_int:iend_int,:]))
            internal_polys = cascaded_union(internal_polys)    
            if hasattr(internal_polys,'geom_type'):    
                projected_polys.append(Polygon(shp_xy_proj[istart:iend,:]).difference(internal_polys))
            else:
                projected_polys.append(Polygon(shp_xy_proj[istart:iend,:]))
        else:
            # Multiple shapes
            poly_parts = []
#                print ipoly_startend
            for (istart,iend),istartend_ints in zip(ipoly_startend,iinternal_se):
                # remake internal features
                internal_polys = []
                for istart_int,iend_int in istartend_ints:
                    if istart_int is not None:
                        internal_polys.append(Polygon(shp_xy_proj_internal[istart_int:iend_int,:]))
                internal_polys = cascaded_union(internal_polys)
                if hasattr(internal_polys,'geom_type'): 
                    poly_parts.append(Polygon(shp_xy_proj[istart:iend,:]).difference(internal_polys))
                else:
                    poly_parts.append(Polygon(shp_xy_proj[istart:iend,:]))
                
            projected_polys.append(MultiPolygon(poly_parts))
    
    return projected_polys 

#%% ------------ Shapefile operations -------------------
def pt_in_shp(in_polygon,XYpts,grid_buffer=None):
    speedups.enable()
    in_polygon,nshapes=shp_to_polygon(in_polygon)
      
    if hasattr(XYpts,'geom_type'):
        pts = XYpts
    elif isinstance(XYpts,np.ndarray):
        pts = MultiPoint(XYpts)
    else:
        x,y = XYpts
        pts = MultiPoint(list(zip(x.ravel(),y.ravel())))
    
    if grid_buffer is None:
        iout=[]
        for ipoly,loop_poly in enumerate(in_polygon):
            prepared_polygon = prep(loop_poly)
            iout.append([ishp for ishp,pt in enumerate(pts) if prepared_polygon.contains(pt)])
    else:
        try:
            new_polys = [shpbox(pt.xy[0]-grid_buffer,pt.xy[1]-grid_buffer,pt.xy[0]+grid_buffer,pt.xy[1]+grid_buffer) for pt in pts]
        except:
            new_polys = [shpbox(pt.xy[0][0]-grid_buffer,pt.xy[1][0]-grid_buffer,pt.xy[0][0]+grid_buffer,pt.xy[1][0]+grid_buffer) for pt in pts]             
        iout=[]
        for ipoly,loop_poly in enumerate(in_polygon):
            prepared_polygon = prep(loop_poly)
            iout.append([ishp for ishp,poly in enumerate(new_polys) if prepared_polygon.intersects(poly)])
    return iout

def gridpts_in_shp(in_polygon,XYpts, print_no_match = False, reduce_extent=True):
    '''
    Find row and column index for grid cell centers within in_polygon
    '''
    speedups.enable()
    in_polygon,nshapes=shp_to_polygon(in_polygon)
            
    all_y_cols,all_x_cols = [],[]
    collect_polys = []
    grid_centers_x,grid_centers_y = XYpts
    if len(in_polygon)>0:
        for ipoly,loop_poly in enumerate(in_polygon):
            poly_extent = loop_poly.bounds# minx,miny,maxx,maxy 
            if reduce_extent:
                pt_spacing = np.abs(np.diff(grid_centers_x,axis=1).mean()+1j*np.diff(grid_centers_y,axis=0).mean())/2. # mean diagonal/2 for cells
                gx,gy,inpts = grid_utils.reduce_extent(poly_extent,grid_centers_x,grid_centers_y,
                                          buffer_size = pt_spacing)
            else:
                gx,gy = grid_centers_x,grid_centers_y
                inpts = (grid_centers_x == grid_centers_x) & (grid_centers_y == grid_centers_y)
            
            inpts_rows_cols = np.array(inpts.nonzero()).T
            pts = MultiPoint(list(zip(gx,gy)))
            prepared_polygon = prep(loop_poly)
            iout=[ishp for ishp,pt in enumerate(pts) if prepared_polygon.contains(pt)]
            
            if len(iout)>0:
                # Convert back to coordinate grid indexes
                loop_y_rows,loop_x_cols = inpts_rows_cols[iout,:].T
                all_y_cols.extend(loop_y_rows)
                all_x_cols.extend(loop_x_cols)
                collect_polys.extend((np.float(ipoly)*np.ones(len(iout))).tolist())
        
        if len(all_x_cols)>0:
            if nshapes > 1:
                out_unq_ind_order = grid_utils.unique_rows(np.c_[all_y_cols,all_x_cols])
                out_rows = np.array(all_y_cols)[out_unq_ind_order]
                out_cols = np.array(all_x_cols)[out_unq_ind_order]
                out_poly = np.array(collect_polys)[out_unq_ind_order]
                return out_rows.ravel(),out_cols.ravel(),out_poly.ravel()
            else:
                return loop_y_rows.ravel(),loop_x_cols.ravel(),np.array(collect_polys).ravel()
        else:
            if print_no_match:
                print('No matches found, all points outside domain:')
                print('polygon extent: {}'.format(np.around(poly_extent,decimals=1)))
                print('points extent: {}'.format(np.around([grid_centers_x.min(),grid_centers_y.min(),grid_centers_x.max(),grid_centers_y.max()],decimals=1)))
            return [],[],[]
    else:
        return [],[],[]
    
def shp_to_pd(shp_fname,save_shps=False,shp_col = 'shape',
              save_polys=False,poly_col='poly'):
    '''Load shapefile to pandas dataframe.
    '''
    in_shp = shapefile.Reader(shp_fname)
    shp_df = pd.DataFrame(np.array(in_shp.records()),columns=np.array(in_shp.fields[1:])[:,0])
    shp_fieldtypes = np.array(in_shp.fields[1:])[:,1:]
    if save_shps:
        shp_df[shp_col] = in_shp.shapes()
        if save_polys:
            try: # assume all shapes are polygons
                out_polys = []
                for ishp in shp_df[shp_col].values.tolist():
                    if len(ishp.parts)==1:
                        out_polys.append(Polygon(ishp.points))
                    else:
                        # Multiple parts to the polygon, need to make a multipolygon
                        shp_temp = []
                        shp_pts = ishp.points
                        shp_parts = ishp.parts
                        for ipart in range(len(shp_parts)):
                            if ipart<len(shp_parts)-1:
                                shp_temp.append(Polygon(shp_pts[shp_parts[ipart]:shp_parts[ipart+1]]))
                            else:
                                shp_temp.append(Polygon(shp_pts[shp_parts[ipart]:])) # remaining points are part of the last part
                        
                        out_polys.append(MultiPolygon(shp_temp))
                    
                shp_df[poly_col] = out_polys
            except:
                poly_list = []
                for ishp in in_shp.shapes():
                    if ishp.shapeType == 1:
                        poly_list.append(MultiPoint(ishp.points))
                    elif ishp.shapeType == 2:
                        poly_list.append(LineString(ishp.points))
                    else:
                        poly_list.append(Polygon(ishp.points))
                shp_df[poly_col] = poly_list
                        
    for col_name,ftype in zip(shp_df.columns,shp_fieldtypes):
        if ftype[0]=='N':
            if int(ftype[-1]) != 0: # float column
                shp_df[col_name] = shp_df[col_name].astype(np.float)
            else: # integer column
                try:
                    shp_df[col_name] = shp_df[col_name].astype(np.int)
                except:
                    shp_df[col_name] = pd.to_numeric(shp_df[col_name])
    
    return shp_df

def nodes_to_cc(XYnodes,grid_transform):
    '''
    
    Returns:
        Variables:  [cc_x,cc_y], [cc_x_proj,cc_y_proj],     [cc_x_latlong, cc_y_latlong]
        Dimensions: [[ny-1,nx-1],[ny-1,nx-1]]      , [[ny-1,nx-1],[ny-1,nx-1]], [[ny-1,nx-1],[ny-1,nx-1]]
        where ny,nx = Xcorners.shape
        
    Dependencies:
        Packages: Numpy, Cartopy
        Functions: xrot, yrot
    '''
    Xcorners,Ycorners = XYnodes
    from_proj,xyshift,rot_angle = grid_transform
    # Calculate cell centers from nodes (grid corners)
    cc_x = Xcorners[:-1,:-1]+np.diff(Xcorners[:-1,:],axis=1)/2.
    cc_y = Ycorners[:-1,:-1]+np.diff(Ycorners[:,:-1],axis=0)/2.
    
    # Unrotate cell_centers to get cell centers in projected coordinate system
    cc_x_proj = xrot(cc_x+xyshift[0],cc_y+xyshift[1],-rot_angle)
    cc_y_proj = yrot(cc_x+xyshift[0],cc_y+xyshift[1],-rot_angle)
    
    # Unproject cell centers to geographic coordinate system (NAD83)
    if from_proj is not None:
        cc_x_latlong,cc_y_latlong = projectXY(np.array([cc_x_proj,cc_y_proj]),from_proj) # to NAD83
    else:
        cc_x_latlong, cc_y_latlong = [],[]
    return [cc_x,cc_y],[cc_x_proj,cc_y_proj],[cc_x_latlong, cc_y_latlong]