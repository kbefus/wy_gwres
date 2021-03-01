# -*- coding: utf-8 -*-
"""
mf_out_utils
Module to read and develop MODFLOW outputs. 

Created on Mon Dec 04 11:20:39 2017

@author: kbefus
"""
from __future__ import print_function
import os
import numpy as np
#import flopy.utils as fu
import matplotlib.pyplot as plt

from . import grid_utils
from . import shp_utils
#from . import proj_utils

osr = grid_utils.osr
gdal = grid_utils.gdal
ogr = grid_utils.ogr

length_unit_dict = {0:'undefined',1:'foot',2:'meter',3:'centimeter'}
time_unit_dict = {0:'undefined',1:'second',2:'minute',3:'hour',4:'day',5:'year'}


# --------- Model spatial outputs ---------------


def save_model_bounds(model_obj=None,xycorners=None,XY_cc=None,save_active_poly=True,inproj=None,save_bounds=True,
                          proj_kwargs=None):
        ''' Save model bounds to shapefiles.
        
        XY_cc: list of np.ndarray's
                XY_cc should be the grid x,y coordinates for the model coordinate system
                for best results (i.e., m_domain.cc)
        '''
        model_name=model_obj.info_obj.model_name
        out_fname = os.path.join(os.path.dirname(os.path.dirname(model_obj.output_path)),
                                 'georef',
                                 '{}.shp'.format(model_name))

        col_name_order = ['CAF','delrc_m']
        data = [int(model_name.split('_')[1]),int(model_name.split('_')[-1][:-1])]
        field_dict = shp_utils.df_field_dict(None,col_names=col_name_order,col_types=['int','int'])
        shp_dict = {'out_fname':out_fname,'field_dict':field_dict,
                    'col_name_order':col_name_order,'data':data}
         
        if save_bounds:           
            shp_utils.write_model_bound_shp(xycorners,**shp_dict)
        
        # Save outline of active model cells to shapefile        
        if save_active_poly:
            active_out_fname = os.path.join(os.path.dirname(os.path.dirname(model_obj.output_path)),
                                 'active_bounds',
                                 '{}.shp'.format(model_name))
                                 
            Zarray = model_obj.dict_obj.cell_types.copy()
            Zarray[Zarray==0.] = np.nan
            
            active_shp,zval,ncells = shp_utils.raster_to_polygon(XY=XY_cc,Z=Zarray,
                                                           cell_spacing=model_obj.dict_obj.dis_obj.cell_spacing,
                                                           unq_Z_vals=False)
            proj_dict= {'polys':active_shp,'proj_kwargs':proj_kwargs}
            active_shp = shp_utils.proj_polys(**proj_dict)
            col_name_order.append('Ncells')
            data.append(ncells)
            field_dict = shp_utils.df_field_dict(None,col_names=col_name_order,col_types=['int','int','int'])
            active_shp_dict = {'polys':active_shp,'data':[data],'out_fname':active_out_fname,
                               'field_dict':field_dict,'col_name_order':col_name_order,
                               'write_prj_file':True,'inproj':inproj,'write_prj_file':True}            
            shp_utils.write_shp(**active_shp_dict)
            

def write_usgs_ref(sim=None):
    '''Export usgs.model.reference file.'''

    sr = sim.dis.sr
    
    fname = os.path.join(sim.model_ws,r'usgs.model.reference')
    with open(fname,'w') as fout:
        # Header info
        fout.write('# Model reference data for model {}\n'.format(sim.gwf.model_name))
        fout.write('xul {0:.5f}\n'.format(sr.xul))
        fout.write('yul {0:.5f}\n'.format(sr.yul))
        fout.write('rotation {}\n'.format(sr.rotation))
        
        # Write length units
        fout.write('length_units {}\n'.format(length_unit_dict[int(sr.length_units)]))
        
        # Write time units
        fout.write('time_units {}\n'.format(sim.tdis.time_units)) 
        
        # Assumes format is "Date Time", eg "01/01/18 12:00:00"
        fout.write('start_date {}\n'.format(sim.tdis.start_date_time.split()[0]))
        fout.write('start_time {}\n'.format(sim.tdis.start_date_time.split()[1]))
        
        # Model version        
        fout.write('model {}\n'.format(sim.version))           
        
        if isinstance(sim.dis.sr,'espg'):
            fout.write('# epsg code')
            fout.write('espg {}\n'.format(sim.dis.sr.espg))
            
        elif isinstance(sim.dis.sr,'proj4'):
            fout.write('# proj4 string')
            fout.write('proj4 {}\n'.format(sim.dis.sr.proj4))
        
        elif isinstance(sim.dis.sr,'prj'):
            fout.write('# wkt')
            fout.write('wkt {}\n'.format(sim.dis.sr.proj4))
        
        fout.close()


def read_model_ref(fname,comment_symbols=['#']):
    '''Import usgs.model.reference file to dictionary.
    
    Note: Now included with flopy through 
    flopy.utils.SpatialReference.read_usgs_model_reference_file.
    '''
    inum=0
    with open(fname,'r') as f_in:
        out_data={}
        for iline in f_in:
            if iline == '\n':
                continue
            if iline.strip()[0] in comment_symbols:
                pieces = ['comment_{}'.format(inum),iline[1:].strip('\n')]
                inum+=1
                out_data.update([pieces])
                continue
            
            iline = iline.strip('\n')
            if iline[0] in ['+']:
                pieces = ['proj4',iline] # add proj4 if not specified
            else:
                pieces = iline.split(' ')
                
            if len(pieces)>2:
                pieces = [pieces[0],' '.join(pieces[1:])]
            try:
                pieces[1] = int(pieces[1])
            except:
                try:
                    pieces[1] = float(pieces[1])
                except:
                    pass
        
            out_data.update([pieces])
    return out_data

def make_grid_transform(in_dict,s_hemi=False,from_ref=False):
    if from_ref:
        if 'proj4' in in_dict.keys():
            # Convert from proj4 string to dict entries
            proj4_entries = in_dict['proj4'].split('+')
            proj4_entries_clean = [entry.strip().split('=') for entry in proj4_entries if len(entry)>0]
            proj4_entries_cull = [entry for entry in proj4_entries_clean if len(entry)>1]
            in_dict.update(proj4_entries_cull)
            in_dict['x_shift'],in_dict['y_shift'] = in_dict['xul'],in_dict['yul']
            in_dict['rot_radians'] = np.deg2rad(in_dict['rotation']) # new flopy is +angle is counterclockwise
    
    # Make osr coordinate system from proj4 string
    proj_out = osr.SpatialReference()
    if 'proj4' in in_dict.keys():
        proj_out.ImportFromProj4(in_dict['proj4'])
    elif 'wkt' in in_dict.keys():
        proj_out.ImportFromWkt(in_dict['wkt'])        
        
    grid_transform = in_dict
    grid_transform.update({'proj_osr':proj_out})   
    return grid_transform
    
def save_head_geotiff(model_obj=None,save_ref_kwargs=None,XY_proj=None):
    load_hds_dict = {'model_name':model_obj.info_obj.model_name,
                     'workspace':model_obj.output_path,'calc_wt':True}
    
    input_kwargs = {'ref_dict':save_ref_kwargs['model_info_dict'],
                    'XY':XY_proj, 'load_hds_dict':load_hds_dict,
                    'dxdy':[model_obj.mf_model.dis.delc[0],model_obj.mf_model.dis.delr[0]],
                    'active_cells':model_obj.dict_obj.cell_types}
    grid_utils.save_head_geotiff(**input_kwargs)

def raster_to_polygon_gdal(XY=None,Z=None,in_proj=None,out_shp=None,gt=None,gdal_dfmt=gdal.GDT_Int32,
                           nan_val=-9999,unq_Z_vals=True,layername="polygonized",
                           field_name='ID',field_fmt=ogr.OFTInteger):
    
    if XY[0].shape != Z.shape:
        Z = Z.T.copy() # Try transposing
    
    nrows,ncols = Z.shape
    
    # make gdal raster in memory from np array
    dst_ds = gdal.GetDriverByName('MEM').Create('', ncols, nrows, 1 ,gdal_dfmt)
    dst_ds.SetGeoTransform(gt)
    dst_rb = dst_ds.GetRasterBand(1)
    dst_rb.Fill(nan_val) #initialise raster with nans
    dst_rb.SetNoDataValue(nan_val)
    
    if unq_Z_vals:
        # Create polygon for each unique value
        make_shp_array=Z.copy()
    else:
        make_shp_array=np.zeros_like(Z,dtype=int)
        make_shp_array[~np.isnan(Z) & Z!=0] = 1
        
    dst_rb.WriteArray(make_shp_array)
    dst_ds.FlushCache()
    
    # Coordinate system management
    srs = osr.SpatialReference()
    srs.ImportFromWkt(in_proj)
    
    shp_driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(out_shp):
        shp_driver.DeleteDataSource(out_shp)
        
    shp_src = shp_driver.CreateDataSource(out_shp)
    shp_layer = shp_src.CreateLayer(layername,srs=srs)
    
    new_field = ogr.FieldDefn(field_name,field_fmt)
    shp_layer.CreateField(new_field)
    
    _ = gdal.Polygonize(dst_rb,None,shp_layer,0,callback=None)
    
    shp_src.Destroy()
    dst_ds=None
    
def unpack_spd(spd_data=None,nlay_nrow_ncol=None):
    '''Convert stress period data to array.'''
    
    if hasattr(spd_data,'get_data'):
        # Convert spd_dict
        spd_data = spd_data.get_data()
    
    spd_cell_locs = np.array(spd_data['cellid'].tolist())
    
    if nlay_nrow_ncol is None:
        # Assume spd_dict contains value for every cell
        max_lay = np.max(spd_cell_locs[:,0]+1)
        max_row = np.max(spd_cell_locs[:,1]+1)
        max_col = np.max(spd_cell_locs[:,2]+1)
    else:
        max_lay,max_row,max_col = nlay_nrow_ncol
    
    data_fields = spd_data.dtype.fields.keys()
    data_fields.remove('cellid')
    
    out_dict = {}
    for ifield in data_fields:
        new_array = np.nan*np.zeros([max_lay,max_row,max_col])
        val_array = spd_data[ifield]
        new_array[spd_cell_locs[:,0],spd_cell_locs[:,1],spd_cell_locs[:,2]] = val_array
        out_dict.update({ifield:new_array.squeeze()})
    
    return out_dict
        
    
def quick_plot(mat,ncols=3,**kwargs):
    
    if len(mat.shape)==3 and mat.shape[0]>1:
        nplots = mat.shape[0]
        if nplots <=3:
            nrows,ncols = 1,nplots
        else:
            nrows = np.int(np.ceil(nplots/float(ncols))) 
        fig,ax  = plt.subplots(nrows,ncols)
        ax = ax.ravel()
        for ilay in np.arange(mat.shape[0]):
            im1=ax[ilay].imshow(np.ma.masked_invalid(mat[ilay,:,:]),
                                        interpolation='none',**kwargs)
            plt.colorbar(im1,ax=ax[ilay],orientation='horizontal')
            ax[ilay].set_title('Layer {}'.format(ilay))
    else:
        fig,ax  = plt.subplots()
        im1=ax.imshow(np.ma.masked_invalid(np.squeeze(mat)),interpolation='none',
                      **kwargs)
        plt.colorbar(im1,ax=ax,orientation='horizontal')

    plt.show()
    
    return ax
        
    
    
    
    
    