# -*- coding: utf-8 -*-
"""
mbase module
    This module contains the base model class to which all other 
    datasets and models are added.
"""

from __future__ import print_function
import os
import numpy as np
import time

# Load wy_gwres components
from .utils import proj_utils
from .utils import grid_utils
from .utils import shp_utils
from .utils import mf_in_utils


class Model(object):
    '''Model class for wy_gwres.
    
    The Model class in wy_gwres is an object that contains all of the routines 
    for modeling most groundwater-reservoir interactions. Individual parts of the Model
    class can be used to perform more complex modeling scenarios.
    
    '''
    def __init__(self,model_name=None, model_dir=None, data_dir=None,
                 cbc_unit=53,use_ll=True,domain_shp=None,cell_spacing=None,
                 maxXY=None,reservoir_shp=None,dam_shp=None,river_shp=None,nlay=1,
                 mf_dict=None,in_proj=None):
        '''Model class constructor.
        
        '''
        self.model_name = model_name
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.cbc_unit = cbc_unit
        self.use_ll = use_ll
        self.in_proj = in_proj
        
        self.cell_spacing = cell_spacing
        self.maxXY = maxXY
        self.domain_shp = domain_shp
        self.reservoir_shp = reservoir_shp
        self.river_shp = river_shp
        self.dam_shp = dam_shp
        
        self.nlay = nlay
        self.mf_dict = mf_dict
        self.mf_dict.update({'sim':{'model_name':self.model_name,'model_dir':self.model_dir},
                            'oc':{'model_name':self.model_name}})
        
        if self.maxXY is None and self.domain_shp is None:
            print('maxXY is required if no domain_shp is given.')
        
        # Make directories if not currently made
        for idir in [self.model_dir,self.data_dir]:
            if not os.path.isdir(idir):
                os.makedirs(idir)
                       
    def make_domain(self):
        '''Make model domain.
        '''
        if self.domain_shp is not None:
            # Create domain from shapefile
            self.domain_poly = shp_utils.shp_to_polygon(self.domain_shp)[0]
            if isinstance(self.domain_poly,list):
                self.domain_poly = self.domain_poly[0]
            
            if self.in_proj is None:
                # Default to projection of self.domain_shp
                prj_fpath = os.path.join('{}.prj'.format(os.path.splitext(self.domain_shp)[0]))
                sr = proj_utils.osr.SpatialReference()
                sr.ImportFromWkt(proj_utils.read_prj(prj_fpath))
                self.in_proj = sr
            
            self.XYnodes,self.model_polygon,self.grid_transform = shp_utils.shp_to_grid(self.domain_poly,
                                                                  cell_spacing=self.cell_spacing,in_proj=self.in_proj)
            
        else:
            # Assume all input data on a regular grid
            x = np.arange(0,self.maxXY[0]+self.cell_spacing,self.cell_spacing)
            y = np.arange(0,self.maxXY[1]+self.cell_spacing,self.cell_spacing)
            self.XYnodes = np.meshgrid(x,y)
            self.model_polygon = None # TO DO: make rectangle polygon from boundary
            self.grid_transform = {'proj':None,
                                   'xyshift':[0.,0.],
                                   'rotation':0.} # TO DO: make grid transform
                
        self.domain_extent =  [self.XYnodes[0].min(),self.XYnodes[1].min(),self.XYnodes[0].max(),self.XYnodes[1].max()]
        self.cc,self.cc_proj,self.cc_ll = proj_utils.nodes_to_cc(self.XYnodes,self.grid_transform)
        self.nrow,self.ncol = self.cc[0].shape
        dlat,dlong = self.cc_ll[1][0,0]-self.cc_ll[1][1,0],self.cc_ll[0][0,0]-self.cc_ll[0][0,1]
        self.domain_extent_ll =  [self.cc_ll[0].min()-dlong/2.,self.cc_ll[1].min()-dlat/2.,
                                  self.cc_ll[0].max()+dlong/2.,self.cc_ll[1].max()+dlat/2.]
        
        if hasattr(self.cc_proj,'data'):
            self.grid_transform.update({'xul':self.cc_proj.data[0][0,0],
                                        'yul':self.cc_proj.data[1][0,0]})
        else:
            self.grid_transform.update({'xul':self.cc_proj[0][0,0],
                                    'yul':self.cc_proj[1][0,0]})
        
        # Update mf_dict
        self.mf_dict.update({'dis':{'delr':self.cell_spacing,
                                    'delc':self.cell_spacing,
                                    'nlay':self.nlay}})
        
    def load_data(self, filename_dict=None, data_dir=None, load_opts=None,
                  top=0.,botm=-10,zthick_elev_min=None,sy=0.2,ss=1e-6):
        '''Load model data.
        
        Inputs
        -----------
        
        filename_dict: dictionary
            Dictionary containing filepath (str), dict, or list of filepaths
            with the following keys:
                - 'elevation'
                - 'recharge'
                - 'k_data' (list/tuple of filenames)
                - 'lake_data' (dictionary)
                    - 'bathymetry'
                    - 'storage'
                    - 'fluxes'
                - 'river_data'
        
        data_dir: str
             Path to directory with the input data files. Only needed if the 
             entries in filename_dict do not have full paths
             
        load_opts: dict
            Dictionary containing load options for grid_utils.load_and_griddata
            
        Outputs
        -----------
        
        None. All stored in mbase.Model object (i.e., self)
        
        '''
        
        if data_dir is None:
            data_dir = self.data_dir
        else:
            # Overwrite stored data_dir with new one
            self.data_dir = data_dir
        
        if load_opts is None:
            load_opts = {}
            
        if self.use_ll: # use long,lat cell centers
            load_opts.update({'new_xy':self.cc_ll})
        else: # Use projected coordinate system cell centers
            load_opts.update({'new_xy':self.cc_proj})
        
        load_dict = {'load_data_dict':filename_dict,'data_dir':self.data_dir,
                     'load_opts':load_opts}
        # Load relevant datasets
        self.data_dict = mf_in_utils.load_model_data(**load_dict)
        
        if 'elevation' in list(self.data_dict.keys()):
            self.mf_dict['dis'].update({'top':self.data_dict['elevation']})
        else:
            self.mf_dict['dis'].update({'top':top})
        
        self.grid_dis = [self.nlay,self.nrow,self.ncol]
        if 'kz' in list(self.data_dict.keys()):
            
            kz = grid_utils.make_zbot(self.mf_dict['dis']['top'],
                                      self.grid_dis,self.data_dict['kz'],
                                      zthick_elev_min)
            self.mf_dict['dis'].update({'botm':kz})
        else:
            kz = grid_utils.make_zbot(self.mf_dict['dis']['top'],
                                      self.grid_dis,botm,
                                      zthick_elev_min)
            self.mf_dict['dis'].update({'botm':botm})
            
        if 'hk' in list(self.data_dict.keys()):
            self.mf_dict.update({'npf':{'k':self.data_dict['hk'],
                                        }})
            if 'vk' in list(self.data_dict.keys()):
                self.mf_dict['npf'].update({'k33':self.data_dict['vk']})
        
        if 'recharge' in list(self.data_dict.keys()):
            self.mf_dict.update({'rch':{'array':self.data_dict['recharge']}})
            
        if 'lake_data' in list(self.data_dict.keys()):
            if filename_dict['lake_data']['bc_type']=='lak':
                self.mf_dict.update({'lak':{'lake_data':self.data_dict['lake_data']}})
            elif filename_dict['lake_data']['bc_type']=='chd':
                self.mf_dict.update({'chd':{'lake_data':self.data_dict['lake_data']}})
        
        if 'river_data' in list(self.data_dict.keys()):
            self.mf_dict.update({'riv':self.data_dict['river_data']})
        
        self.mf_dict.update({'sto':{'ss':ss,'sy':sy}})
    
    def assign_cell_types(self,rmv_isolated=True,min_area=1e3,lake_layers=[0],
                          river_buffer=None):
        '''Assign cell types from masks.'''
        
        # Find active cells
        geodata_in = {'X':self.cc_proj[0],'Y':self.cc_proj[1],'Vals':self.cc_proj[0]}
        self.geodata = grid_utils.make_geodata(**geodata_in)[1] # only save geodata
        self.active_cells = grid_utils.define_mask(self.domain_shp,[self.nrow,self.ncol,self.geodata])
        if rmv_isolated:
            self.active_cells = grid_utils.clean_ibound(self.active_cells,check_inactive=True,min_area=min_area)
       
        self.cell_types = np.zeros(self.cc[0].shape)
        self.cell_types[self.active_cells] = grid_utils.grid_type_dict['active']
        
        
        # Find lake/reservoir cells
        if self.reservoir_shp is not None:
            reservoir_mask_temp = grid_utils.define_mask(self.reservoir_shp,[self.nrow,self.ncol,self.geodata])
            self.cell_types[reservoir_mask_temp] = grid_utils.grid_type_dict['reservoir']
            self.reservoir_mask = np.zeros((self.nlay,self.nrow,self.ncol),dtype=bool)
            for ilay in lake_layers:
                self.reservoir_mask[ilay] = reservoir_mask_temp.copy()
            
            if 'lak' in list(self.mf_dict.keys()):
                if 'lake_data' in list(self.mf_dict['lak'].keys()):
                    self.mf_dict['lak']['lake_data'].update({'reservoir_mask':self.reservoir_mask})
                else:
                    self.mf_dict['lak'].update({'lake_data':{'reservoir_mask':self.reservoir_mask}})
            elif 'chd' in list(self.mf_dict.keys()):
                if 'lake_data' in list(self.mf_dict['chd'].keys()):
                    self.mf_dict['chd']['lake_data'].update({'reservoir_mask':self.reservoir_mask})
                else:
                    self.mf_dict['chd'].update({'lake_data':{'reservoir_mask':self.reservoir_mask}})
            
        if self.river_shp is not None:
            self.river_poly = shp_utils.shp_to_line(self.river_shp)[0]
            # Need to make river into a polygon by buffering by some amount
            if river_buffer is None:
                river_buffer = self.cell_spacing/2.
                
            self.river_cells = grid_utils.define_mask(self.river_shp,[self.nrow,self.ncol,self.geodata])
            self.cell_types[self.river_cells & ~self.reservoir_mask[0]] = grid_utils.grid_type_dict['river']
            
            if 'riv' in list(self.mf_dict.keys()):
                self.mf_dict['riv'].update({'mask':self.river_cells})
        
        if self.dam_shp is not None:
            dam_mask_temp = grid_utils.define_mask(self.dam_shp,[self.nrow,self.ncol,self.geodata])
            self.cell_types[dam_mask_temp] = grid_utils.grid_type_dict['inactive']
            self.dam_mask = np.tile(dam_mask_temp,(self.nlay,1,1))
        
            # Overwrite other masks
            if self.reservoir_shp is not None:
                self.reservoir_mask[self.dam_mask] = False
                if 'lak' in list(self.mf_dict.keys()):
                    if 'lake_data' in list(self.mf_dict['lak'].keys()):
                        self.mf_dict['lak']['lake_data']['reservoir_mask'][self.dam_mask] = False
                elif 'chd' in list(self.mf_dict.keys()):
                    if 'lake_data' in list(self.mf_dict['chd'].keys()):
                        self.mf_dict['chd']['lake_data']['reservoir_mask'][self.dam_mask] = False
            if 'riv' in list(self.mf_dict.keys()):
                self.mf_dict['riv']['mask'][self.dam_mask[0]] = False

    
    def make_mfpackages(self,write_bool=True,ic_head=None,silent=True):
        '''Make MODFLOW inputs and packages.'''
        
        # Use top of layer elevations as starting head if not specified
        if ic_head is not None:
            self.mf_dict.update({'ic':{'strt':ic_head}})
        else:
            zbot_temp = self.mf_dict['dis']['botm']
            zbot_temp = np.roll(zbot_temp,1,axis=0)
            zbot_temp[0] = 0.
            top_elevs = self.mf_dict['dis']['top']-np.cumsum(zbot_temp,axis=0)
            self.mf_dict.update({'ic':{'strt':top_elevs}})
        
        self.mf_dict['dis'].update({'idomain':self.active_cells.astype(int)})
        
        self.mf_sim = mf_in_utils.make_model(**self.mf_dict)
        if write_bool:
            self.mf_sim.write_simulation(silent=silent)
    
    def run_model(self,mf_exception=True,report=False,silent=True):
        '''Run MODFLOW model.'''
        
        print("Running model at {}".format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())))
        model_start = time.time()
        self.success, self.mfoutput = self.mf_sim.run_simulation(silent=silent,
                                                                 report=report)
        self.model_elapsed = time.time()-model_start
        if not self.success and mf_exception:
            raise Exception('MODFLOW did not terminate normally.')
            
    def process_model(self):
        '''Standard post-processing steps.'''
        pass
        
        