# -*- coding: utf-8 -*-
"""
mf_in_utils
Module to read and develop MODFLOW inputs. 

Created on Mon Dec 04 11:20:39 2017

@author: kbefus
"""
from __future__ import print_function
import os
import numpy as np
import flopy.modflow as mf
import flopy.mf6 as mf6
import flopy.utils as fu
from . import grid_utils
from . import shp_utils
from . import mf_out_utils



def load_model_data(load_data_dict=None,data_dir=None,
                    load_opts=None):
    '''Load model input files.
    
    Inputs
    ----------
    
    load_data_dict: dictionary
            Dictionary containing filepath (str), dict, or list of filepaths
            with the following keys:
                - 'elevation'
                - 'recharge'
                - 'k_data' (list/tuple of filenames; 
                list of lists if contains hk, vk, and thickness/ layer bottoms)
                - 'lake_data' (dictionary)
                    - 'bathymetry'
                    - 'storage'
                    - 'flux_data'
                       dictionary of {stress period:[nlakes x [prcplk,evaplk,rnf,wthdrw,ssmn,ssmx]],sp2:[[]]}
    '''
    
    # Check for nc data containing all data in dictionary
    if 'nc_data' in list(load_data_dict.keys()):
        data_dict = grid_utils.load_nc(load_data_dict['nc_data'])
    else:
        data_dict = {}
        for key in list(load_data_dict.keys()):
            if key in ['lake_data']:
                # Load lake data
                lake_dict = read_lake_data(load_data_dict[key],load_opts=load_opts)
                data_dict.update({key:lake_dict})
            
            elif key in ['river_data']:
                # Load lake data
                riv_dict = read_riv_data(load_data_dict[key],load_opts=load_opts)
                data_dict.update({key:riv_dict})
                                
            elif isinstance(load_data_dict[key],list):
                if key.lower() in ['k_data'] and len(load_data_dict[key])>0:
                    if len(load_data_dict[key])>2:
                        load_vk = True
                    else:
                        load_vk = False
                    if isinstance(load_data_dict[key][0],str):    
                        data_dict['hk'],data_dict['vk'],data_dict['kz']=read_hk(load_data_dict[key],load_vk=load_vk,**load_opts)
                    else:
                        if load_vk:
                            data_dict['hk'],data_dict['vk'],data_dict['kz']=load_data_dict[key]
                        else:
                            data_dict['hk'],data_dict['kz']=load_data_dict[key]
            else:
                if load_data_dict[key] is not None:
                    data_dict.update({key:grid_utils.load_and_griddata(load_data_dict[key],**load_opts)})
                else:
                    data_dict.update({key:None})
    
    return data_dict

def read_lake_data(lake_in_dict=None,load_opts=None,lake_num=0,bc_type='chd',
                   flux_data_columns=['STRESSPERIOD','STATUS', 'STAGE','RAINFALL',
                                      'EVAPORATION','RUNOFFON','WITHDRAWAL',
                                      'AUXILIARY']):
    
    lake_dict = {}
    if bc_type.lower() in ['LAK','lak']:
        if 'bathymetry' in list(lake_in_dict.keys()):
            # Load bathymetry data
            l_in=lake_in_dict['bathymetry']
            if isinstance(l_in,str):
                lake_dict['bathymetry'] = grid_utils.load_and_griddata(l_in,**load_opts)
            else:
                lake_dict['bathymetry'] = l_in # array or constant
        
        if 'storage' in list(lake_in_dict.keys()):
            # Load lake storage data
            l_in = lake_in_dict['storage']
            if isinstance(l_in,str):
                if '.ts' in l_in:
                    lake_dict['tables'] = [lake_num,l_in] # point to tab6_filename
                    lake_dict['make_table'] = False
                else:
                    lake_dict['tables'],header = grid_utils.read_txtgrid(l_in)
                    lake_dict['make_table'] = True
            else:
                lake_dict['tables'] = l_in # [stage,volume,sarea]#,barea]
                lake_dict['make_table'] = True # Need to use ModflowUtllaktab later
                
        if 'flux_data' in list(lake_in_dict.keys()):
            l_in = lake_in_dict['flux_data']
            if isinstance(l_in,str):
                # Load csv with form [[stress_period,status,stage,evap,runoff,withdrawal,aux]]
                # Note that the entries can be str indicating .ts files
                all_lines, header = grid_utils.read_txtgrid(l_in)
                out_lakeperioddata = {}
                for iline in all_lines:
                    new_iline=[]
                    # Clean up line
                    for ientry in iline:
                        if ientry.lower() in ['nan','none']:
                            ientry = None
                        else:
                            try:
                                ientry = float(ientry)
                            except:
                                pass # do nothing
                            
                        new_iline.append(ientry)
                    
                    sp_entry = list(zip(lake_num*np.ones(len(flux_data_columns),dtype=int),
                                        flux_data_columns,
                                        new_iline))
                    sp_entry = [stemp for stemp in sp_entry if stemp[-1] is not None \
                                and stemp[1] != flux_data_columns[0]]
                    out_lakeperioddata.update({int(iline[0]):sp_entry}) # insert for stress period
                    
                lake_dict['lakeperioddata'] = out_lakeperioddata
        
        if 'start_head' in list(lake_in_dict.keys()):
            strt = lake_in_dict['start_head']
            if isinstance(strt,(list,np.ndarray,tuple)):
                lake_dict.update({'start_head':strt})
            elif isinstance(strt,(float,int)):
                lake_dict.update({'start_head':[strt]})
                
        if 'bed_leakance' in list(lake_in_dict.keys()):
            bed_leakance = lake_in_dict['bed_leakance']
            if isinstance(strt,(list,np.ndarray,tuple)):
                lake_dict.update({'bed_leakance':bed_leakance})
            elif isinstance(strt,(float,int)):
                lake_dict.update({'bed_leakance':[bed_leakance]})
        
        if 'bathy_as_elev' in list(lake_in_dict.keys()):
            lake_dict.update({'bathy_as_elev':lake_in_dict['bathy_as_elev']})
    
    elif bc_type.lower() in ['chd']:
        # Constant/changing head boundary condition
        if 'timeseries' not in list(lake_in_dict.keys()):
            if 'start_head' in list(lake_in_dict.keys()):
                strt = lake_in_dict['start_head']
                if isinstance(strt,(list,np.ndarray,tuple)):
                    lake_dict.update({'start_head':strt})
                elif isinstance(strt,(float,int)):
                    lake_dict.update({'start_head':[strt]})
        else:
            ts_name='reservoir_ts'
            if 'time_series_namerecord' in list(lake_in_dict['timeseries']):
                ts_name = lake_in_dict['timeseries']['time_series_namerecord']
            lake_dict.update({'timeseries':lake_in_dict['timeseries'],
                              'timeseries_name': ts_name})
        
        
    return lake_dict


def read_riv_data(riv_in_dict=None,load_opts=None):
    
    riv_dict = {}
    if 'stage' in list(riv_in_dict.keys()):
        riv_dict.update({'stage':riv_in_dict['stage']})
    if 'cond' in list(riv_in_dict.keys()):
        riv_dict.update({'cond':riv_in_dict['cond']})
    if 'rbot' in list(riv_in_dict.keys()):
        riv_dict.update({'rbot':riv_in_dict['rbot']})
    
    return riv_dict
        
def read_hk(k_fnames, load_opts=None, load_last_botm=True,load_vk=False):
    '''Read hydraulic conductivity and bottom elevation rasters
    
    Parameters
    ----------
    
    k_fnames: list
        list of nlayer lists of [k_value_layer.tif,k_bottom_elev.tif]
    
    Returns
    --------
    
    hk_array: np.ndarray
        nlay x nrow x ncol array of hk values
            
    hk_botm_array: np.ndarray
        nlay x nrow x ncol array of hk layer bottom elevations
    '''
    hk_list = []
    botm_list = []
    vk_list = []
    for ilay,k_fname in enumerate(k_fnames):
        hk_list.append(grid_utils.load_and_griddata(k_fname[0],**load_opts))

        if load_vk and len(k_fname)==3:
            vk_list.append(grid_utils.load_and_griddata(k_fname[2],**load_opts))
        
        if ilay != len(k_fnames)-1 or load_last_botm:
            botm_list.append(grid_utils.load_and_griddata(k_fname[1],**load_opts))
    
    return np.array(hk_list),np.array(vk_list),np.array(botm_list)    

def make_model(data_dict=None, sim=None,tdis=None,gwf=None,ims=None,dis=None,ic=None,
               npf=None,sto=None,chd=None,well=None,evt=None,drn=None,
               ghb=None,obs=None,riv=None,lak=None,rch=None,oc=None):
    '''Make Modflow model from package dictionaries.
    
    Please see MODFLOW 6 and flopy documentation for more information on each 
    of these inputs. All are expected to either be dictionaries or None. Each 
    of the supporting functions show what key entries each package dictionary
    should have.
    '''
    
    idomain = dis['idomain']
    
    if sim is not None:
        sim_dict = make_sim(sim)
        sim_obj = mf6.MFSimulation(**sim_dict)    
    
    if tdis is not None:
        tdis_dict = make_tdis(tdis)
        tdis_obj = mf6.ModflowTdis(sim_obj,**tdis_dict)
    
    if gwf is not None:
        if 'model_name' not in list(gwf.keys()):
            gwf.update({'model_name':sim['model_name']})
            
        gwf_dict = make_gwf(gwf)
        if 'newtonoptions' not in list(gwf_dict.keys()):
            gwf_dict['newtonoptions'] = False
        
            
        mf_model = mf6.ModflowGwf(sim_obj,**gwf_dict)
        
    
    if ims is not None:
        ims_dict = make_ims(ims)
        ims_obj = mf6.ModflowIms(sim_obj,**ims_dict)
        sim_obj.register_ims_package(ims_obj,[mf_model.name])
        
    if dis is not None:
        if 'model_name' not in list(dis.keys()):
            dis.update({'model_name':sim['model_name']})
        dis_dict = make_dis(dis)
        dis_obj = mf6.ModflowGwfdis(mf_model,**dis_dict)
    
    if ic is not None:
        ic_dict = make_ic(ic)
        ic_obj = mf6.ModflowGwfic(mf_model,**ic_dict)
    
    if sto is not None:
        sto_dict = make_sto(sto)
        sto_obj = mf6.ModflowGwfsto(mf_model,**sto_dict)
    
    if npf is not None:
        npf_dict = make_npf(npf)
        npf_obj = mf6.ModflowGwfnpf(mf_model,**npf_dict)
    
    if chd is not None:
        if 'idomain' not in list(chd.keys()):
            chd.update({'idomain':idomain})
            
        chd_dict = make_chd(chd,elev=dis_dict['top'],nper=tdis_dict['nper'])

        if 'timeseries_name' in list(chd_dict.keys()):
            ts_name_temp = chd_dict.pop('timeseries_name')
            chd_obj = mf6.ModflowGwfchd(mf_model,**chd_dict)
            chd_obj.ts.time_series_namerecord = ts_name_temp
        else:
            chd_obj = mf6.ModflowGwfchd(mf_model,**chd_dict)
        
    
    if drn is not None:
        if not isinstance(drn,dict):
            # make drn from elevation and k data
            drn={'vk':npf_obj.k33.array,'top':dis['top'],
                 'botm':dis['botm'],'delr':dis['delr'],
                 'delc':dis['delc']}

        if 'idomain' not in list(drn.keys()):
            drn.update({'idomain':idomain})
            
        drn_dict = make_drn(drn)
        drn_obj = mf6.ModflowGwfdrn(mf_model,**drn_dict)
        
    if well is not None:
        well_dict = make_well(well)
        well_obj = mf6.ModflowGwfwel(mf_model,**well_dict)
    
    if evt is not None:
        if 'idomain' not in list(evt.keys()):
            evt.update({'idomain':idomain})
            
        evt_dict = make_evt(evt)
        evt_obj = mf6.ModflowGwfevt(mf_model,**evt_dict)
    
    if ghb is not None:
        if 'idomain' not in list(ghb.keys()):
            ghb.update({'idomain':idomain})
            
        ghb_dict = make_ghb(ghb)
        ghb_obj = mf6.ModflowGwfghb(mf_model,**ghb_dict)
    
    # Obs not yet implemented, load hds for observations.
#    if obs is not None:
#        obs_dict = make_obs(obs)
#        obs_obj = mf6.ModflowGwfobs(mf_model,**obs_dict)
        
    if riv is not None:
        if 'idomain' not in list(riv.keys()):
            riv.update({'idomain':idomain,'vk':npf_obj.k33.array,'top':dis['top']})
            
        riv_dict = make_riv(riv)
        riv_obj = mf6.ModflowGwfriv(mf_model,**riv_dict)
        
    if lak is not None:
        lak['lake_data'].update({'top':dis['top'],
                                 'botm':dis['botm'],'delr':dis['delr'],
                                 'delc':dis['delc'],
                                 'mf_model':mf_model})
        if 'model_name' not in list(lak.keys()):
            lak.update({'model_name':sim['model_name']})
        
        lak_tab_fname = '{}.lak.tab'.format(sim['model_name'])
        lak.update({'lak_tab_fname':lak_tab_fname})
        lak_dict = make_lak(lak)
        lak_obj = mf6.ModflowGwflak(mf_model,**lak_dict)
        
        if 'tables' in list(lak['lake_data'].keys()):
            if lak['lake_data']['make_table']:
                # make lake tables
                nrow,ncol = np.array(lak['lake_data']['tables']).shape
                lak_tab = [tuple(i) for i in lak['lake_data']['tables']]
                    
                lak_tables = mf6.ModflowUtllaktab(mf_model,table=lak_tab,
                                                  pname='lak_tab',filename=lak_tab_fname,
                                                  nrow=nrow,ncol=ncol,
                                                  parent_file=lak_obj)
            
    
    if rch is not None:
        if 'idomain' not in list(rch.keys()):
            rch.update({'idomain':idomain})
            
        rch_dict = make_rch(rch)
        rch_obj = mf6.ModflowGwfrch(mf_model,**rch_dict)
    
    if oc is not None:
        if 'model_name' not in list(oc.keys()):
            oc.update({'model_name':sim['model_name']})
        oc_dict = make_oc(oc)
        oc_obj = mf6.ModflowGwfoc(mf_model,**oc_dict)
    
    return sim_obj

# __________________ MF model components __________________________
def make_sim(sim_dict=None,version='mf6', exe_name='mf6'):
    out_sim_dict = {'sim_name':"sim_{}".format(sim_dict['model_name']),
                    'sim_ws':sim_dict['model_dir']}
    
    if 'version' not in list(sim_dict.keys()):
        out_sim_dict.update({'version':version})
    else:
        out_sim_dict.update({'version':sim_dict['version']})
        
    if 'exe_name' not in list(sim_dict.keys()):
        out_sim_dict.update({'exe_name':exe_name})
    else:
        out_sim_dict.update({'exe_name':sim_dict['exe_name']})
        
    return out_sim_dict
    
def make_tdis(tdis_dict=None,perlen=1,nstp=1,
              time_units='days',tsmult=1.,sim_tdis_file='simulation.tdis'):
    if tdis_dict['perioddata'] is None:
        perioddata = np.column_stack([perlen,nstp,tsmult])
    else:
        perioddata = tdis_dict['perioddata']

    if 'time_units' in list(tdis_dict.keys()):
        time_units = tdis_dict['time_units']
        
    nper = len(perioddata)
    out_tdis_dict = {'time_units':time_units,
                     'perioddata':perioddata,
                     'nper':nper,'pname':'tdis',
                     'filename':sim_tdis_file}
    return out_tdis_dict

def make_gwf(gwf_dict=None, ext='nam',newton=None):
    
    nam_file = ''.join(('{}.',ext)).format(gwf_dict['model_name'])
    
    if 'newton' in list(gwf_dict.keys()):
        newton=gwf_dict['newton']
    
    out_gwf_dict = {'modelname':gwf_dict['model_name'],
                    'model_nam_file':nam_file,
                    'newtonoptions':newton}
    return out_gwf_dict

def make_ims(ims_dict=None,under_relaxation='SIMPLE',outer_hclose=0.01,
             outer_maximum=50,print_option='SUMMARY',linear_acceleration='CG'):
    
    out_ims_dict = {'under_relaxation':under_relaxation,'outer_maximum':outer_maximum,
                    'outer_hclose':outer_hclose,'print_option':print_option,
                    'pname':'ims','linear_acceleration':linear_acceleration}
    
    if ims_dict is not None:
        out_ims_dict.update(ims_dict)
    
    return out_ims_dict

def make_dis(dis_dict=None,length_units='FEET',nlay=1,
             fname_fmt='{}.dis',fill_value=0.,
             zthick_elev_min=1.):
    
    if 'nlay' in list(dis_dict.keys()):
        nlay = dis_dict['nlay']
    
    if 'ncol' not in list(dis_dict.keys()) and 'nrow' not in list(dis_dict.keys()):
        nrow,ncol = dis_dict['top'].shape
    else:
        nrow = dis_dict['nrow']
        ncol = dis_dict['ncol']
    
    if 'length_units' in list(dis_dict.keys()):
        length_units = dis_dict['length_units'] 
    
    # Replace nan values for arrays
    dis_dict['botm'] = grid_utils.fill_nan(dis_dict['botm'],fill_value)
    dis_dict['top'] = grid_utils.fill_nan(dis_dict['top'],fill_value)
    
    # Make all thicknesses nonzero
    dis_dict['botm'] = grid_utils.adj_zbot(dis_dict['botm'],dis_dict['top'],zthick_elev_min)
    
    out_dis_dict = {'nlay':nlay, 'nrow':nrow, 'ncol':ncol,
                    'idomain':dis_dict['idomain'],
                    'top':dis_dict['top'],
                    'botm':dis_dict['botm'],
                    'delc':dis_dict['delc'],
                    'delr':dis_dict['delr'],
                    'length_units':length_units,
                    'filename':fname_fmt.format(dis_dict['model_name']),
                    'pname':'dis'}
    
    return out_dis_dict

def make_ic(ic_dict=None,strt=1.,fill_value=0.):
    
    if 'strt' in list(ic_dict.keys()):
        strt = ic_dict['strt']
        
    strt=grid_utils.fill_nan(strt,fill_value)
    
    out_ic_dict = {'strt':strt,'pname':'ic'}
    return out_ic_dict

def make_sto(sto_dict=None,iconvert=1,steady_state={0:True},transient={1:True},
             save_flows=True):
    
    out_sto_dict = {'iconvert':iconvert,'steady_state':steady_state,
                    'transient':transient,'save_flows':save_flows}
    
    if sto_dict is not None:
        out_sto_dict.update(sto_dict)
    
    return out_sto_dict

def make_npf(npf_dict=None,icelltype=1,wetdry=1,k22=None,k33=None,k_ani=1.,
             save_flows=True,fill_value=0.):

    npf_dict['k'] = grid_utils.fill_nan(npf_dict['k'],fill_value)
    
    if 'k_ani' in list(npf_dict.keys()):
        k_ani = npf_dict.pop('k_ani')
        
    if k22 is None:
        k22 = npf_dict['k']
    
    if k33 is None:
        k33 = npf_dict['k']/k_ani # default vertical k = horizontal k
    
    k22 = grid_utils.fill_nan(k22,fill_value)
    k33 = grid_utils.fill_nan(k33,fill_value)
    
    out_npf_dict = {'icelltype':icelltype,'wetdry':wetdry,
                    'k22':k22, 'k33':k33,
                    'pname':'npf','save_flows':save_flows}
    # Overwrite defaults
    out_npf_dict.update(npf_dict)
    
    return out_npf_dict

def make_drn(drn_dict=None,default_drn=[0.,10.],active_layer=0,mult_factor=100.):
    
    
    surface_conductance = (drn_dict['delr']*drn_dict['delc']*drn_dict['vk'][active_layer,:,:].squeeze())/((drn_dict['top']-drn_dict['botm'][0])/2.)
    surface_conductance[drn_dict['idomain']==0] = np.nan
    ir,ic = (~np.isnan(surface_conductance)).nonzero()
    ind_list = np.column_stack([np.zeros_like(ir),ir,ic])
    drn_rec = list(zip(ind_list,
               drn_dict['top'][~np.isnan(surface_conductance)],
               mult_factor*surface_conductance[~np.isnan(surface_conductance)]))
    
    
    out_drn_dict={'pname':'drn',
                  'stress_period_data':drn_rec,
                  'maxbound':len(drn_rec)}
    return out_drn_dict
    
def make_chd(chd_dict=None,elev=None,save_flows=True,default_head=1.,active_layers=[0],nper=None):
    '''Make constant head package information.
    '''
    
    chd_spd_dict = {'in_dict':chd_dict,'ptype':'chd', 'active_layers':active_layers,
                        'default_values':default_head,'nvals':1,
                        'idomain':chd_dict['idomain']}
    
    if 'lake_data' in list(chd_dict.keys()):
        lake_data = chd_dict['lake_data']
        mask = lake_data['reservoir_mask']
        
        # === Starting/constant head ====
        if 'start_head' not in list(lake_data.keys()):
            lake_data['start_head'] = [lake_data['timeseries']['timeseries'][0][1]]
            
        # Find cells within reservoir_mask that are inundated
        if len(lake_data['start_head'])>1:
            inds = np.array(((elev[None,:,:]<=lake_data['start_head']) & mask).nonzero()).T # start_head as array same size as mask
        else:
            inds = np.array(((elev[None,:,:]<=lake_data['start_head'][0]) & mask).nonzero()).T
        
        # Constant head only - no additional heads provided for later time steps OR starting head
        out_spd0 = list(zip(inds.astype(int).tolist(),len(inds[:,0])*lake_data['start_head'])) # should use this, testing line below
#        out_spd0 = list(zip(inds.astype(int).tolist(),len(inds[:,0])*[lake_data['timeseries_name']]*len(inds)))

        chd_rec = {0:out_spd0}
        bound_lens = [len(chd_rec[0])]
        # ==== Transient head =====
        if 'timeseries' in list(lake_data.keys()):
            # Use time series to set lake water level and lake extent
            
            # Find cells within reservoir_mask that are inundated
            for iper in np.arange(len(lake_data['timeseries']['timeseries'])):
                if iper+1 < nper:
                    inds = np.array(((elev[None,:,:]<=lake_data['timeseries']['timeseries'][iper][1]) & mask).nonzero()).T
                
                    out_spd = list(zip(inds.astype(int).tolist(),
                                       [lake_data['timeseries_name']]*len(inds),
                                       ))
                
                    if len(out_spd)>0:
                        chd_rec.update({iper+1:out_spd})
                    else:
                        chd_rec.update({iper+1:None})

                    bound_lens.append(len(out_spd))
        
    else:
        chd_rec = prep_spd(**chd_spd_dict)
    
    if 'save_flows' in list(chd_dict.keys()):
        save_flows = chd_dict['save_flows']
        
    out_chd_dict={'save_flows':save_flows,
                  'stress_period_data':chd_rec,
                  'maxbound':np.max(bound_lens),
                  'pname':'chd'}
    
    if lake_data is not None:
        if 'timeseries' in list(lake_data.keys()):
            out_chd_dict['timeseries']=lake_data['timeseries']
            out_chd_dict['timeseries_name']=lake_data['timeseries_name']

    return out_chd_dict

def make_well(well_dict=None,default_q=1.):
    wel_spd_dict = {'in_dict':well_dict,'ptype':'wel',
                    'default_values':default_q,'nvals':1,
                    'idomain':well_dict['idomain']}
    
    wel_rec = prep_spd(**wel_spd_dict)
    
    out_well_dict={'pname':'wel',
                   'stress_period_data':wel_rec}
    return out_well_dict

def make_evt(evt_dict=None,active_layers=[0],default_evt=[0,.1,1.,0.,1.,]):
    evt_spd_dict = {'in_dict':evt_dict,'ptype':'evt',
                    'default_values':default_evt,
                    'active_layers':active_layers,'nvals':5,
                    'idomain':evt_dict['idomain']}
    
    evt_rec = prep_spd(**evt_spd_dict)
    
    out_evt_dict={'pname':'evt',
                  'stress_period_data':evt_rec}
    return out_evt_dict

def make_ghb(ghb_dict=None,active_layers=[0],default_ghb=[0.,0.,1.]):
    ghb_spd_dict = {'in_dict':ghb_dict,'ptype':'ghb',
                    'default_values':default_ghb,
                    'active_layers':active_layers,'nvals':3,
                    'idomain':ghb_dict['idomain']}
    
    ghb_rec = prep_spd(**ghb_spd_dict)
    
    out_ghb_dict={'pname':'ghb',
                  'stress_period_data':ghb_rec,
                  'maxbound':len(ghb_rec)}
    return out_ghb_dict

def make_riv(riv_dict=None,active_layers=[0],default_riv=[0.5,.1,0]):
    
    mask = riv_dict['mask']
    inds = np.array(mask.nonzero()).T
    
    if 'stage' not in list(riv_dict.keys()):
        stage = riv_dict['top'][mask]
    else:
        if isinstance(riv_dict['stage'],str):
            stage = [riv_dict['stage']]*len(inds)
        elif isinstance(riv_dict['stage'],np.ndarray):
            stage = riv_dict['stage'][mask]
        else:
            stage = riv_dict['stage']*np.ones(len(inds))
    
    if 'cond' in list(riv_dict.keys()):
        if riv_dict['cond'] is not None:
            if isinstance(riv_dict['cond'],np.ndarray):
                cond = riv_dict['cond']
            elif isinstance(riv_dict['cond'],str):
                cond = [riv_dict['cond']]*len(stage) # for only 1 river
            else:
                cond = riv_dict['cond']*np.ones_like(stage)
        else:
            cond = riv_dict['vk'][0,mask]
    else:
        cond = riv_dict['vk'][0,mask]
    
    if 'rbot' not in list(riv_dict.keys()):
        rbot = riv_dict['top']-default_riv[0]
    else:
        if isinstance(riv_dict['rbot'],np.ndarray):
            rbot = riv_dict['rbot'][mask]
        else:
            rbot = riv_dict['rbot']*np.ones_like(stage)
            
    riv_xyv = list(zip(inds[:,0],inds[:,1],stage,cond,rbot))
    
    in_riv_dict = {'xyv':riv_xyv}
    
    riv_spd_dict = {'in_dict':in_riv_dict,'ptype':'riv',
                    'default_values':default_riv,
                    'active_layers':active_layers,'nvals':3,
                    'idomain':riv_dict['idomain']}
    
    riv_rec = prep_spd(**riv_spd_dict)
    
    
    out_riv_dict={'pname':'riv',
                  'stress_period_data':riv_rec,
                  'maxbound':len(riv_rec)}
    return out_riv_dict

def make_lak(lak_dict=None,lake_data=None,save_flows=True,
             lak_name_fmt='LAKE_{}',active_layers=[0],lake_num=0,
             bathy_as_elev=True,outlet_dict=None, stagefrec_fmt = '{}_lak_stage.lk',
             budgetfrec_fmt = '{}_lak_budget.lk',surfdep=None):
    '''Make Lake package.
    
    Inputs
    ----------
    
    
    
    outlet_dict: dictionary
        Dictionary containing key:value for outlets and outletperioddata
        
        outlets: list with each outlet specified as new list, e.g.,
            [[outletno, lakein, lakeout, couttype, invert, width, rough,slope]]
            
        outletperioddata: list of lists containing pairs of [outletno, outletsetting]
        
        see https://github.com/modflowpy/flopy/blob/develop/flopy/mf6/modflow/mfgwflak.py
        for more information
        
    
    Outputs
    ------------
    
    out_lak_dict: dictionary
        Dictionary with the inputs for the mfgwflak initialization
    
    
    
    '''
    
    out_lak_dict={'save_flows':save_flows,
                  'budget_filerecord':budgetfrec_fmt.format(lak_dict['model_name']),
                  'stage_filerecord':stagefrec_fmt.format(lak_dict['model_name']),
                  'surfdep':surfdep}
    
    if 'lake_data' in list(lak_dict.keys()):
        lake_data_temp = lak_dict['lake_data']
    
    if lake_data is None:
        lake_data = lake_data_temp
    else:
        # use user input over what is loaded
        lake_data_temp.update(lake_data)
        lake_data = lake_data_temp

    if 'bathy_as_elev' in list(lake_data.keys()):
        bathy_as_elev = lake_data['bathy_as_elev']
    
    # Make bathymetry an array if provided as a single number
    if isinstance(lake_data['bathymetry'],(float,int)):
        bathy = lake_data['bathymetry']*np.ones_like(lake_data['reservoir_mask'],dtype=float)
        bathy[~lake_data['reservoir_mask']]=np.nan
        if bathy_as_elev:
            lake_data['bathymetry'] = bathy
        else:
            # bathymetry provided as depth
            lake_data['bathymetry'] = lake_data['top']-bathy
    
    # Define connectiondata
    if 'connectiondata' not in list(lake_data.keys()) and \
        'reservoir_mask' in list(lake_data.keys()):
        res_mask = lake_data['reservoir_mask'].squeeze()
        bed_leakance = lake_data['bed_leakance']
        
        belev = lake_data['bathymetry'].squeeze()[res_mask]
        telev = lake_data['top'][res_mask]
        
        if len(res_mask.shape)==3:
            ilay,ir,ic = res_mask.nonzero()
        else:
            ir,ic = res_mask.nonzero()
        
        if isinstance(bed_leakance,np.ndarray) and isinstance(res_mask[0,0],bool):
            vbed_leakance = bed_leakance[res_mask]
        elif isinstance(bed_leakance,(float,int)):
            vbed_leakance = bed_leakance*np.ones_like(ir)
        else:
            vbed_leakance = bed_leakance[0]*np.ones_like(ir)
        
        
        # Identify vertical flow connections for all lake cells
        all_list = []
        nconnsv = len(ir)
        for ilay in active_layers:
            all_list.extend(zip(*[lake_num*np.ones_like(ir),np.arange(nconnsv),
                             np.column_stack([ilay*np.ones_like(ir),ir,ic]),
                            ['VERTICAL']*len(ir),vbed_leakance,
                            belev,telev,
                            np.zeros_like(ir),np.zeros_like(ir)]))

        
        # Identify horizontal connections
        res_edge_mask = grid_utils.raster_edge(res_mask.astype(int),search_val=1)
        belev = lake_data['bathymetry'].squeeze()[res_edge_mask]
        telev = lake_data['top'][res_edge_mask]
        connlen = lake_data['delc']
        connwidth = lake_data['delc']
        ir,ic = res_edge_mask.nonzero()
        
        if isinstance(bed_leakance,np.ndarray):
            hbed_leakance = bed_leakance[res_edge_mask]
        elif isinstance(bed_leakance,(float,int)):
            hbed_leakance = bed_leakance*np.ones_like(ir)
        else:
            hbed_leakance = bed_leakance[0]*np.ones_like(ir)
        
        nconnsh = len(ir)
        for ilay in active_layers:
            all_list.extend(zip(*[lake_num*np.ones_like(ir),nconnsv+np.arange(nconnsh),
                             np.column_stack([ilay*np.ones_like(ir),ir,ic]),
                            ['HORIZONTAL']*len(ir),hbed_leakance,
                            belev,telev,connlen*np.ones_like(ir),connwidth*np.ones_like(ir)]))
        
        
        
        lake_data.update({'connectiondata':list(all_list)})
    
    
    # Include lake storage data
    if 'tables' in list(lake_data.keys()):
        if lake_data['make_table']:
            table_info = [(lake_num,lak_dict['lak_tab_fname'])]
        else:
            table_info = [tuple(lake_data['tables'])]
        print(table_info)
        out_lak_dict.update({'tables':table_info,'ntables':len(table_info)})
            
    
    # Include lake period data
    if 'lakeperioddata' in list(lake_data.keys()):
        out_lak_dict.update({'lakeperioddata':lake_data['lakeperioddata']})
    
    # Include cell connection information
    if 'connectiondata' in list(lake_data.keys()):
        out_lak_dict.update({'connectiondata':lake_data['connectiondata']})
        # Use connection data to make packagedata
        packlist=[]
        unique_lakenums,inv1= np.unique([i[0] for i in lake_data['connectiondata']],return_inverse=True)
        nlakes = len(unique_lakenums)
#        packdata = mf6.ModflowGwflak.packagedata.empty(lake_data['mf_model'],maxbound=nlakes)
        for ilake,unique_lake in enumerate(unique_lakenums):
            nlakeconn = len((inv1==ilake).nonzero()[0])
            packlist.append((unique_lake,lake_data['start_head'][ilake],nlakeconn)) # ,lak_name_fmt.format(unique_lake)

        out_lak_dict.update({'nlakes':nlakes, 'packagedata':packlist})
    
    if outlet_dict is not None:
        out_lak_dict.update(outlet_dict)
    
    # Allow additional inputs using lak_options key/dict
    if 'lak_options' in list(lak_dict.keys()):
        lak_options = lak_dict.pop('lak_options')
        out_lak_dict.update(lak_options)
    
    return out_lak_dict

def make_rch(rch_dict=None,default_rch=1.,active_layers=[0]):
    rch_spd_dict = {'in_dict':rch_dict,'ptype':'rch',
                    'default_values':default_rch,
                    'active_layers':active_layers,'nvals':1,
                    'idomain':rch_dict['idomain']}
    
    rch_rec = prep_spd(**rch_spd_dict)
    out_rch_dict={'pname':'rch',
                  'stress_period_data':rch_rec}
    return out_rch_dict

def make_oc(oc_dict=None,budget_fmt='{}.cbc',head_fmt='{}.hds',
            saverecord=[('HEAD','ALL'),('BUDGET','ALL')],
            printrecord=None):
    
    model_name=oc_dict.pop('model_name')
    
    out_oc_dict={'pname':'oc',
                 'head_filerecord':[head_fmt.format(model_name)],
                 'budget_filerecord':[budget_fmt.format(model_name)],
                 'saverecord':saverecord,
                 'printrecord':printrecord}
    
    if oc_dict is not None:
        out_oc_dict.update(oc_dict)
        
    return out_oc_dict
# -----------------------------------------------------------------------------    

def prep_spd(in_dict=None,ptype=None, active_layers=[0],default_values=None,
             nvals=1,idomain=None):
    if default_values is not None:
        if not isinstance(default_values,(float,int)):
            nvals = len(default_values)
        
    if 'spd_rec' not in list(in_dict.keys()):
        if 'array' in list(in_dict.keys()):
            # Build indexes from non-nan portions of array
            if len(in_dict['array'].shape)==3:
                # 3D array, has layer information
                ind_list=[]
                val_list=[]
                for ilay in range(in_dict['array'].shape[0]):
                    temp_array = in_dict['array'][ilay]
                    ir,ic = (~np.isnan(temp_array)).nonzero()
                    ind_list.extend(np.column_stack([ilay*np.ones_like(ir),ir,ic]))
                    val_list.extend(temp_array[~np.isnan(temp_array)])
            elif len(in_dict['array'].shape)==2:
                # 2d array
                ind_list=[]
                val_list=[]
                ir,ic = (~np.isnan(in_dict['array'])).nonzero()
                non_nan_vals = in_dict['array'][~np.isnan(in_dict['array'])]
                for ilay in active_layers:
                    ind_list.extend(np.column_stack([ilay*np.ones_like(ir),ir,ic]))
                    val_list.extend(non_nan_vals)
            elif 'xy' in list(in_dict.keys()):
                # 1d array, assume each entry applies to one layer
                ind_list=[]
                val_list=[]
                nrow,ncol = in_dict['xy'][0].shape
                NR,NC = np.meshgrid(np.arange(ncol),np.arange(nrow))
                val_orig = in_dict['array']
                if len(val_orig)==1:
                    val_orig = val_orig*len(active_layers)
                    
                for ival,ilay in zip(val_orig,active_layers):
                    ind_list.extend(np.column_stack([ilay*np.ones_like(NR.ravel()),NR.ravel(),NC.ravel()]))
                    val_list.extend(ival*np.ones_like(NR.ravel()))
                    
            in_dict['inds'] = np.array(ind_list)
            vals = np.array(val_list)
        if 'inds' not in list(in_dict.keys()):
            if 'shp' in list(in_dict.keys()):
                # Make 
                in_dict['inds'] = shp_utils.gridpts_in_shp(in_dict['shp'],in_dict['xy'])
                if 'vals' in list(in_dict.keys()):
                    vals = in_dict['vals']
                else:
                    vals = np.array(default_values)*np.ones((in_dict['inds'].shape[0],np.array(default_values).shape[1]))
                    print("Warning: Using default_val = {} for all {} cells".format(default_values,ptype))
                
            elif 'xyv' in list(in_dict.keys()):
                # Use supplied xy coordinates
                if len(in_dict['xyv'][0])==2+nvals:
                    in_dict['inds'] = np.array(in_dict['xyv'])[:,:2].astype(int)
                    vals = np.array(in_dict['xyv'])[:,2:]
                else:
                    # Layer also given
                    in_dict['inds'] = np.array(in_dict['xyv'])[:,:3].astype(int)
                    vals = np.array(in_dict['xyv'])[:,3:]
                    
            elif 'csv' in list(in_dict.keys()):
                # Load csv file with x,y,vals
                in_data,_ = read_txtgrid(in_dict['csv'])
                if in_data.shape[1]==2+nvals:
                    in_dict['inds'] = np.array(in_data)[:,:2].astype(int)
                    vals = np.array(in_data)[:,2:]
                else:
                    # Layer also given
                    in_dict['inds'] = np.array(in_data)[:,:3].astype(int)
                    vals = np.array(in_data)[:,3:]
            else:
                raise Exception('{0}_shp, {0}_xyv, or {0}_csv must be provided to create {0} indexes'.format(ptype))
        
        # Use chd_inds to make chd_rec
        use_dotT = False
        if len(vals.shape)>1:
            if vals.shape[1]>1:
                use_dotT=True
        
        in_dict['inds'] = in_dict['inds'].astype(int) # force indexes to be integers
        
        spd_rec=[]
        if in_dict['inds'].shape[1]==3:
            # Layer given in inds
            if use_dotT:
                spd_rec.extend(list(zip(np.column_stack([in_dict['inds']]).astype(int).tolist(),*vals.T)))
            else:
                spd_rec.extend(list(zip(np.column_stack([in_dict['inds']]).astype(int).tolist(),vals)))
        else:
            new_inds = []
            for ilayer in active_layers:
                if use_dotT:
                    spd_rec.extend(list(zip(np.column_stack([ilayer*np.ones(len(in_dict['inds'])),
                                                                     in_dict['inds']]).astype(int).tolist(),
                                                                     *vals.T)))
                    new_inds.extend(list(zip(ilayer*np.ones(len(in_dict['inds'])),*in_dict['inds'].T)))
                else:
                    spd_rec.extend(list(zip(np.column_stack([ilayer*np.ones(len(in_dict['inds'])),
                                                                     in_dict['inds']]).astype(int).tolist(),
                                                                     vals)))
                    new_inds.extend(list(zip(ilayer*np.ones(len(in_dict['inds'])),in_dict['inds'])))
            in_dict['inds'] = np.array(new_inds,dtype=int)
    else:
        spd_rec = in_dict['spd_rec']
    
    if idomain is not None:
        # Remove cells that are outside the active domain
        inactive_lrc = np.array((idomain==0).nonzero()).T
        if inactive_lrc.shape[1]==2:
            new_inact_lrc = []
            for ilayer in active_layers:
                new_inact_lrc.extend(np.hstack([ilayer*np.ones([inactive_lrc.shape[0],1]),inactive_lrc]))
                
            new_inact_lrc = np.array(new_inact_lrc).astype(int)
        
        active_inds = grid_utils.remove_lrc(in_dict['inds'],new_inact_lrc)

        if not isinstance(spd_rec,np.recarray):           
            spd_rec = [spd_rec[i] for i in active_inds]
            
        else:
            
            spd_rec = spd_rec[active_inds]
    
    return spd_rec

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

def load_mf_model(fname=None,model_dir=None,
                  ref_fname='usgs.model.reference',
                  proj_in=None,xy_ul=None,
                  rotation=0.,load_only=None,version='mf2005'):
    '''Load MODFLOW model.
    
    Inputs
    ------------
    
    fname: str
         Namefile filename and optionally path (if model_dir==None)
        
    model_dir: str
         Full path to model namefile. Can be left as None if path provided to
         fname
         
    ref_fname: str
        Filename (and path if not in model_dir or model_dir==None) for USGS-
        style reference text file. Provides proj_in, xy_ul, and rotation.
    
    proj_in: str
        Projection of model provided as either EPSG code, well-known text, or 
        as an OSR projection object
    
    xy_ul: list
        List of the upper-left coordinates of the model in the original 
        projection: e.g., [2.4,6.8]
        
    rotation: float
        Float number giving rotation of model grid in radians
        
    load_only: list
        List of MODFLOW packages to load. None loads all packages available.
        
    version: str
        Version of MODFLOW used to run the original model. Changes loading 
        functions in flopy for mf2000, mf2005, and mf6.
        
    Outputs
    --------------
    mf_model: flopy.model_object
        Python object containing the flopy model structure.
    
    trans_dict: dictionary
        Python dictionary containing information for reprojecting model to a
        new coordinate system.
        
        
    '''
    if model_dir is not None:
        namefpath = os.path.join(model_dir,fname)
        ref_fname = os.path.join(model_dir,ref_fname)
        
    else:
        namefpath = fname
        
    namef = os.path.basename(namefpath)  
    load_dict = {'f':namef,'model_ws':model_dir,'version':version}
    mf_model = mf.Modflow.load(**load_dict)
    
    trans_dict=None
    if os.path.isfile(ref_fname):
        ref_dict = mf_out_utils.read_model_ref(ref_fname)
        trans_dict = mf_out_utils.make_grid_transform(ref_dict,from_ref=True)
        
    return mf_model, trans_dict

def calc_watertable(h_array):
    '''Extract head from first non-null layer of the model.
    '''
    nlay,nrow,ncol = h_array.shape
    newmat = np.nan*np.zeros((nrow,ncol))
    laymat = np.nan*np.zeros((nrow,ncol))
    itrue = True
    ilay_count = 0
    
    while itrue:
        cond1 = ((~np.isnan(h_array[ilay_count,:,:])) & (np.isnan(newmat)))
        newmat[cond1] = h_array[ilay_count,cond1]
        laymat[cond1] = ilay_count
        
        if len((np.isnan(newmat)==True).nonzero()[0]) > 1:
            # More nan values to fill
            ilay_count += 1
        else:
            itrue=False
        
        if ilay_count == nlay:
            # All layers checked
            itrue = False
    
    return newmat,laymat

def load_hds(model_name=None,workspace=None,inactive_head=np.nan,
             time2get=-1,min_head=-998.,info_array=None, calc_wt=False,
             ext='hds'):
    ''' Load groundwater head elevations from Modflow output.
    '''
    
    # Create the headfile object and load head
    headobj = fu.binaryfile.HeadFile(os.path.join(workspace,'{}.{}'.format(model_name,ext)))
    times = headobj.get_times()
    head = np.squeeze(headobj.get_data(totim=times[time2get]))
    headobj.close()
    
    if info_array is not None:  
    
        if len(head.shape)>2:
            head[:,info_array==grid_utils.grid_type_dict['inactive']] = inactive_head
        else:
            head[info_array==grid_utils.grid_type_dict['inactive']] = inactive_head
            
    head[head<min_head] = inactive_head
    
    if calc_wt:
        if len(head.shape)==3:
            head_out,wt_layer_mat = calc_watertable(head)
        else:
            head_out=head.copy()
            wt_layer_mat = np.zeros_like(head)
    
    else: 
        wt_layer_mat = np.nan
        head_out = head.copy()
        
    return head_out,wt_layer_mat
    
def load_cbc(model_name=None,workspace=None,extract_time=-1,
             entries_out = ['   CONSTANT HEAD','FLOW RIGHT FACE',
                            'FLOW FRONT FACE','FLOW LOWER FACE',
                            '          DRAINS',
                            ' HEAD DEP BOUNDS','        RECHARGE'],
                            ext='cbc',
                            budget_out=True):
    cbb = fu.CellBudgetFile(os.path.join(workspace,'{}.{}'.format(model_name,ext)))
    out_cbc_dict = {}
    budget_array = None   
    for entry in entries_out:
        mat_out = cbb.get_data(text=entry,full3D=True)[extract_time]
        mat_out = grid_utils.fill_mask(mat_out,fill_value=0.)
        out_cbc_dict.update({entry.strip():mat_out})
        if budget_out:
            if budget_array is None:
                budget_array = np.zeros_like(mat_out)

            budget_array += mat_out
            
    cbb.close()
    
    return out_cbc_dict, budget_array    
    