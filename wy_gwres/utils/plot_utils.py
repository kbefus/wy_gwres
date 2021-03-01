# -*- coding: utf-8 -*-
"""
Created on Fri Dec 08 11:52:22 2017

@author: kbefus

"""


import numpy as np
import matplotlib.pyplot as plt

from .shp_utils import shp_to_patchcollection,poly_bound_to_extent


# ------------- General plotting ---------------------
def quick_plot(mat,ncols=3,ax=None,**kwargs):
    
    if len(mat.shape)==3 and mat.shape[0]>1:
        nplots = mat.shape[0]
        if nplots <=3:
            nrows,ncols = 1,nplots
        else:
            nrows = np.int(np.ceil(nplots/float(ncols))) 
        
        if ax is None:
            fig,ax  = plt.subplots(nrows,ncols)
            ax = ax.ravel()
        for ilay in np.arange(mat.shape[0]):
            im1=ax[ilay].imshow(np.ma.masked_invalid(mat[ilay,:,:]),
                                        interpolation='none',**kwargs)
            plt.colorbar(im1,ax=ax[ilay],orientation='horizontal')
            ax[ilay].set_title('Layer {}'.format(ilay))
    else:
        if ax is None:
            fig,ax  = plt.subplots()
        im1=ax.imshow(np.ma.masked_invalid(np.squeeze(mat)),interpolation='none',
                      **kwargs)
        plt.colorbar(im1,ax=ax,orientation='horizontal')

    plt.show()
    
    return ax

def plot_shp(in_polys=None,in_shp=None, ax=None,
             extent=None,radius=500., cmap='Dark2',
             edgecolor='scaled', facecolor='scaled',
             a=None, masked_values=None,
             **kwargs):
    """
    Generic function for plotting a shapefile.
    
    
    Parameters
    ----------
    shp : string
        Name of the shapefile to plot.
    radius : float
        Radius of circle for points.  (Default is 500.)
    linewidth : float
        Width of all lines. (default is 1)
    cmap : string
        Name of colormap to use for polygon shading (default is 'Dark2')
    edgecolor : string
        Color name.  (Default is 'scaled' to scale the edge colors.)
    facecolor : string
        Color name.  (Default is 'scaled' to scale the face colors.)
    a : numpy.ndarray
        Array to plot.
    masked_values : iterable of floats, ints
        Values to mask.
    kwargs : dictionary
        Keyword arguments that are passed to PatchCollection.set(``**kwargs``).
        Some common kwargs would be 'linewidths', 'linestyles', 'alpha', etc.

    Returns
    -------
    pc : matplotlib.collections.PatchCollection

    Examples
    --------
    
    Source: modified from flopy.plot.plotutil.plot_shapefile
    """
    import matplotlib.pyplot as plt
    
    if 'vmin' in kwargs:
        vmin = kwargs.pop('vmin')
    else:
        vmin = None

    if 'vmax' in kwargs:
        vmax = kwargs.pop('vmax')
    else:
        vmax = None

    if ax is None:
        ax = plt.gca()
    cm = plt.get_cmap(cmap)
    pc,bpc = shp_to_patchcollection(in_polys=in_polys,in_shp=in_shp,radius=radius)
    pc.set(**kwargs)
    if a is None:
        nshp = len(pc.get_paths())
        cccol = cm(1. * np.arange(nshp) / nshp)
        if facecolor == 'scaled':
            pc.set_facecolor(cccol)
        else:
            pc.set_facecolor(facecolor)
        if edgecolor == 'scaled':
            pc.set_edgecolor(cccol)
        else:
            pc.set_edgecolor(edgecolor)
    else:
        pc.set_cmap(cm)
        if masked_values is not None:
            for mval in masked_values:
                a = np.ma.masked_equal(a, mval)
        if edgecolor == 'scaled':
            pc.set_edgecolor('none')
        else:
            pc.set_edgecolor(edgecolor)
        pc.set_array(a)
        pc.set_clim(vmin=vmin, vmax=vmax)
    # add the patch collection to the axis
    ax.add_collection(pc)
    
    # overlap polygons with white/blank polygons of interior holes
    if bpc is not None:
        bpc.set_edgecolor('none')
        bpc.set_facecolor('w')
        ax.add_collection(bpc)
    
    if (extent is not None):
        ax.axis(extent)
    else:
        ax.axis(poly_bound_to_extent(in_polys))
    plt.show()
    return ax,pc