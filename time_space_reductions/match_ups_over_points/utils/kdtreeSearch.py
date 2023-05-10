"""
Agorithm derived from:

https://notebook.community/lesserwhirls/unidata-python-workshop/netcdf-by-coordinates

"""

import numpy as np
import xarray as xr
from math import pi


from typing import Union

from scipy.spatial import cKDTree

from baseTree import BaseTree, Array2xN, Array1xN



class KdtreePolarCoordinateSearch(BaseTree):
    """ 
    
    Description:
    
    A KD-tree implementation for fast point lookup on a 2D grid
    
    This class diverges from the KdtreeWithReprojectionSearch in terms of the 
    pre-processing of the coordinates that is done prior to building the
    Kdtree.
    
    Here, the coordinates are converted to polar coordinates in x,y and z axis
    prior to creating the kdtree
    
    Keyword arguments: 
    dataset -- a xarray DataArray containing lat/lon coordinates
               (named 'lat' and 'lon' respectively)
               
               
    """
    
    
    def __init__(self, 
                 ncfile: Union[xr.DataArray,xr.Dataset], 
                 latCoordVarname: str, 
                 lonCoordVarname: str):
        """
        

        Parameters
        ----------
        ncfile : Union[xr.DataArray,xr.Dataset]
            DESCRIPTION.
        latvarname : str
            Coordinate name for the latitude within the netcdf.
        lonvarname : str
            Coordinate name for the longitude within the netcdf.


        """

        self.rad_factor = pi/180.0 
        self.ds = ds
        
        self.latvar = self.ds.coords[latCoordVarname]

        self.lonvar = self.ds.coords[lonCoordVarname]        
        
        self.shape = self.latvar.shape
        
        self.fixRangeOfCoordinates(ds, latCoordVarname, lonCoordVarname)
        
        polarCoordinates = self.toPolarCoordinates(self.latvar, self.lonvar)

        self.tree = cKDTree(polarCoordinates)
    
    
    
    def query(self, latitudes: Array1xN, longitudes: Array1xN) -> Array2xN:
        """

        This method uses the great arch equation for querying the nearest 
        coordinates within the netcdf.

        Parameters
        ----------
        latitudes : 1xN array of latitudes
            DESCRIPTION.
        longitudes : 1xN array of latitudes
            DESCRIPTION.

        Returns
        -------
        Array2xN
            DESCRIPTION.

        """

        lat0_rad = latitudes * self.rad_factor

        lon0_rad = longitudes * self.rad_factor

        clat0,clon0 = np.cos(lat0_rad), np.cos(lon0_rad)

        slat0,slon0 = np.sin(lat0_rad), np.sin(lon0_rad)

        dist_sq_min, minindex_1d = self.tree.query([clat0*clon0,clat0*slon0,slat0])

        iy_min, ix_min = np.unravel_index(minindex_1d, self.shape)

        return iy_min, ix_min
    
    def query_ball_point(self, point, radius):
        """ Query the kd-tree for all point within distance 
        radius of point(s) x
        
        Keyword arguments:
        point -- a (lat, lon) tuple or array of tuples
        radius -- the search radius (km)
        """
        
        index = self.tree.query_ball_point(self.transform_coordinates(point),
                                           radius)

        # regrid to 2D grid 
        index = np.unravel_index(index[0], self.shape)
        
        # return DataArray indexers
        return xr.DataArray(index[0], dims='pixel'), \
               xr.DataArray(index[1], dims='pixel')
    
    
        
        
if "__main__" == __name__:
    
    
    ds = xr.tutorial.open_dataset("rasm").load()

    searcher = KdtreePolarCoordinateSearch(ds,'yc','xc')
    
    iy,ix = searcher.query(65, 255.6)
    
    
    print('Closest lat lon:', iy,ix)
    
    nearest_points = searcher.pointsToXrArray([iy, ix], "y", "x")
    
    print(nearest_points)

