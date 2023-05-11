"""


Agorithm derived from:

https://notebook.community/lesserwhirls/unidata-python-workshop/netcdf-by-coordinates


"""



from typing import Union
import numpy as np
import xarray as xr
from math import pi
from sklearn.neighbors import BallTree
from sklearn import metrics

from baseTree import BaseTree, Array2xN, Array1xN, END


class BallTreeSearch(BaseTree):
    def __init__(self, 
                 ds:Union[xr.DataArray,xr.Dataset], 
                 latCoordVarname: str, 
                 lonCoordVarname: str
             ):
        
        self.rad_factor = pi/180.0 
        self.ds = ds
        
        self.latvar = self.ds.coords[latCoordVarname]

        self.lonvar = self.ds.coords[lonCoordVarname]        
        
        self.shape = self.latvar.shape
        
        self.fixRangeOfCoordinates(ds, latCoordVarname, lonCoordVarname)
        
        # polarCoordinates = self.toPolarCoordinates(self.latvar, self.lonvar)
        
        latvalsInRadians = self.latvar * self.rad_factor

        lonvalsInRadians = self.lonvar * self.rad_factor
        
        coords = np.array([latvalsInRadians.values.ravel(), 
                                       lonvalsInRadians.values.ravel()]).T

        self.tree = BallTree(coords,
                             metric="haversine")
        
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
        
        latitudes_asRadians = np.array(latitudes).ravel() * self.rad_factor 
        longitudes_asRadians = np.array(longitudes).ravel() * self.rad_factor 
        points = np.array([latitudes_asRadians, longitudes_asRadians]).T
        dist_sq_min, minindex_1d = self.tree.query(points)
        iy_min, ix_min = np.unravel_index(minindex_1d, self.shape)
        return iy_min,ix_min
    
    def query_ball_point(self, point, radius):
        """ Query the kd-tree for all point within distance 
        radius of point(s) x
        
        Keyword arguments:
        point -- a (lat, lon) tuple or an (lat,lon) 2xN array of coordinates
        radius -- the search radius (km)
        """
        
        point = np.array(point).reshape(-1, 1)
        
        
        latitudes_asRadians = point[0, :] * self.rad_factor
        longitudes_asRadians = point[1, :] * self.rad_factor
        
        index = self.tree.query_ball_point([latitudes_asRadians, longitudes_asRadians],
                                           radius)

        # regrid to 2D grid 
        index = np.unravel_index(index[0], self.shape)
        
        # return DataArray indexers
        return index
    

        
if "__main__" == __name__:

    ds = xr.tutorial.open_dataset("rasm").load()

    searcher = BallTreeSearch(ds,'yc','xc')
    
    iy, ix = searcher.query([65], [255.6])
 
    nearest_points = searcher.pointsToXrArray([iy, ix], "y", "x")
    
    print(nearest_points, end=END)
    
    print('Closest lat lon:', nearest_points.coords.items())
    

