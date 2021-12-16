from pyproj import Transformer
import numpy as np
from scipy import spatial
import xarray as xr
from shapely.geometry import Point

class Query_Nearest_Points():
    
    """ 
    
    Description:
    
    A KD-tree implementation for fast point lookup on a 2D grid
    
    Keyword arguments: 
    dataset -- a xarray DataArray containing lat/lon coordinates
               (named 'lat' and 'lon' respectively)
               
    
    Implementation Reference:
        https://notes.stefanomattia.net/2017/12/12/The-quest-to-find-the-closest-ground-pixel/
        
        
               
               
    """
    
    def __init__(self, 
                 dataset, 
                 lat_coord_name='lat',
                 lon_coord_name='lon',
                 da_array_crs_epsg=4326, 
                 target_epsg=4978):
        
        
        self.transformer = Transformer.from_crs("epsg:{0}".format(da_array_crs_epsg), 
                                           "epsg:{0}".format(target_epsg))
        
        self.dataset = dataset
        # store original dataset shape
        self.shape = dataset.shape
        
        # reshape and stack coordinates
        coords = np.column_stack((dataset.coords[lat_coord_name].values.ravel(),
                                  dataset.coords[lon_coord_name].values.ravel()))
        
        # construct KD-tree
        self.tree = spatial.cKDTree(self.transform_coordinates(coords))

    
    def transform_coordinates(self, coords):
        """ Transform coordinates from geodetic to cartesian
        
        Keyword arguments:
        coords - a set of lan/lon coordinates (e.g. a tuple or 
                 an array of tuples)
        """


        if isinstance(coords, tuple) and len(coords)==2:
            xy = np.stack([coords])
        
        
        elif isinstance(coords, Point):
            xy = [coords.y, coords.x]
        
        else:
            xy = np.stack(coords)
        
        xy = np.asanyarray(xy)
        if xy.shape[0] == 2 and len(xy.shape) == 1:
            xy = xy.reshape(-1, 1).T
        

        new_coord = np.column_stack(self.transformer.transform(xy[:,0], xy[:,1]))
        
        
        
        return new_coord
        

        
    def query(self, point, k=1):
        """ Query the kd-tree for nearest neighbour.

        Keyword arguments:
        point -- a (lat, lon) tuple or array of tuples
        """
        _, index = self.tree.query(self.transform_coordinates(point), k=k)
        
        print(index)
        
        # regrid to 2D grid
        index = np.unravel_index(np.array(index), self.shape)
        index = np.asanyarray(index)
        
        if len(index.shape)> 2:
            if index.shape[1] == 1:
                index = index.squeeze(1)
            else:
                pass
                
        else:
            pass
        xx = index[0].ravel()
        yy = index[1].ravel()
        
        # return DataArray indexers
        return xr.DataArray(xx, dims='pixel'), \
               xr.DataArray(yy, dims='pixel')
    
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