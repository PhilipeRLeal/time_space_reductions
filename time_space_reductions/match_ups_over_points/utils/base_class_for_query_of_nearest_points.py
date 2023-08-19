from pyproj import Transformer as CRSTransformer
import numpy as np
from scipy import spatial
import xarray as xr
from shapely.geometry import Point
from typing import Tuple, List, Union
from abc import ABC, abstractmethod

class Transformer(ABC):
    
    @abstractmethod
    def transform(self, point: Point, point_origin_SRC_EPSG: str, target_SRC_EPSG: str):
        """
        
        Parameters
        ----------
        point : Point
            DESCRIPTION.
        point_origin_SRC_EPSG : str
            EPSG relative to the shapely.geometry.Point's coordinate reference system
        target_SRC_EPSG : str
            EPSG relative of the coordinate reference system

            Returns
            ----------
            
            np.NDArray[2,N], being N = number of pairs of [ycoords:float, xcoords:float].

        """
    
    
    
    
class PointConverter(Transformer):
    
    def pointToArray(self, coords: Point) -> Tuple[float, float]:
        """ convert's a shapely.geometry.Point to the
        a Tuple[float, float]
        
        Keyword arguments:
        coords - a shapely.geometry.Point object
        
        
        Returns
            1D of length 1 numpy array = [ycoords:float, xcoords:float]
        """

        if not isinstance(coords, Point):
            raise TypeError("Coordinates must be of type shapely.geometry.Point")
        
       
        yx = np.array([coords.y, coords.x])
        
        return yx
    
    def transform(self, point: Point, point_origin_SRC_EPSG: str, target_SRC_EPSG: str):
        """ Transform coordinates from a shapely.geometry.Point to the
        netcdf target's coordinate reference system
        
        Keyword arguments:
        point - a shapely.geometry.Point object

        
        
        Returns
        ---------
            np.NDArray[2,N], being N = number of pairs of [ycoords:float, xcoords:float].
        """

        if not isinstance(point, Point):
            raise TypeError("Coordinates must be of type shapely.geometry.Point")
        
        else:
            ycoord, xcoord = self.pointToArray(point)
        
        Point_coordinate_transformer = CRSTransformer.from_crs("epsg:{0}".format(point_origin_SRC_EPSG), 
                                           "epsg:{0}".format(target_SRC_EPSG))
        
        new_coord = Point_coordinate_transformer.transform(ycoord,
                                                           xcoord)
        
        new_coord_asArray = np.column_stack(new_coord)

        return new_coord_asArray


class ArrayConverter(Transformer):
    
    
    def transform(self, xyArrayCoordinates: , point_origin_SRC_EPSG: str, target_SRC_EPSG: str):
        """ Transform coordinates from a shapely.geometry.Point to the
        netcdf target's coordinate reference system
        
        Keyword arguments:
        point - a shapely.geometry.Point object

        Returns
        ---------
            np.NDArray[2,N], being N = number of pairs of [ycoords:float, xcoords:float].
        """

        if not isinstance(point, np.NDArray):
            raise TypeError("Coordinates must be of type np.array")
        
        else:
            ycoord, xcoord= self.pointToArray(point)
        
        Point_coordinate_transformer = CRSTransformer.from_crs("epsg:{0}".format(point_origin_SRC_EPSG), 
                                           "epsg:{0}".format(target_SRC_EPSG))
        
        new_coord = Point_coordinate_transformer.transform(ycoord, xcoord)
        
        new_coord_asArray = np.column_stack(new_coord)

        return new_coord_asArray
    
    

class TupleConverter(Transformer):
    
    def transform(self, point: Point, point_origin_SRC_EPSG: str, target_SRC_EPSG: str):
        """ Transform coordinates from a shapely.geometry.Point to the
        netcdf target's coordinate reference system
        
        Keyword arguments:
        point - a shapely.geometry.Point object
        
        returns
            1x2 array
        """

        if not isinstance(point, tuple) and len(point)==2:
            raise TypeError("Coordinates must be of type shapely.geometry.Point")
        
        else:
            ycoord, xcoord= np.stack([point])


        Point_coordinate_transformer = CRSTransformer.from_crs("epsg:{0}".format(point_origin_SRC_EPSG), 
                                           "epsg:{0}".format(target_SRC_EPSG))
        
        new_coord = Point_coordinate_transformer.transform(ycoord, xcoord)
        
        new_coord_asArray = np.column_stack(new_coord)

        return new_coord_asArray
        
    


class KdtreeWithReprojectionSearch:
    
    """ 
    
    Description:
    
    A KD-tree implementation for fast point lookup on a 2D grid
    
    This class diverges from the KdtreePolarCoordinateSearch in terms of the 
    pre-processing of the coordinates that is done prior to building the
    Kdtree.
    
    
    Here, the coordinates are reprojected into a planar coordinate
    reference system (CRS), in terms of x and y axis,
    prior to creating the kdtree. Therefore, one must be aware of the 
    original CRS from the dataset and the desired (target) CRS that the coordinates
    will be converted into prior to building the search tree.
    
   
    
    Implementation Reference:
        https://notes.stefanomattia.net/2017/12/12/The-quest-to-find-the-closest-ground-pixel/
        
        
               
               
    """
    
    def __init__(self, 
                 dataset: Union[xr.DataArray,xr.Dataset], 
                 lat_coord_name='lat',
                 lon_coord_name='lon',
                 da_array_crs_epsg=4326, 
                 target_epsg=4978):
        """
        

        Parameters
        ----------
        dataset : Union[xr.DataArray,xr.Dataset]
            DESCRIPTION.
        lat_coord_name : TYPE, optional
            DESCRIPTION. The default is 'lat'.
        lon_coord_name : TYPE, optional
            DESCRIPTION. The default is 'lon'.
        da_array_crs_epsg : TYPE, optional
            DESCRIPTION. The default is 4326.
        target_epsg : TYPE, optional
            DESCRIPTION. The default is 4978.

        Returns
        -------
        None.

        """
        
        self.da_array_crs_epsg = da_array_crs_epsg
        self.target_epsg = target_epsg
        self.transformer = CRSTransformer.from_crs("epsg:{0}".format(da_array_crs_epsg), 
                                           "epsg:{0}".format(target_epsg))
        
        
        
        self.dataset = dataset
        # store original dataset shape
        self.shape = dataset.shape
        
        # reshape and stack coordinates
        coords = np.column_stack((dataset.coords[lat_coord_name].values.ravel(),
                                  dataset.coords[lon_coord_name].values.ravel()))
        
        # construct KD-tree
        self.tree = spatial.cKDTree(self.transform_coordinates(coords))
        
        
    
    def transform_coordinates(self, point: List[Point], point_origin_SRC_EPSG:"str" = "4326"):
        """ Transform coordinates from geodetic to cartesian
        
        Keyword arguments:
        coords - a set of lan/lon coordinates (e.g. a tuple or 
                 an array of tuples)
        """


        if isinstance(point, [tuple, list]) and len(point)==2:
            new_coord = TupleConverter.transform(point, point_origin_SRC_EPSG, self.target_epsg)
        
        
        elif isinstance(point, Point):
            new_coord = PointConverter.transform(point, point_origin_SRC_EPSG, self.target_epsg)
        
        else:
           raise TypeError("point type unknown")


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
