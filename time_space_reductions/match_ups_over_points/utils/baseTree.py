# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:46:36 2023

@author: 55119
"""

import xarray as xr
import numpy as np
from numpy import cos, sin
from math import pi

from numpy.typing import ArrayLike, NDArray

from typing import Union, List, Annotated, Literal, TypeVar
from abc import ABC, abstractmethod

END = "\n"*3


DType = TypeVar("DType", bound=np.generic)

Array3xN = Annotated[NDArray[DType], Literal[3, "N"]]

Array2xN = Annotated[NDArray[DType], Literal[2, "N"]]

ArrayNx2 = Annotated[NDArray[DType], Literal["N", 2]]

Array1xN = Annotated[NDArray[DType], Literal[1, "N"]]

class BaseTree(ABC):
    
    @abstractmethod
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
        
        pass
    
    
    def toPolarCoordinates(self, 
                           latitudes:NDArray, 
                           longitudes:NDArray) -> Array3xN:
        """
        Method responsible for converting longitude and latitude coordinates into
        an array of 3D polar coordinates

        Parameters
        ----------
        latitudes : NDArray
            DESCRIPTION.
        longitudes : NDArray
            DESCRIPTION.

        Returns
        -------
        Array3xN
            DESCRIPTION.

        """
        

        rad_factor = pi/180.0 # for trignometry, need angles in radians

        latvalsInRadians = latitudes * rad_factor

        lonvalsInRadians = longitudes * rad_factor

        clat, clon = cos(latvalsInRadians), cos(lonvalsInRadians)

        slat, slon = sin(latvalsInRadians), sin(lonvalsInRadians)

        triples = zip(np.ravel(clat*clon.values), np.ravel(clat*slon.values), np.ravel(slat.values))
        
        polarCoordinates = np.array([x for x in triples])
        
        return polarCoordinates
    
    
    def fixRangeOfCoordinates(self, 
                              ds:Union[xr.DataArray,xr.Dataset], 
                              latCoordVarname:str, 
                              lonCoordVarname:str):
        self.ds = ds
        
        self.latvarname = latCoordVarname
        self.lonvarname = lonCoordVarname

        
        self.ds.coords[self.lonvarname] = (self.ds.coords[self.lonvarname] + 180) % 360 - 180
        self.ds.coords[self.latvarname] = (self.ds.coords[self.latvarname] + 90) % 180 - 90
        
        
    def pointsToXrArray(self, 
                        points: List[ArrayNx2],
                        xdimName: str,
                        ydimName: str) -> Union[xr.DataArray,xr.Dataset]:
        """
        Transforms the provided points's list into an xr.DataArray 
        (or xr.Dataset) derived from the provided self.ds itself.

        Parameters
        ----------
        points : List[ArrayNx2]
            a list of Nx2 samples of pairs of latitudes and longitudes
        xdimName : str
            the X dimension name of the xarray.DataArray|Dataset
        ydimName : str
            the Y dimension name of the xarray.DataArray|Dataset

        Returns
        -------
        nearest_points : Union[xr.DataArray,xr.Dataset]
            the same type of the input.

        """
        
        if (np.array(points).ndim):
            points = np.array(points).reshape(1, -1)
        
        ydataArray = xr.DataArray(points[:,0], dims=[self.latvarname])
        xdataArray = xr.DataArray(points[:,1], dims=[self.lonvarname])
        
        nearest_points = self.ds.isel({xdimName:ydataArray, ydimName:xdataArray})
        
        return nearest_points
    