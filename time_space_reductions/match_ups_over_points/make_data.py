# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:49:09 2020

@author: Philipe_Leal
"""
import pandas as pd
import numpy as np

import geopandas as gpd

import xarray as xr
from shapely.geometry import Point
 

def _make_xrArray():
    num_sl = 12 # number of scanlines
    num_gp = 10 # number of ground pixels
    
    lon, lat = np.meshgrid(np.linspace(-20, 20, num_gp),
                           np.linspace(30, 60, num_sl))
    lon += lat/10
    lat += lon/10
    
    data = (np.linspace(30, 0, num_sl*num_gp).reshape(num_sl, num_gp) + 
             6*np.random.rand(num_sl, num_gp))
    
    temperatures = xr.DataArray(data, dims=['scanline', 'ground_pixel'],
                                coords = {'lat': (('scanline', 'ground_pixel'), lat),
                                          'lon': (('scanline', 'ground_pixel'), lon)},
                               name='Temperature',
                               attrs={'Units':'Temperature in °Celsius'})
    return temperatures



def _make_xrArray2():
    num_sl = 12 # number of scanlines
    num_gp = 10 # number of ground pixels
    num_time = 20 # number of days since 2000
    lon, lat = np.meshgrid(np.linspace(-20, 20, num_gp),
                           np.linspace(30, 60, num_sl))
    lon += lat/10
    lat += lon/10
    
    times = pd.date_range('2000-01-01', freq='D', periods=num_time)
    
    data = (np.linspace(30, 0, num_sl*num_gp*num_time).reshape(num_sl, num_gp, num_time) + 
             6*np.random.rand(num_sl, num_gp, num_time))
    
    temperatures = xr.DataArray(data, dims=['scanline', 'ground_pixel', 'time'],
                                coords = {'lat': (('scanline', 'ground_pixel'), lat),
                                          'lon': (('scanline', 'ground_pixel'), lon),
                                          'time': (('time'), times)},
                               name='Temperature',
                               attrs={'Units':'Temperature in °Celsius'})
    return temperatures




def make_xrArray(with_time=False):
    
    if with_time:
        
        return _make_xrArray2()
    else:
        return _make_xrArray()
    



def make_gdf():

    
    
    rome = (41.9028, 12.4964) # lat, lon
    paris = (48.8566, 2.3522) # lat, lon
    london = (51.5074, 0.1278)# lat, lon
    
    
    dates = pd.date_range('2000-01-01', freq='D', periods=2)
    index = pd.MultiIndex.from_product([np.arange(3),dates],
                             names=['Locations','datetime'])
    
    gdf = gpd.GeoSeries([Point(x[::-1]) for x in [rome, paris, london, rome, paris, london]], 
                           index=index,
                           name='geometry',
                          crs="EPSG:4326").to_frame()
    
    return gdf
