# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:01:04 2019

@author: lealp
"""

import pandas as pd
import glob

from shapely.geometry import Point
import geopandas as gpd
import numpy as np
import os
import xarray as xr


from unittest import TestCase



from time_space_reductions.match_ups_over_centroids import get_match_up_over_centroids
from time_space_reductions.match_ups_over_polygons import get_zonal_match_up


def make_fake_data(N=200):
    
    # creating example GeoDataframe for match-ups in EPSG 4326
    
    xx = np.random.randint(low=-60, high=-33, size=N)*1.105
    
    yy = np.random.randint(low=-4, high=20, size=N)*1.105
    
    df = pd.DataFrame({'lon':xx, 'lat':yy})
    
    
    df['geometry'] = df.apply(lambda x: Point(x['lon'], x['lat']), axis=1)
    
    
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs={'init':'epsg:4326'})
    
    gdf['Datetime'] = pd.date_range('2010-05-19', '2010-06-24',  periods=gdf.shape[0])
    
    gdf.crs = {'init' :'epsg:4326'}
    
    
    return gdf


def get_netcdf_example():
    
    cpath = '/'.join([os.path.dirname(__file__), 'data'])
    path_file = glob.glob(cpath + '/*.nc' )
    
    
    return  xr.open_mfdataset(path_file[0])



class Tester_match_ups_over_centroids(TestCase):
    def test_match_ups_over_centroids(self):
        
        gdf = make_fake_data()
        
        
        CDOM = get_netcdf_example()
    
        
        gdf2 = get_match_up_over_centroids(netcdf=CDOM,
                                           gdf=gdf, 
                                           netcdf_varnames =['adg_443_qaa'],
                                           verbose=True,
                                           dict_of_windows=dict(x_window=50, 
                                                                  y_window=50, 
                                                                  time_window=10,                                                               
                                                                  time_unit='d'),
                                           agg_functions=['mean', 'max', 'min', 'std'])
        

    
        
        self.assertTrue(isinstance(gdf2, gpd.geodataframe.GeoDataFrame))    





class Tester_Space_Time_Agg_over_polygons(TestCase):
    def test_match_ups_over_polygons(self):
        
        gdf = make_fake_data(3)
        
        gdf.geometry = gdf.geometry.buffer(2.15) # in degrees
		
        CDOM = get_netcdf_example()
		
        
        gdf2 = get_zonal_match_up(gdf=gdf, 
                                  netcdf=CDOM,
                                  netcdf_varnames =['adg_443_qaa'],
                                  dict_of_windows=dict(time_window='1M'),
                                  agg_functions=['mean', 'max', 'min', 'std']
                                  
                                   )
    
        
        self.assertTrue(isinstance(gdf2, gpd.geodataframe.GeoDataFrame))    





