# -*- coding: utf-8 -*-
"""
Editor Spyder

Este é um arquivo de script temporário.
"""
import numpy as np
import xarray as xr
import geopandas as gpd
import pandas as pd


from .utils.base_KNN_match_ups_over_points import (get_only_KNN_spatial_matchUps, 
                                                   get_time_dependent_matchups_from_gdf)

from .utils.metric_distance_base_match_ups_over_points import (get_only_spatial_radius_based_matchUps, 
                                                              get_radius_and_time_dependent_nearest_pixels_from_gdf)



class XR_GDF_Pixels_MatchUps():
    def __init__(self, 
                 da, 
                 gdf, 
                 k=1,
                 reductions = {'mean', 'max', 'min', 'std', 'count'},
                 lat_coord_name='lat',
                 lon_coord_name='lon',
                 da_time_coord_name = 'time',
                 gdf_time_coord_name='datetime',
                 ds_crs_epsg=4326,
                 target_epsg=4978):
                 
        self.da = da
        self.gdf = gdf
        self.k = k
        self.reductions = reductions
        self.lat_coord_name = lat_coord_name
        self.lon_coord_name = lon_coord_name
        self.ds_crs_epsg = ds_crs_epsg
        self.target_epsg = target_epsg
        self.da_time_coord_name = da_time_coord_name
        self.gdf_time_coord_name = gdf_time_coord_name


    def get_only_spatial_KNN_matchUps(self,):
    
        if isinstance(self.gdf, gpd.GeoSeries):
            self.gdf = self.gdf.to_frame()
        
        self.gdf = get_only_KNN_spatial_matchUps(self.da, 
                           self.gdf, 
                           self.k,
                           self.reductions,
                           self.lat_coord_name,
                           self.lon_coord_name,
                           self.ds_crs_epsg,
                           self.target_epsg)
        
        return self.gdf
    
    
    def get_only_spatial_radius_based_matchUps(self, radius):
    
        return get_only_spatial_radius_based_matchUps(self.da, 
                                                      self.gdf, 
                                                      radius,
                                                      self.reductions,
                                                      self.lat_coord_name,
                                                      self.lon_coord_name,
                                                      self.ds_crs_epsg,
                                                      self.target_epsg)
    
    
    
    
    def get_time_space_KNN_matchUps(self,
                                    k=None,
                                    dict_of_windows = {'time_window':3,
                                                       'time_unit':'D'} ):
        if k == None:
            k = self.k
        
        
        
        
        return get_time_dependent_matchups_from_gdf(self.da, 
                                                    self.gdf, 
                                                    k,
                                                    self.reductions,
                                                    self.lat_coord_name,
                                                    self.lon_coord_name,
                                                    self.da_time_coord_name,
                                                    self.gdf_time_coord_name,
                                                    self.ds_crs_epsg,
                                                    self.target_epsg,
                                                    dict_of_windows=dict_of_windows)
    
    def get_time_space_radius_matchUps(self, radius):

        return get_radius_and_time_dependent_nearest_pixels_from_gdf(self.da, 
                                                   self.gdf, 
                                                   radius,
                                                   self.lat_coord_name,
                                                   self.lon_coord_name,
                                                   self.da_time_coord_name,
                                                   self.gdf_time_coord_name,
                                                   self.ds_crs_epsg,
                                                   self.target_epsg)

        
    