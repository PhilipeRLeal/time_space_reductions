# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:46:57 2019

@author: lealp
"""


import pandas as pd
import numpy as np

try:
    from netcdf_gdf_setter import Base_class_space_time_netcdf_gdf
except:
    
    from .netcdf_gdf_setter import Base_class_space_time_netcdf_gdf

import geopandas as gpd

    

class _Space_Time_Agg_over_centroids(Base_class_space_time_netcdf_gdf):
    
    def get_centroids_from_geometries_of_gdf(self, feature):
        '''
        Function description:
            This function returns the centroids (longitude and latitude) from the features of the geodataframe 
            
        returns (array, array):
            x, y
        
        '''
    
        
        return (feature.geometry.centroid.x, feature.geometry.centroid.y)
    
    def get_neighbors_from_axis(self, 
                                axis_mean_centroid, 
                                delta_coordinate, 
                                window_size):
        '''
        Parameters:
            
            axis_mean_centroid: the mean coordinate to generate the convolutional slicing window
            
            delta_coordinate: the delta to be applied in the given coordinate
        
        '''
        
        return (  axis_mean_centroid - (window_size*delta_coordinate), 
                  axis_mean_centroid + (window_size*delta_coordinate))
        
    def space_time_window(self, 
                          geo_series, 
                          dict_of_windows=dict(x_window=5, y_window=1, time_window=None, time_unit='D', time_init='1970-01-01'),
                          verbose=False):
        
        '''
        Function description:
            
            x_window: it is in data coordinate units. (i.e.: longitude units). Therefore for a x_window of size 1, 
                      the resultant reduction operation will be applied over the Point and a 'x -1' and 'x+1' longitude coordinates
            
            y_window: it is in data coordinate units. (i.e.: latitude units).  Therefore for a y_window of size 1, 
                      the resultant reduction operation will be applied over the Point and a 'y -1' and 'y+1' latitude coordinates
            
            time_unit (string): it sets the time offset to be evaluated in the Match-Up
            
            time_init: the Datetime string for temporal filtering.
            
            
            For standard application, we suggest the use of x_window=1 and y_window=1. This way, the Match-Up will be applied 
                    over a 3x3 matrix. 
        
        Returns: (result, x, y)
        
            result: the space-time filtered netcdf DataArray
            x, y: the longitude/latitude coordinates of the centroid used
        
        '''
        x, y  = self.get_centroids_from_geometries_of_gdf(geo_series)
        
        dy = self.coord_resolution(self.spatial_coords['y'])
    
        dx = self.coord_resolution(self.spatial_coords['x'])
        
        
        # getting geographic window
        
        lon_init, lon_final = self.get_neighbors_from_axis(x, dx, dict_of_windows['x_window'])
        
        lat_init, lat_final = self.get_neighbors_from_axis(y, dy, dict_of_windows['y_window'])
        
        # sort netcdf in ascending mode (especially for time axis) to ensure slicing operations.

        dims = self.get_dims_names_as_list()
        
        df_sorted = self.netcdf_ds.sortby(dims)
        
        
        # adding a condition to time. Whether or not to apply a time slicing.
        
        if dict_of_windows['time_window'] != None:
            
            
            time_mean = geo_series[self._datetime_parameters]
            
            time_delta = pd.Timedelta(dict_of_windows['time_window'], dict_of_windows['time_unit'])
            
            time_init = time_mean - time_delta
            
            final_time = time_mean + time_delta
            
            
            
            # slicing both time and space.
            
            
            result = df_sorted.sel( { self.spatial_coords['y']:slice(lat_init, lat_final), 
                                      self.spatial_coords['x']:slice(lon_init, lon_final),
                                      self.temporal_coords:slice(time_init, final_time)
                                    })
            
            
                
        else:
            
            # slice only spatially. Therefore, all time intervals will be aggregated.
            
            result = df_sorted.sel( { self.spatial_coords['y']:slice(lat_init, lat_final), 
                                      self.spatial_coords['x']:slice(lon_init, lon_final)
                                    })
            
        if verbose == True:
            print('\n\n', '\t'*2, 'Sliced time space window {0}'.format(dict(result.dims)), '\n'*2)
            
        return result.compute()
  
    def _aggregation_over_time_space(self, 
                                    geo_series,
                                    agg_functions=['nanmean','nansum','nanstd', 'nanmax', 'nanmin'], 
                                    netcdf_varnames=['air'],
                                    dict_of_windows=dict(x_window=5, 
                                                         y_window=1, 
                                                         time_window=None,        
                                                         time_unit='D'),
                                                         
                                    verbose=False):
                                        
        '''
        Function description:
            This is the main function for Match-Up (space-time aggregation) operations. It operates space and time 
            slicing operation over the netcdf dataset and applies a reduction operation over the nearest coordinates
            of each centroid of the geodataframe.
            
            
        Parameters:
            geo_series: 
            -----------
            
                the Geopandas geoseries object to be applied the operation
                
                
            agg_functions (list): 
            -----------
            
                The reduction functions that will be applied over the netcdf. 
                It must be a list of numpy name functions, or list containing 
                callables (user defined functions), or a mixture of both. 
                The order of the callables in the agg_function are not important.
                
                
            netcdf_varnames (list): 
            ----------------------
            
                the list of variables to apply the reduction over the netcdf
            
            dict_of_windows (dict): 
            ------------------------
            
                A dictionary containing the information for slicing the netcdf 
                according to longitude (x_window), latitude (y_window) and time 
                (time_window) operations.
                
               
                Dict attributes:
                ------------------------
                        
                    x_window: 
                    -----------------
                              the amount of x coordinates that will be used to create a bounding box around each centroid 
                              in the netcdf slicing (Attention: this operation is in 'X' coordinate units, and not in matrix units)
                              
                    y_window: 
                    -----------------
                            the amount of y coordinates that will be used to create a bounding box around each centroid 
                            in the netcdf slicing (Attention: this operation is in 'Y' coordinate units, and not in matrix units)
                    
                
                    time_window(float): 
                    -------------------
                    
                        the amount of time coordinates that will be used to create a bounding box 
                        around each centroid in the netcdf time slicing operation 
                        (Attention: this operation is in 'time_unit' coordinate units, 
                        and not in matrix units).
                        
                        For standard time_window=None. If None is applied, no time slice operation is applied. 
                        Only spatial reduction is applied. In this case, all time 
                        instants are aggregated for a given bounding box.
                
                
                time_unit(string): 
                -------------------
                
                    the string names that can be used in the pandas Timedelta operation
                                   
                                    {'ns': nanoseconds
                                    'us':microseconds
                                    'ms': miliseconds
                                    's': secibds
                                    'm': minutes
                                    'h':hours'
                                    'D': days
                                    'W': weeks
                                    'M': months
                                    'Y': years}
                                    
                                    
                verbose(Boolean): 
                -------------------
                
                    if False (standard), no slicing message operation is printed in the screen. 
                    If True, all slice operations are printed in the screen.
                                    
                                    
        Returns:
            A new geodataframe with the new columns. One column for each reduced operation for each netcdf_varname
        
        '''
                                        
        
        # checking the list of aggregation functions. 
        # If any is a custom function, a custom name will be created for each one separately and in order.
        
        agg_function_names = []                
        
        custom_counter=0
        for agg_function in agg_functions:
            if isinstance(agg_function, str):
                
                agg_function_names.append(agg_function)
            
            else:
                if custom_counter >0:
                    
                    agg_function_names.append('custom_agg_function'+'_' +str(custom_counter))
                
                else:                   
                    agg_function_names.append('custom_agg_function')
                    
                custom_counter +=1
        
        
        
        # Starting the aggregation algorithm:
        
            # gettin the sliced netcdf.
        
        df = self.space_time_window(geo_series, dict_of_windows, verbose=verbose)
        
        # creating a single multi Index for the pandas Series aggregated data:
        index = pd.MultiIndex.from_product( (netcdf_varnames ,agg_functions)) 
        
      
        
        # checking if the slice resulted in an empty Xarray-Dataset / DataArray
        
        if df.to_dataframe().empty == True:
            # if empty, return nan to the given feature
            
            agg_results = pd.Series(np.nan, index=index)
            
        
        else:

            # effective sliced pixels that are not masked.
            
            if df.to_dataframe().dropna().empty == True:
                if verbose == True:
                    
                    
                    print('all nan cases for the sliced netcdf','\n',
                          'Setting to NaN all aggregation functions')
                else:
                    pass
                
                agg_results = pd.Series(np.nan, index=index)
                 
            else:
                
                agg_results =  df[netcdf_varnames].to_dataframe().agg(agg_functions)
                
                agg_results = agg_results.melt() # converting to a pandas Series
    
                agg_results = pd.Series(agg_results['value'].values, index= index)
                
        # adding the aggregated data to the Geoseries (being NaNs or real aggregated values):
        
        geo_series = geo_series.append(agg_results)
        
        
        if verbose == True:
            print(geo_series, '\n'*2)
        
        
        return geo_series
    
    
    def __call__(self,
                 geo_series, 
                 netcdf_varnames=['adg_443_qaa'],
                 agg_functions=['nanmean','nansum','nanstd', 'nanmax', 'nanmin'], 
                 dict_of_windows=dict(x_window=5, y_window=1, time_window=None, time_unit='D'),
                 verbose=False):
        
        """
            Match up analyses can be done by directly calling the given class, as long as one passes
            
            the geo_series, and at least the netcdf_varname to be applied over.
            
            Parameters:
                netcdf_ds: the xarray netcdf dataframe object
                gdf: the geodataframe object
                agg_functions (list - optional): it can be a list of callables or a numpy callable
                netcdf_varnames: a list of all netcdf variables that must be evaluate by the match-up algorithm
                dict_of_windows(dict): a dictionary containing the windows to be applied for each dimension in the xarray
        
        """
        
        
        return self._aggregation_over_time_space(geo_series=geo_series, 
                                                 agg_functions=agg_functions, 
                                                 netcdf_varnames=netcdf_varnames, 
                                                 dict_of_windows=dict_of_windows, 
                                                 verbose=verbose)
        
        
        
        
class Match_up_over_centroids(object):
    def __init__(self, netcdf,
                 netcdf_temporal_coord_name='time',
                 geo_series_temporal_attribute_name = 'Datetime',
                 longitude_dimension='lon',
                 latitude_dimension='lat',):
        
        
        self.Match_Upper = _Space_Time_Agg_over_centroids(netcdf,
                                                         netcdf_temporal_coord_name=netcdf_temporal_coord_name,
                                                         geo_series_temporal_attribute_name=geo_series_temporal_attribute_name,
                                                         latitude_dimension=latitude_dimension,
                                                         longitude_dimension=longitude_dimension)
 

    def _base(self, geo_series,
                      netcdf_varnames =['adg_443_qaa'],
                      verbose=True,
                      dict_of_windows=dict(x_window=200, 
                                                  y_window=20000, 
                                                  time_window=50,                                                               
                                                  time_unit='d'),
                      agg_functions=['mean', 'max', 'min', 'std']
                            
                            ):
  
        
       
        return self.Match_Upper(geo_series=geo_series,
                           netcdf_varnames=netcdf_varnames,
                           verbose = verbose,
                           agg_functions=agg_functions,
                           dict_of_windows = dict_of_windows)
        
    
             
                         
                     
def get_match_up_over_centroids(netcdf, 
                                 gdf, 
                                 netcdf_varnames =['adg_443_qaa'],
                                 verbose=True,
                                 dict_of_windows=dict(x_window=200, 
                                                      y_window=20000,
                                                      time_window=50, 
                                                      time_unit='d'),
                                 agg_functions=['mean', 'max', 'min', 'std']):
                    
    """
    This function does Match - Up operations from centroids of Geoseries or GeoDataFrames over Netcdfs.
    
	Attributes:
	
		netcdf (xarray Dataset/Dataarray):
		--------------------------------------------------------------------------
		
		
		gdf (geopandas geoseries/GeoDataFrame):
		--------------------------------------------------------------------------
		
		
		netcdf_varnames (list): a list containing the netcdf variable names to apply the aggregation.

			Example: netcdf_varnames=['adg_443_qaa']
		--------------------------------------------------------------------------
		
		verbose (bool): it sets the function to verbose (or not). 
		
			Example verbose=True
		--------------------------------------------------------------------------
		
		
		dict_of_windows(dictionary)
		
		
			Example: dict_of_windows = dict(x_window=20, # this will look for a 'x' window of 20 coordinate units
                             y_window=20, # this will look for a 'y' window of 20 coordinate units
                             time_window=5,  # this will look for a 'time_unit' window of 5 coordinate units
                             time_unit='D')  # this sets the time unit for "Day". Other options use sintax from pandas.
							 
		--------------------------------------------------------------------------
		
		
		agg_functions(list):
		
			Example: agg_functions = ['mean', 'max', 'min', 'std']
			
		--------------------------------------------------------------------------
    
	Returns:
		(geopandas geoseries/GeoDataFrame)
		
	
    """
    
    Match_upper = Match_up_over_centroids(netcdf) 
    
    if isinstance(gdf, (gpd.geodataframe.GeoDataFrame)):
        
       return gdf.apply(lambda geo_series: Match_upper._base(geo_series = geo_series,
                                                      netcdf_varnames = netcdf_varnames,
                                                      verbose=verbose,
                                                      dict_of_windows=dict_of_windows, 
                                                      agg_functions=agg_functions),
    
                     axis=1  )
        
    else:
         return Match_upper._base(geo_series = gdf,
                           netcdf_varnames = netcdf_varnames,
                           verbose=verbose,
                           dict_of_windows=dict_of_windows)





