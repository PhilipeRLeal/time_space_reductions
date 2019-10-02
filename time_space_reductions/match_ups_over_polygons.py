# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 18:10:15 2019

@author: lealp
"""

import pandas as pd
import geopandas as gpd

try:
    from netcdf_gdf_setter import Base_class_space_time_netcdf_gdf
except:
    
    from .netcdf_gdf_setter import Base_class_space_time_netcdf_gdf
    
import numpy as np

import warnings
from shapely.geometry import Point



def make_fake_data(N=200):
    
    # creating example GeoDataframe for match-ups in EPSG 4326
    
    xx = np.random.randint(low=-60, high=-33, size=N)*1.105
    
    yy = np.random.randint(low=-4, high=20, size=N)*1.105
    
    df = pd.DataFrame({'lon':xx, 'lat':yy})
    
    
    df['geometry'] = df.apply(lambda x: Point(x['lon'], x['lat']), axis=1)
    
    
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs={'init':'epsg:4326'})
    
    gdf['Datetime'] = pd.date_range('2010-05-19', '2010-06-24',  periods=gdf.shape[0])
    
    
    return gdf



class Space_Time_Agg_over_polygons(Base_class_space_time_netcdf_gdf):
    
    def __init__(self, gdf, xarray_dataset=None, 
                 netcdf_temporal_coord_name='time',
                 geo_series_temporal_attribute_name = 'Datetime',
                 longitude_dimension='lon',
                 latitude_dimension='lat',
				):
        
        
        '''
        Class description:
        ------------------
        
            This class is a base class for ensuring that the given netcdf is in conformity with the algorithm.
            
            Ex: the Netcdf has to be sorted in ascending order for all dimensions (ex: time ,longitude, latitude). 
            Otherwise, the returned algorithm would return Nan values for all slices
            
            Also, it is mandatory for the user to define the longitude and latitude dimension Names (ex: 'lon', 'lat'),
            since there is no stadardization for defining these properties in the Netcdf files worldwide.
            
            
            
        Attributes:
            
            gdf (geodataframe):
            -----------------------
                The geodataframe object containing geometries to be analyzed.
            
            xarray_dataset (None): 
            -----------------------
            
                the Xarry Netcdf object to be analyzed
                
                
            netcdf_temporal_coord_name (str = 'time'): 
            -----------------------------------
            
                the name of the time dimension in the netcdf file
                
                
            geo_series_temporal_attribute_name(str = 'Datetime'):
            -----------------------------------
            
                the name of the time dimension in the geoseries file    
                
                
            longitude_dimension (str = 'lon'): 
            ----------------------------------
            
                the name of the longitude/horizontal dimension in the netcdf file
            
            
            latitude_dimension (str = 'lat'): 
            ----------------------------------
                the name of the latitude/vertical dimension in the netcdf file
        

        
        '''
        
        self.__netcdf_ds = xarray_dataset
        
        self._temporal_coords = netcdf_temporal_coord_name
        self.__geo_series_temporal_attribute_name = geo_series_temporal_attribute_name
        # setting Standard spatial_coords/dimension names 
        self.spatial_coords = dict(x = longitude_dimension,
                                   y = latitude_dimension)
        
        self.__gdf = gdf
        
        
        self.__netcdf_ds = self.netcdf_ds.sortby([self._temporal_coords, 
                                                  longitude_dimension,
                                                  latitude_dimension])
        
        self.netcdf_ds = self._slice_bounding_box()
    
    
    @ property
    
    def netcdf_ds(self):
        
        return self.__netcdf_ds
    
    @ netcdf_ds.setter
    
    def netcdf_ds(self, new_netcdf_ds):
        '''
        This property-setter alters the former netcdf_ds for the new gdf provided
        
        
        '''
     
        self.__netcdf_ds = new_netcdf_ds
    
    
    
    @ property
    
    def gdf(self):
        
        return self.__gdf
    
    @ gdf.setter
    
    def gdf(self, new_gdf):
        '''
        This property-setter alters the former GDF for the new gdf provided
        
        
        '''
     
        self.__gdf = new_gdf
    
    
    def _slice_bounding_box(self):
        
        xmin, ymin, xmax, ymax = self.gdf.geometry.total_bounds
        
        
        dx = float(self.coord_resolution(self.spatial_coords['x']))
        
        dy = float(self.coord_resolution(self.spatial_coords['y']))
        
        xmin -= dx # to ensure full pixel slicing
        
        xmax += dx # to ensure full pixel slicing 
        
        ymin -= dy # to ensure full pixel slicing 
        
        ymax += dy # to ensure full pixel slicing 
        
        
        result = self.netcdf_ds.sel({self.spatial_coords['x']:slice(xmin, xmax),
                                     self.spatial_coords['y']:slice(ymin, ymax)})
        
        return result
        
    
    def _slice_time_interval(self, time_init, final_time):
        
        
        result = self.netcdf_ds.sel({self._temporal_coords:slice(time_init, final_time)})
        

        return result
        
    
    
    
    def _make_time_space_aggregations(self, 
                                      geoDataFrame, 
                                      date_offset,
                                      netcdf_varnames, 
                                      agg_functions):
        
        Tmin = geoDataFrame[self.__geo_series_temporal_attribute_name].min()
        
        Tmax = geoDataFrame[self.__geo_series_temporal_attribute_name].max()
        
        time_init = Tmin - date_offset
                
        final_time = Tmax + date_offset
        
        netcdf_sliced = self._slice_time_interval(time_init, final_time)
        
        netcdf_sliced_as_gpd_geodataframe = self.netcdf_to_gdf(netcdf_sliced).copy()
            
            
        
        if not  netcdf_sliced_as_gpd_geodataframe.empty:
            
            
            
            
            sjoined = gpd.sjoin(netcdf_sliced_as_gpd_geodataframe, geoDataFrame, how="inner", op='within')
            
            if not sjoined.empty:
                
                
                sjoined_agg = sjoined.groupby([self.__geo_series_temporal_attribute_name, 'IDX'])[netcdf_varnames].agg(agg_functions)
                
                
                sjoined_agg['period_sliced'] = time_init.strftime("%Y/%m/%d %H:%M:%S") + ' <-> ' + final_time.strftime("%Y/%m/%d %H:%M:%S")
                
                
               
                
                return sjoined_agg
               
            else:
                pass
    
        else:
            pass

    
    
    def _evaluate_space_time_agg(self, 
                                 netcdf_varnames=['adg_443_qaa'], 
                                 dict_of_windows=dict(time_window='1D'),
                                 agg_functions=['nanmean','nansum','nanstd'],
                                 verbose=True):
        
        self.gdf2 = gpd.GeoDataFrame()
        
        
        date_offset = pd.tseries.frequencies.to_offset(dict_of_windows['time_window'])
        
        self.gdf['IDX'] = np.arange(self.gdf.shape[0])
        
        
        for idx, group in self.gdf.groupby(self.gdf[self.__geo_series_temporal_attribute_name].unique()):
            
            
            group2 = self._make_time_space_aggregations(group.copy(),
                                                        date_offset=date_offset,
                                                        netcdf_varnames=netcdf_varnames,
                                                        agg_functions=agg_functions)
            
            if verbose == True:
                print('\n'*3,  group2,  '\n'*2)
            
            self.gdf2 = self.gdf2.append(group2)
            
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            self.gdf2 = (self.gdf[['geometry', 'IDX', self.__geo_series_temporal_attribute_name]]
                          .merge(self.gdf2, 
                                 on=['IDX', self.__geo_series_temporal_attribute_name])
                        )
                          
                      
        self.gdf2.crs = self.gdf.crs
            
        
        return self.gdf2.copy()
    
    
    
def _base(gdf,
          netcdf,
          netcdf_varnames =['adg_443_qaa'],
          netcdf_temporal_coord_name='time',
          geo_series_temporal_attribute_name = 'Datetime',
          longitude_dimension='lon',
          latitude_dimension='lat',
          dict_of_windows=dict(time_window='M'),
          agg_functions=['mean', 'max', 'min', 'std'],
          verbose=True):
    
    
    
    
    Match_Upper = Space_Time_Agg_over_polygons(  gdf=gdf, 
                                                 xarray_dataset=netcdf, 
                                                 netcdf_temporal_coord_name='time',
                                                 geo_series_temporal_attribute_name = 'Datetime',
                                                 longitude_dimension='lon',
                                                 latitude_dimension='lat')
    
   
    return Match_Upper._evaluate_space_time_agg(netcdf_varnames=['adg_443_qaa'], 
                                                dict_of_windows=dict_of_windows,
                                                agg_functions=agg_functions,
                                                verbose=verbose)
            
            
            

def get_zonal_match_up(netcdf, 
					   gdf, 
                       netcdf_varnames =['adg_443_qaa'],
                       dict_of_windows=dict(time_window='5D'),
                       agg_functions=['mean', 'max', 'min', 'std'],
                       verbose=True):
    
    """
    This function does Match - Up operations from centroids of Geoseries or GeoDataFrames over Netcdfs.
    
	Attributes:
	
		netcdf (xarray Dataset/Dataarray):
		--------------------------------------------------------------------------
		
		
		gdf (geopandas GeoDataFrame):
		--------------------------------------------------------------------------
		
		
		netcdf_varnames (list): a list containing the netcdf variable names to apply the aggregation.

			Example: netcdf_varnames=['adg_443_qaa']
		--------------------------------------------------------------------------
		
		
		dict_of_windows(dictionary)
		
		
			Example: dict_of_windows=dict(time_window='5D') # for 5 day window integration
			
				Other time integration options, follow pandas pattern (e.g.: 'Q', '3Y',...etc.)
							 
		--------------------------------------------------------------------------
		
		
		agg_functions(list):
		
			Example: agg_functions = ['mean', 'max', 'min', 'std']
			
		--------------------------------------------------------------------------
		
		
		verbose (bool): it sets the function to verbose (or not). 
		
			Example verbose=True
		--------------------------------------------------------------------------
		
		
		
	Returns:
		(geopandas GeoDataFrame)
		
	
    """
				   
    return _base(gdf=gdf, 
                netcdf=netcdf, 
                netcdf_varnames=netcdf_varnames,
                dict_of_windows=dict_of_windows,
                agg_functions=agg_functions,
                verbose=verbose
                )
                

