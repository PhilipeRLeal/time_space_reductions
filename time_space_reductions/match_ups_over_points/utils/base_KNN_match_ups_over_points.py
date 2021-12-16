from shapely import geometry

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

from .base_class_for_query_of_nearest_points import Query_Nearest_Points


def _get_nearest_pixels(ground_pixel_tree, xy, k):
    
    
    
    error_message = 'The geometry must have both lon and latitude (geographic crs), \
        or x and y (planar crs).\n \
            The current geometry is {0} \n {1}'.format(xy, type(xy))
    
    if isinstance(xy, geometry.Point):

        point = (xy.y, xy.x)
    
    elif len(xy) == 2:
        point = xy
    elif isinstance(xy, pd.Series):
        if hasattr(xy, 'geometry'):
            point = xy['geometry']
        else:
            raise AttributeError(error_message)
    else:
        raise AttributeError(error_message)
        
    rome_index = ground_pixel_tree.query(point, k)
    
    return ground_pixel_tree.dataset[rome_index]

def get_nearest_pixels(da_array, 
                       xy, 
                       k,
                       lat_coord_name='lat',
                       lon_coord_name='lon',
                       da_array_crs_epsg=4326,
                       target_epsg=4978):
    
    ground_pixel_tree = Query_Nearest_Points(da_array, 
                                             lat_coord_name=lat_coord_name,
                                             lon_coord_name=lon_coord_name,
                                             da_array_crs_epsg=da_array_crs_epsg,
                                             target_epsg=target_epsg)
    
    return _get_nearest_pixels(ground_pixel_tree, xy, k=k)


def reduce_df(da, reductions = {'mean', 'max', 'min', 'std', 'count'},
              variable_columns=[]):
    
    df = da.to_dataframe()
    df_reduced = df.agg(reductions)

    df_reduced = df_reduced.loc[: , variable_columns]
    
    return df_reduced    
    



def query_temporal_K_nearest_points(ground_pixel_tree,
                                    xy,
                                    k,
                                    da_time_coord_name = 'time',
                                    gdf_time_coord_name='datetime',                                          
                                    dict_of_windows = {'time_window':3,
                                                       'time_unit':'D'}                                        
                                    ):
    if len(xy.name) >1:
        for enum, i in enumerate(xy.name):
            if isinstance(i, pd.Timestamp):
                t_index = enum
                
    time_delta = pd.Timedelta(dict_of_windows['time_window'],
                              dict_of_windows['time_unit'])
    
    dt = xy.name[t_index]
    
    time_init = dt - time_delta
            
            
    time_end = dt + time_delta
    
    
    
    
    sliced_da = _get_nearest_pixels(ground_pixel_tree,
                                    xy=xy, 
                                    k=k)
    
    
    
    
    
    sliced_da = sliced_da.sel({da_time_coord_name:slice(time_init, time_end)})
    
    
    print('Time sliced da: ', sliced_da)
    
    print('\n'*5)
    
    return sliced_da


def get_time_dependent_KNN_pixels_from_gdf(da, 
                                           gdf, 
                                           k,
                                           lat_coord_name='lat',
                                           lon_coord_name='lon',
                                           da_time_coord_name = 'time',
                                           gdf_time_coord_name='datetime',
                                           da_array_crs_epsg=4326,
                                           target_epsg=4978,
                                           dict_of_windows = {'time_window':3,
                                                       'time_unit':'D'} ):
    
    ground_pixel_tree = Query_Nearest_Points(da, 
                                             lat_coord_name=lat_coord_name,
                                             lon_coord_name=lon_coord_name,
                                             da_array_crs_epsg=da_array_crs_epsg,
                                             target_epsg=target_epsg)
    
                                    
            
    match_ups = gdf.apply(lambda xy: query_temporal_K_nearest_points(ground_pixel_tree,
                                                                     xy,
                                                                     k,
                                                                     da_time_coord_name,
                                                                     gdf_time_coord_name,
                                                                     dict_of_windows),
                                                                     
                                     axis=1)
    
    match_ups.name = 'Variables'
    

    return match_ups    
                                
    
def get_time_dependent_matchups_from_gdf(da, 
                                         gdf, 
                                         k,
                                         reductions = {'mean', 'max', 'min', 'std', 'count'},
                                         lat_coord_name='lat',
                                         lon_coord_name='lon',
                                         da_time_coord_name = 'time',
                                         gdf_time_coord_name='datetime',
                                         da_array_crs_epsg=4326,
                                         target_epsg=4978,
                                         dict_of_windows = {'time_window':3,
                                                            'time_unit':'D'} ):
    
    
    Match_ups = get_time_dependent_KNN_pixels_from_gdf(da, 
                                                       gdf, 
                                                       k,
                                                       lat_coord_name,
                                                       lon_coord_name,
                                                       da_time_coord_name,
                                                       gdf_time_coord_name,
                                                       da_array_crs_epsg,
                                                       target_epsg,
                                                       dict_of_windows)
    
    Reductions = _concat_reduce(Match_ups,
                                gdf,
                                reductions,
                                )
    return Reductions

def get_nearest_pixels_from_gdf(da_array, gdf, k,
                                lat_coord_name='lat',
                                lon_coord_name='lon',
                                da_array_crs_epsg=4326,
                                target_epsg=4978,):
    
    
                                
    '''

    Description:
        This function evaluates the nearest pixels in a xr.dataarray given a geopandas.GeoDataFrame.
        it does not consider time in the analysis. If time is of importance,
        use the function 'get_time_dependent_KNN_pixels_from_gdf' - also available in this module
        
        
        
    Returns 
        a numpy of xr.DataArrays. 
        
        Each xr.DataArray contains the values of the K-nearest pixels closest to each point of the gdf.

    '''
    
    ground_pixel_tree = Query_Nearest_Points(da_array, 
                                             lat_coord_name=lat_coord_name,
                                             lon_coord_name=lon_coord_name,
                                             da_array_crs_epsg=da_array_crs_epsg,
                                             target_epsg=target_epsg)
    
    
   
    match_ups = gdf.geometry.apply(lambda geom: _get_nearest_pixels(ground_pixel_tree,
                                                             xy=geom, 
                                                             k=k))
    match_ups.name = 'Variables'
    
    
    
    return match_ups



def get_only_KNN_spatial_matchUps(da_array, 
                             gdf, 
                             k,
                             reductions = {'mean', 'max', 'min', 'std', 'count'},
                             lat_coord_name='lat',
                             lon_coord_name='lon',
                             da_array_crs_epsg=4326,
                             target_epsg=4978):
    
    '''
    Description:
        This function applies match-Up operations over netcdf files given \
        a geopandas.GeoDataFrame (gdf)  of single-points.
        
        Steps for analysis:
        
        1) applies a KD-Tree for extraction of the closest pixel positions\
        to each given geometry in a gdf.
        
        2) slices the xr.DataArray given these pixel positions
        
        3) applies a series of reductions over each of these sliced xr.DatArrays
        
        4) concatenates back the results with the given gdf
        
        5) returns the updated gdf
    
    
    Parameters:
    
        da (xr.DataArray): the dataArray from which the analysis will be done 
        
        gdf (geopandas.GeoDataFrame)
        
        k (int==1, as standard): the number of closest pixels for evaluate for each geometry
        
        reductions (set == {'mean', 'max', 'min', 'std', 'count'}, as Standard): the set of reduction operations for be evaluated. The set also supports custom reduction functions (as described in the pandas library).
    
    
    
    Returns:
        gdf (geopandas.GeoDataFrame): with the new column added, and another index-column regarding the reduction operation.
        
    
    Algo Requirements:
        The algorithm requires the xarray, geopandas, shapely, scipy-KdTree, \
        pandas, numpy libraries
         
    
    
    Example:
    
    #### Create a netcdf of Temperature data (regular or irregular grid).
     
     # Here, we consider a irregular grid, since in this case, \
     # the problem is even harder to solve (proving the algorithm)
        
        import numpy as np
        import xarray as xr
         
         
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
        
        
        
        ######## Now creating a geopandas (with position and time as MultiIndex)
        
        
        import pandas as pd
        import geopandas as gpd
        from shapely.geometry import Point
        rome = (41.9028, 12.4964) # lat, lon
        paris = (48.8566, 2.3522) # lat, lon
        london = (51.5074, 0.1278)# lat, lon


        dates = pd.date_range('2008-01-01', freq='M', periods=2)
        index = pd.MultiIndex.from_product([np.arange(3),dates],
                                 names=['Locations','Dates'])

        gdf = gpd.GeoSeries([Point(x[::-1]) for x in [rome, paris, london, rome, paris, london]], 
                               index=index,
                               name='geometry',
                              crs="EPSG:4326").to_frame()

    
    
        # Checking the created data
    
        print(gdf)
                                geometry
        Locations	Dates	    
        0	        2008-01-31	POINT (12.49640 41.90280)
                    2008-02-29	POINT (2.35220 48.85660)
        1	        2008-01-31	POINT (0.12780 51.50740)
                    2008-02-29	POINT (12.49640 41.90280)
        2	        2008-01-31	POINT (2.35220 48.85660)
                    2008-02-29	POINT (0.12780 51.50740)
    
    
        results = get_only_KNN_spatial_matchUps(temperatures, 
                                 pixels, 
                                 k=9,
                                 reductions = {'mean', 'max', 'min', 'std', 'count'})

        print(results)
        
                                            geometry	                    Temperature
        Locations	Dates	    reductions		
        0	        2008-01-31	max	        POINT (12.49640 41.90280)	    29.125004
                                min	        POINT (12.49640 41.90280)	    16.749057
                                count	    POINT (12.49640 41.90280)	    9.000000
                                std	        POINT (12.49640 41.90280)	    3.791392
                                mean	    POINT (12.49640 41.90280)	    22.829173
                    2008-02-29	max	        POINT (2.35220 48.85660)	    19.040629
                                min	        POINT (2.35220 48.85660)	    12.391792
                                count	    POINT (2.35220 48.85660)	    9.000000
                                std	        POINT (2.35220 48.85660)	    2.080726
                                mean	    POINT (2.35220 48.85660)	    15.743346
        1	        2008-01-31	max	        POINT (0.12780 51.50740)	    17.322450
                                min	        POINT (0.12780 51.50740)	7.077198
                                count	    POINT (0.12780 51.50740)	9.000000
                                std	        POINT (0.12780 51.50740)	3.624569
                                mean	    POINT (0.12780 51.50740)	12.162124
                    2008-02-29	max	        POINT (12.49640 41.90280)	29.125004
                                min	        POINT (12.49640 41.90280)	16.749057
                                count	    POINT (12.49640 41.90280)	9.000000
                                std	        POINT (12.49640 41.90280)	3.791392
                                mean	    POINT (12.49640 41.90280)	22.829173
        2	        2008-01-31	max	        POINT (2.35220 48.85660)	19.040629
                                min	        POINT (2.35220 48.85660)	12.391792
                                count	    POINT (2.35220 48.85660)	9.000000
                                std	        POINT (2.35220 48.85660)	2.080726
                                mean	    POINT (2.35220 48.85660)	15.743346
                    2008-02-29	max	        POINT (0.12780 51.50740)	17.322450
                                min	        POINT (0.12780 51.50740)	7.077198
                                count	    POINT (0.12780 51.50740)	9.000000
                                std	        POINT (0.12780 51.50740)	3.624569
                                mean	    POINT (0.12780 51.50740)	12.162124
                    
        
    
    
    '''    
    
    # if int(gdf.crs.to_authority()[1]) != target_epsg:
    #     gdf = gdf.to_crs(epsg=int(target_epsg))
    
    
    Match_ups = get_nearest_pixels_from_gdf(da_array=da_array, gdf=gdf, k=k,
                                            lat_coord_name=lat_coord_name,
                                            lon_coord_name=lon_coord_name,
                                            da_array_crs_epsg=da_array_crs_epsg,
                                            target_epsg=target_epsg)

    

    Reductions = _concat_reduce(Match_ups,
                                gdf,
                                reductions,
                                )
    
    return Reductions


def _concat_reduce(Match_ups,
                   gdf,
                   reductions = {'mean', 'max', 'min', 'std', 'count'},
                             ):
    
    Reductions = {}
    for idx, da_array in Match_ups.items():
        
        reduced = reduce_df(da_array, reductions,
                            variable_columns=da_array.name)
        reduced.name = idx

        Reductions[idx] = reduced
    
    Reductions = pd.DataFrame(Reductions).T.stack().to_frame(da_array.name)
    

    print('Reductions: \n\n')
    
    print(Reductions)
    
    if isinstance(gdf.index, pd.Index) and not isinstance(gdf.index, pd.MultiIndex):
        name = gdf.index.name
        Reductions.index.names = [name, 'reductions']
        
    elif isinstance(gdf.index, pd.MultiIndex):    

        Reductions.index.names = gdf.index.names + ['reductions']
    


    Reductions = (gdf.merge(Reductions.reset_index('reductions'), 
                 on=gdf.index.names).set_index('reductions', append=True)

                 )
    return Reductions
    