# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:51:15 2020

@author: Philipe_Leal
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import geopandas as gpd
import cartopy.crs as ccrs

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)
import xarray as xr


from .make_data import make_gdf, make_xrArray

gdf, da = make_gdf(), make_xrArray()


from .gdf_xarray_space_reduction import XR_GDF_Pixels_MatchUps

Matchuper = XR_GDF_Pixels_MatchUps(da, gdf)


# Only spatial:

Results = Matchuper.get_only_spatial_KNN_matchUps()
print(Results)


# Time and space:
print('\n\n\n\n', '-'*40, '\n'*4)
da_T = make_xrArray(True)
Matchuper = XR_GDF_Pixels_MatchUps(da_T, gdf)

Results = Matchuper.get_time_space_KNN_matchUps(k=10, 
                                                dict_of_windows={'time_window': 2, 
                                                                 'time_unit': 'D'})

print(Results)


