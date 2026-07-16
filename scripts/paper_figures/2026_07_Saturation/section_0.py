# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 15:37:29 2025

@author: richarj2
"""

import itertools as it
from src.methods.saturation.merge_region_sc import merge_sc_in_region
from src.processing.reading import import_processed_data
from src.methods.saturation.sc_delay_time import shift_sc_to_bs

# %% Merge

for sample_interval in ('1min','5min','15min'):

    merge_sc_in_region('sw', data_pop='plasma', sample_interval=sample_interval)

for sample_interval in ('1min','5min','15min'):

    merge_sc_in_region('msh', data_pop='plasma', sample_interval=sample_interval)


# %% Shifting

for region, sample_interval in it.product(('sw','msh'),('1min','5min','15min')):

    df_sc = import_processed_data(region, dtype='plasma', resolution=sample_interval, file_name=f'{region}_times_combined')
    _ = shift_sc_to_bs(df_sc, sample_interval, region, write_to_file=True)

# %% Plots

# plots wanted
# - one driver and response, different lags
# - one driver and response, different averages
# - one driver and multiple responses
# - one response and multiple drivers

# make proper list and then change functions to reflect this

# why do some propagation times come out as nan - is this just because of my threshold of 60 mins or is one of the params NaN. How can I decouple it from OMNI.
# could store as column then slice the indepdnent dataframe by the different shift values?