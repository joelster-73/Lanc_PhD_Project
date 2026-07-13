# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 15:37:29 2025

@author: richarj2
"""

from src.methods.magnetosheath_saturation.merge_region_sc import merge_sc_in_region
import itertools as it
from src.processing.reading import import_processed_data
from src.methods.magnetosheath_saturation.sc_delay_time import shift_sc_to_bs

# %% Merge

for sample_interval in ('1min','5min','15min'):

    merge_sc_in_region('sw', data_pop='plasma', sample_interval=sample_interval)


for sample_interval in ('1min','5min','15min'):

    merge_sc_in_region('msh', data_pop='plasma', sample_interval=sample_interval)


# %% Shifting

for region, sample_interval in it.product(('sw','msh'),('1min','5min','15min')):


    df_sc = import_processed_data(region, dtype='plasma', resolution=sample_interval, file_name=f'{region}_times_combined')
    _ = shift_sc_to_bs(df_sc, sample_interval, region, write_to_file=True)


# %% TEMP

for region in ('sw','msh'):

    merge_sc_in_region(region, data_pop='plasma', sample_interval='15min')


    # this just shifts to the BSN to create essentially OMNI but using Cluster, THEMIS, MMS
    # the old code had the time lag built into the indices files
    # that's been replaced, so instead the functions in section 1 onwards need to instead compare data with a number of rows later
    # check how to update this and then say this in the github push
    # also check the timestamps on all spacecraft files incase a bug was encountered


    df_sc = import_processed_data(region, dtype='plasma', resolution='15min', file_name=f'{region}_times_combined')
    _ = shift_sc_to_bs(df_sc, sample_interval, region, write_to_file=True)