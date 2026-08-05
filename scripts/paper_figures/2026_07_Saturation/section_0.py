# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 15:37:29 2025

@author: richarj2
"""


# %% Filter

from src.methods.saturation.filter import filter_sc_region, filter_spacecraft
from src.processing.reading import import_processed_data

for resolution in ('1min','5min','15min'):

    print(f'{resolution} resolution.\n')

    df_omni = import_processed_data('omni', resolution=resolution)

    for region in ('sw','msh'):

        for spacecraft in filter_spacecraft.get(region):

            filter_sc_region(spacecraft, region, resolution=resolution, df_omni=df_omni, test_return=False)


# MMS undercounts density in the solar wind (coarse energy/angle bins)
# Cluster undercounts density in the msh (count-rate/dead-time saturation)


# %% Merge

import itertools as it
from src.methods.saturation.merge import merge_sc_in_region


for region, resolution in it.product(('sw','msh'),('1min','5min','15min')):

    print(f'{region} region | {resolution} resolution.\n')

    merge_sc_in_region(region, resolution=resolution)

