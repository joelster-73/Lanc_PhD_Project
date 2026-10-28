# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 15:37:29 2025

@author: richarj2
"""


# %% Filter

from src.methods.saturation.filter import filter_sc_region, filter_spacecraft
from src.processing.reading import import_processed_omni
from src.processing.handling import create_log_file
from src.config import get_proc_directory

print('REMINDER: check if any files aren\'t the same data structure - suggests data folders contain files that are outdated.')

resolutions = ('1min', '5min') # 15min
regions     = ('sw', 'msh') # msh20

for resolution in resolutions:

    df_omni = import_processed_omni(resolution=resolution)

    for region in regions: # excluding msh20 for now

        direc = get_proc_directory(region, dtype='plasma', resolution=resolution, create=False, empty=False)
        create_log_file('Filtering_Information.txt', overwrite=True, direc=direc)

        for spacecraft in filter_spacecraft.get(region):

            filter_sc_region(spacecraft, region, data_pop='plasma', resolution=resolution, df_omni=df_omni, test_return=False)


# MMS undercounts density in the solar wind (coarse energy/angle bins)
# Cluster undercounts density in the msh (count-rate/dead-time saturation)


# %% Merge

import itertools as it
from src.methods.saturation.merge import merge_sc_in_region


for region, resolution in it.product(regions,resolutions):

    print(f'{region} region | {resolution} resolution.\n')

    merge_sc_in_region(region, resolution=resolution)

