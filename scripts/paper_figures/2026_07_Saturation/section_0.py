# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 15:37:29 2025

@author: richarj2
"""

from src.methods.saturation.merge_region_sc import merge_sc_in_region

# %% Merge

for sample_interval in ('1min','5min','15min'):

    merge_sc_in_region('sw', data_pop='plasma', sample_interval=sample_interval)

for sample_interval in ('1min','5min','15min'):

    merge_sc_in_region('msh', data_pop='plasma', sample_interval=sample_interval)


# MMS undercounts density in the solar wind (coarse energ/angle bins)
# Cluster undercounts density in the msh (count-rate/dead-time saturation)