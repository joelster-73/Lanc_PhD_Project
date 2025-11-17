# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 15:37:29 2025

@author: richarj2
"""

from src.methods.magnetosheath_saturation.merge_region_omni import merge_sc_in_region

for sample_interval in ('1min','5min'):
    merge_sc_in_region('msh', data_pop='with_plasma', sample_interval=sample_interval)


for sample_interval in ('1min','5min'):
    merge_sc_in_region('sw', data_pop='with_plasma', sample_interval=sample_interval)



