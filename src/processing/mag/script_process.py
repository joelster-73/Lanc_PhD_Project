# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:42:09 2025

@author: richarj2
"""


# %% Lagged_indices
from src.processing.mag.indices import build_lagged_indices

for sample_interval in ('1min','5min'): # for OMNI's indices
    build_lagged_indices(sample_interval)


# %% Add magnetometer data
from src.processing.mag.handling import process_supermag_data


THL = process_supermag_data('THL')