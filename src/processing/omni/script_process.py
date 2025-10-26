# -*- coding: utf-8 -*-
"""
Created on Mon May 19 23:02:24 2025

@author: joels
"""

# Process all the data from the CDF files and save to a new CDF file

from src.config import LUNA_OMNI_DIR, OMNI_DIR, LUNA_OMNI_DIR_5MIN, PROC_OMNI_DIR_5MIN

from src.processing.reading import import_processed_data

from src.processing.omni.config import omni_columns, omni_columns_5min
from src.processing.omni.handling import process_omni_files
from src.processing.omni.analysis import add_index_lag

# %% Definitive

process_omni_files(LUNA_OMNI_DIR, OMNI_DIR, omni_columns, ext='lst')

# %% Definitive_5min

process_omni_files(LUNA_OMNI_DIR_5MIN, OMNI_DIR, omni_columns_5min, ext='lst')

# %% Test

omni = import_processed_data(PROC_OMNI_DIR_5MIN)

# %% Lag

for sample_interval in ('1min','5min'):
    add_index_lag(OMNI_DIR, sample_interval)