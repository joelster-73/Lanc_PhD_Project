# -*- coding: utf-8 -*-
"""
Created on Mon May 19 23:02:24 2025

@author: joels
"""

# Process all the data from the CDF files and save to a new CDF file

from src.processing.reading import import_processed_data

from src.processing.omni.handling import process_omni_files

# %% Definitive

process_omni_files(resolution='1min', ext='lst')

# %% Definitive_5min

process_omni_files(resolution='5min', ext='lst')

# %% Test

omni = import_processed_data('omni', resolution='5min')

