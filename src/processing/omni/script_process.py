# -*- coding: utf-8 -*-
"""
Created on Mon May 19 23:02:24 2025

@author: joels
"""

# Process all the data from the CDF files and save to a new CDF file

from src.processing.omni.handling import process_omni_files, resample_omni_files

print('DROP TEMP YEAR ARGUMENT')

# %% Definitive

process_omni_files(resolution='1min', ext='lst', year=2025)

# %% Definitive_5min

process_omni_files(resolution='5min', ext='lst', year=2025)

# %% Resample

resample_omni_files(raw_res='1min', sample_intervals=('15min',))