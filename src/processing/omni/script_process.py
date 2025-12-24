# -*- coding: utf-8 -*-
"""
Created on Mon May 19 23:02:24 2025

@author: joels
"""

# Process all the data from the CDF files and save to a new CDF file

from src.processing.omni.handling import process_omni_files

# %% Definitive

process_omni_files(resolution='1min', ext='lst')

# %% Definitive_5min

process_omni_files(resolution='5min', ext='lst')

# %% Test
from src.processing.reading import import_processed_data


omni = import_processed_data('omni', resolution='5min', year=2024)

omni_subset = omni.loc[(omni.index.month==5)&(omni.index.day==10)]

from src.coordinates.magnetic import convert_GSE_to_GSM

result = convert_GSE_to_GSM(omni_subset, 'B', time_col='index')
