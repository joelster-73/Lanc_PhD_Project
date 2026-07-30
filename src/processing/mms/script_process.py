# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""

from src.processing.mms.handling import process_mms_files, resample_mms_files
from src.processing.updating import update_plasma_data

print('REMOVE TEMP YEAR ARGUMENT')

# %% Field

process_mms_files('mms1', 'state', sample_intervals=('raw','1min','5min','15min'), year=2025)

process_mms_files('mms1', 'fgm', sample_intervals=('raw','1min','5min','15min'), year=2025)


# %% HPCA

process_mms_files('mms1', 'hpca', sample_intervals=('raw',), year=2025) # Keeps separate ion quantities for fpi

process_mms_files('mms1', 'hpca', sample_intervals=('spin','1min','5min','15min'), year=2025)

# %% FPI

process_mms_files('mms1', 'fpi', sample_intervals=('none',), year=2025)

# %% Update

# Using HPCA heavy ion densities
update_plasma_data('mms1', 'fgm', 'fpi', 'hpca', (), convert_fields=('V',), year=2025)

resample_mms_files('mms1', 'fpi', 'spin', sample_intervals=('1min','5min','15min'), year=2025)


# %% Resample-only

resample_mms_files('mms1', 'state', 'raw', sample_intervals=('1min','5min','15min'))
resample_mms_files('mms1', 'fgm',   'raw', new_grouping='monthly', sample_intervals=('1min','5min','15min'))
resample_mms_files('mms1', 'hpca', 'spin', sample_intervals=('1min','5min','15min'))
resample_mms_files('mms1', 'fpi',  'spin', sample_intervals=('1min','5min','15min'))
