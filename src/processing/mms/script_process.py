# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""

from src.processing.mms.handling import process_mms_files

# %% Field

process_mms_files('mms1', 'state', sample_intervals=('raw','1min','5min'))

process_mms_files('mms1', 'fgm', sample_intervals=('raw',))

#process_mms_files('mms1', 'fgm', sample_intervals=('raw','1min','5min')) # resample to 1min, 5min

# %% HPCA

process_mms_files('mms1', 'hpca', sample_intervals=('raw',)) # Keeps separate ion quantities

process_mms_files('mms1', 'hpca', sample_intervals=('spin','1min','5min'))

# %% FPI

process_mms_files('mms1', 'fpi', sample_intervals=('none',))

# %% Update

from src.processing.updating import resample_monthly_files, update_plasma_data

# Using HPCA heavy ion densities
update_plasma_data('mms1', 'fgm', 'fpi', 'hpca', ('sw','msh'), convert_fields=('V',))

resample_monthly_files('mms1', 'sw', 'spin', sample_intervals=('1min','5min'))
resample_monthly_files('mms1', 'msh', 'spin', sample_intervals=('1min','5min'))