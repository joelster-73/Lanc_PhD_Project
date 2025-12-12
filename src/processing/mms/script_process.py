# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""

from src.processing.mms.handling import process_mms_files

# %% Field

for year in range(2015,2026):
    #process_mms_files('mms1', 'fgm', sample_intervals=('1min','5min'), year=year)
    process_mms_files('mms1', 'fgm', sample_intervals=('raw',), year=year)


# %% HPCA

process_mms_files('mms1', 'hpca', sample_intervals=('spin','1min','5min'))

# %% FPI


# Processing fgm to raw first
# Then import and combine with fpi data

process_mms_files('mms1', 'fpi', sample_intervals='none', time_col='epoch')



# %%

from src.processing.reading import import_processed_data

test = import_processed_data('mms1', dtype='fpi', resolution='raw', year=2017)