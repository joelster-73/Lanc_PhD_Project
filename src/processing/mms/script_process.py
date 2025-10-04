# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""

from src.config import LUNA_MMS_DIR, MMS_DIR
from src.processing.mms.config import MMS_VARIABLES
from src.processing.mms.handling import process_mms_files


# %% 1min and 5min
for year in range(2015,2026):
    process_mms_files(LUNA_MMS_DIR, MMS_DIR, MMS_VARIABLES, sample_intervals=('1min','5min'), year=year)

# %%
process_mms_files(LUNA_MMS_DIR, MMS_DIR, MMS_VARIABLES, sample_intervals=('1min','5min'), year=2025)


