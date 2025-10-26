# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""
from src.config import LUNA_MMS_DIR_FGM, LUNA_MMS_DIR_HPCA, MMS_DIR
from src.processing.mms.config import MMS_VARIABLES, MMS_VARIABLES_HPCA
from src.processing.mms.handling import process_mms_files
from src.processing.writing import resample_cdf_files

# %% Field

for year in range(2015,2026):
    process_mms_files(LUNA_MMS_DIR_FGM, MMS_DIR, MMS_VARIABLES, sample_intervals=('1min','5min'), year=year)



# %% Plasma

process_mms_files(LUNA_MMS_DIR_HPCA, MMS_DIR, MMS_VARIABLES_HPCA, sample_intervals=('spin','1min','5min'))


# %% Resample
import os
spin_directory = os.path.join(MMS_DIR, 'plasma', 'spin')
resample_cdf_files(spin_directory, sample_interval='1min', yearly_files=True)
resample_cdf_files(spin_directory, sample_interval='5min', yearly_files=True)


