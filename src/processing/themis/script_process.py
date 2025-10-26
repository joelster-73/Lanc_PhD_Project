# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""
import os

from src.processing.themis.config import LUNA_THEMIS_DIRECTORIES, PROC_THEMIS_DIRECTORIES, THEMIS_VARIABLES_FGM, THEMIS_VARIABLES_STATE, THEMIS_VARIABLES_PEIM
from src.processing.writing import resample_cdf_files
from src.processing.themis.handling import process_themis_files, process_themis_plasma, combine_spin_data, filter_spin_data

all_spacecraft = ('tha','thb','thc','thd','the')

# %% Field
for spacecraft in all_spacecraft:

    process_themis_files(spacecraft, LUNA_THEMIS_DIRECTORIES, PROC_THEMIS_DIRECTORIES, THEMIS_VARIABLES_FGM, THEMIS_VARIABLES_STATE, sub_folders=True)

# %% Plasma

for spacecraft in all_spacecraft:

    if spacecraft!='the':
        continue

    process_themis_plasma(spacecraft, LUNA_THEMIS_DIRECTORIES, PROC_THEMIS_DIRECTORIES, THEMIS_VARIABLES_PEIM, sub_folders=True)

# %% Combine
for spacecraft in all_spacecraft:

    if spacecraft!='the':
        continue

    combine_spin_data(spacecraft, PROC_THEMIS_DIRECTORIES)

# %% Filter
for spacecraft in all_spacecraft:

    if spacecraft!='the':
        continue

    filter_spin_data(spacecraft, PROC_THEMIS_DIRECTORIES)


# %% Sample

for spacecraft in all_spacecraft:

    if spacecraft!='the':
        continue

    sc_dir = PROC_THEMIS_DIRECTORIES[spacecraft]
    region_dir = os.path.join(sc_dir, 'msh', 'raw')

    resample_cdf_files(region_dir, sample_interval='1min', yearly_files=True)
    resample_cdf_files(region_dir, sample_interval='5min', yearly_files=True)

