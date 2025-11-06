# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""
import os

from src.processing.themis.config import LUNA_THEMIS_DIRECTORIES, PROC_THEMIS_DIRECTORIES, THEMIS_VARIABLES_FGM, THEMIS_VARIABLES_STATE, THEMIS_VARIABLES_PEIM
from src.processing.writing import resample_cdf_files
from src.processing.themis.handling import process_themis_files, process_themis_position, process_themis_plasma, combine_spin_data, filter_spin_data

all_spacecraft = ('tha','thb','thc','thd','the')
plasma_spacecraft = ('thb','the')

# %% Position

for spacecraft in all_spacecraft:

    process_themis_position(spacecraft, LUNA_THEMIS_DIRECTORIES, PROC_THEMIS_DIRECTORIES, THEMIS_VARIABLES_STATE, sub_folders=True)

# %% Field

for spacecraft in all_spacecraft:

    if spacecraft in plasma_spacecraft:
        sample_intervals = ('raw','1min','5min')
    else:
        sample_intervals = ('1min','5min')

    process_themis_files(spacecraft, LUNA_THEMIS_DIRECTORIES, PROC_THEMIS_DIRECTORIES, THEMIS_VARIABLES_FGM, sub_folders=True, sample_intervals=sample_intervals)

# %% Plasma

for spacecraft in plasma_spacecraft:
    # thb for msh
    # the for sw
    process_themis_plasma(spacecraft, LUNA_THEMIS_DIRECTORIES, PROC_THEMIS_DIRECTORIES, THEMIS_VARIABLES_PEIM, sub_folders=True, sample_intervals='raw')

# %% Filter
for spacecraft in plasma_spacecraft:

    combine_spin_data(spacecraft, PROC_THEMIS_DIRECTORIES)

    if spacecraft=='thb':
        filter_spin_data(spacecraft, PROC_THEMIS_DIRECTORIES, region='sw')

    elif spacecraft=='the':
        filter_spin_data(spacecraft, PROC_THEMIS_DIRECTORIES, region='msh')


# %% Sample

for spacecraft in plasma_spacecraft:

    sc_dir = PROC_THEMIS_DIRECTORIES[spacecraft]

    if spacecraft=='thb':
        region_dir = os.path.join(sc_dir, 'sw', 'raw')

    elif spacecraft=='the':
        region_dir = os.path.join(sc_dir, 'msh', 'raw')

    else:
        continue

    resample_cdf_files(region_dir, sample_interval='1min', yearly_files=True)
    resample_cdf_files(region_dir, sample_interval='5min', yearly_files=True)

