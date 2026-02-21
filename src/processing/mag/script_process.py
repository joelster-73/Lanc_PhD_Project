# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:42:09 2025

@author: richarj2
"""

from src.processing.mag.config import PC_STATIONS

# %% Use_API

from src.processing.mag.supermag import download_supermag_data

for station in PC_STATIONS:
    if station=='THL':
        continue
    download_supermag_data(station)

# %% NetCDF_Files
from src.processing.mag.supermag import process_supermag_data

for station in PC_STATIONS:
    process_supermag_data(station)

# %% Convert_to_GSE
from src.processing.mag.supermag import convert_supermag_gse

for station in PC_STATIONS:
    convert_supermag_gse(station)

# %% Convert_to_GSM
from src.processing.mag.supermag import convert_supermag_gsm

for station in PC_STATIONS:
    convert_supermag_gsm(station)

# %% Lagged_indices
from src.processing.mag.indices import build_lagged_indices

# for OMNI's indices
for sample_interval in ('1min','5min'):
    # if sample_interval=='1min':
    #     continue
    build_lagged_indices(sample_interval, PC_stations=PC_STATIONS, to_include=('mag','gsm'))
