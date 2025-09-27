# -*- coding: utf-8 -*-
"""
Created on Thu May  8 16:17:14 2025

@author: richarj2
"""
# src/config.py

from datetime import datetime

import os

# Define DATA_DIR relative to the module's location
MODULE_DIR             = os.path.dirname(os.path.abspath(__file__))  # Path of the current module
DATA_DIR               = os.path.join(MODULE_DIR, '..', 'data')
TEMP_DATA_DIR          = os.path.join(MODULE_DIR, '..', 'tempData')
FIGURES_DIR            = os.path.join(MODULE_DIR, '..', 'figures')

# Base Directories
PROCESSED_DATA_DIR     = f'{DATA_DIR}/Processed_Data'
CLUSTER_DIR            = f'{PROCESSED_DATA_DIR}/Cluster1'
WIND_DIR               = f'{PROCESSED_DATA_DIR}/WIND'
OMNI_DIR               = f'{PROCESSED_DATA_DIR}/OMNI'
THEMIS_DIR             = f'{PROCESSED_DATA_DIR}/THEMIS'
CFA_DIR                = f'{PROCESSED_DATA_DIR}/CFA'
HELSINKI_DIR           = f'{PROCESSED_DATA_DIR}/HELSINKI'
SHOCKS_DIR             = f'{PROCESSED_DATA_DIR}/SHOCKS'
MSH_DIR                = f'{PROCESSED_DATA_DIR}/MSH'
CROSSINGS_DIR          = f'{CLUSTER_DIR}/Crossings'

# LUNA Directories
LUNA_CLUS_DIR_SPIN     = 'Z:/spacecraft/cluster/c1/C1_CP_FGM_SPIN/'
LUNA_CLUS_DIR_5VPS     = 'Z:/spacecraft/cluster/c1/C1_CP_FGM_5VPS/'
LUNA_CLUS_DIR_HIAM     = 'Z:/spacecraft/cluster/c1/C1_CP_CIS-HIA_ONBOARD_MOMENTS/'
LUNA_CLUS_DIR_HIAQ     = 'Z:/spacecraft/cluster/c1/C1_CP_CIS-HIA_QUALITY/'
LUNA_CROS_DIR          = 'Z:/spacecraft/cluster/c1/C1_CT_AUX_GRMB/'

LUNA_OMNI_DIR          = 'Z:/omni/omni_min_def/'
LUNA_OMNI_DIR_5MIN     = 'Z:/omni/omni_5min_def/'
LUNA_WIND_DIR          = 'Z:/spacecraft/wind/mfi/mfi_h0/'
LUNA_THEMIS_DIR        = 'Z:/spacecraft/themis/'

# Processed Cluster Data
PROC_CLUS_DIR_SPIN     = f'{CLUSTER_DIR}/SPIN'
PROC_CLUS_DIR_5VPS     = f'{CLUSTER_DIR}/5VPS'
PROC_CLUS_DIR_FGM      = f'{PROC_CLUS_DIR_5VPS}/raw'
PROC_CLUS_DIR_FGM1     = f'{PROC_CLUS_DIR_5VPS}/1min'
PROC_CLUS_DIR_CIS      = f'{PROC_CLUS_DIR_SPIN}/plasma/raw'
PROC_CLUS_DIR_COM      = f'{PROC_CLUS_DIR_SPIN}/combined/raw'

PROC_CLUS_DIR_MSH      = f'{PROC_CLUS_DIR_SPIN}/msh'

PROC_CLUS_DIR_MSHS     = f'{PROC_CLUS_DIR_MSH}/raw'
PROC_CLUS_DIR_MSH1     = f'{PROC_CLUS_DIR_MSH}/1min'
PROC_CLUS_DIR_MSH5     = f'{PROC_CLUS_DIR_MSH}/5min'

# Combined Cluster OMNI Data in MSH
PROC_MSH_FIELD_DIR     = f'{MSH_DIR}/field_only'
PROC_MSH_PLASMA_DIR    = f'{MSH_DIR}/with_plasma'

PROC_MSH_FIELD_5MIN   = f'{PROC_MSH_FIELD_DIR}/5min'
PROC_MSH_PLASMA_1MIN  = f'{PROC_MSH_PLASMA_DIR}/1min'
PROC_MSH_PLASMA_5MIN  = f'{PROC_MSH_PLASMA_DIR}/5min'



# Crossings Data
GOOD_CROSSINGS         = f'{CROSSINGS_DIR}/good_time_windows.txt'
BAD_CROSSINGS          = f'{CROSSINGS_DIR}/bad_time_windows.txt'
COMPRESSIONS_DIR       = f'{CROSSINGS_DIR}/compression_ratios.npz'

# Processed OMNI Data
PROC_OMNI_DIR_1MIN     = f'{OMNI_DIR}/1min'
PROC_OMNI_DIR_5MIN     = f'{OMNI_DIR}/5min'

# Processed WIND Data
PROC_WIND_DIR_1MIN     = f'{WIND_DIR}/1_min/'
PROC_WIND_DIR_1HOUR    = f'{WIND_DIR}/1_hour/'
PROC_WIND_DIR_3SEC     = f'{WIND_DIR}/3_sec/'

# Processed THEMIS Data
PROC_THEMIS_DIR        = THEMIS_DIR

# Processed Shocks Data
PROC_SHOCKS_DIR        = SHOCKS_DIR

# Processed Bowshock Data
PROC_SCBS_DIR          = f'{PROCESSED_DATA_DIR}/Bowshock/'

R_E = 6370 # Cluster takes 1 earth radius to be 6370 km

# Define the date range to analyse data
C1_RANGE = (datetime(2001, 1, 1), datetime(2024, 1, 1))