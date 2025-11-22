# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""
import os
from src.config import LUNA_CLUS_DIR_SPIN, PROC_CLUS_DIR_SPIN, LUNA_CLUS_DIR_HIAM, LUNA_CLUS_DIR_HIAQ, PROC_CLUS_DIR_5VPS, LUNA_CLUS_DIR_5VPS, PROC_CLUS_DIR_FGM, PROC_CLUS_DIR_MSH, PROC_CLUS_DIR_SW

from src.processing.writing import resample_cdf_files
from src.processing.cluster.config import CLUSTER_VARIABLES_SPIN, CLUSTER_VARIABLES_HIA, CLUSTER_VARIABLES_HIA_QUALITY, CLUSTER_VARIABLES_5VPS
from src.processing.cluster.handling import process_cluster_files, combine_spin_data, filter_spin_data


# %%  Process_5VPS

process_cluster_files(LUNA_CLUS_DIR_5VPS, PROC_CLUS_DIR_5VPS, CLUSTER_VARIABLES_5VPS)

#process_cluster_files(LUNA_CLUS_DIR_5VPS, PROC_CLUS_DIR_5VPS, CLUSTER_VARIABLES_5VPS, sub_folders=True, year=2001)

# %% Average_FGM

for sample in ('1min','5min'):
    resample_cdf_files(PROC_CLUS_DIR_FGM, sample_interval=sample, yearly_files=True)

# %% Process_spin
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*DatetimeProperties.to_pydatetime.*')

process_cluster_files(LUNA_CLUS_DIR_SPIN, PROC_CLUS_DIR_SPIN, CLUSTER_VARIABLES_SPIN, sample_interval='none')

process_cluster_files(LUNA_CLUS_DIR_HIAM, PROC_CLUS_DIR_SPIN, CLUSTER_VARIABLES_HIA, sub_folders=True, sample_interval='none', quality_directory=LUNA_CLUS_DIR_HIAQ, quality_variables=CLUSTER_VARIABLES_HIA_QUALITY)

# %% Combine_spin data

combine_spin_data(PROC_CLUS_DIR_SPIN, PROC_CLUS_DIR_FGM)

# %% MSH_data

filter_spin_data(PROC_CLUS_DIR_SPIN, region='msh')

raw_dir = os.path.join(PROC_CLUS_DIR_MSH,'raw')
for sample in ('1min','5min'):
    resample_cdf_files(raw_dir, sample_interval=sample, yearly_files=True)


# %% SW_data

filter_spin_data(PROC_CLUS_DIR_SPIN, region='sw')

raw_dir = os.path.join(PROC_CLUS_DIR_SW,'raw')
for sample in ('1min','5min'):
    resample_cdf_files(raw_dir, sample_interval=sample, yearly_files=True)

