# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""

from src.config import LUNA_CLUS_DIR_SPIN, PROC_CLUS_DIR_SPIN, LUNA_CLUS_DIR_HIAM, LUNA_CLUS_DIR_HIAQ, PROC_CLUS_DIR_5VPS, LUNA_CLUS_DIR_5VPS, PROC_CLUS_DIR_FGM

from src.processing.cluster.config import CLUSTER_VARIABLES_SPIN, CLUSTER_VARIABLES_HIA, CLUSTER_VARIABLES_HIA_QUALITY, CLUSTER_VARIABLES_5VPS

from src.processing.cluster.handling import process_cluster_files, combine_spin_data, filter_spin_data


# %%  Process_5VPS

process_cluster_files(LUNA_CLUS_DIR_5VPS, PROC_CLUS_DIR_5VPS, CLUSTER_VARIABLES_5VPS)

process_cluster_files(LUNA_CLUS_DIR_5VPS, PROC_CLUS_DIR_5VPS, CLUSTER_VARIABLES_5VPS,
                      sub_folders=True, year=2001)

# %% Process_spin_fgm

process_cluster_files(LUNA_CLUS_DIR_SPIN, PROC_CLUS_DIR_SPIN, CLUSTER_VARIABLES_SPIN, sample_interval='None')

# %% Process_spin_plasma

for year in range(2001,2022):
    process_cluster_files(LUNA_CLUS_DIR_HIAM, PROC_CLUS_DIR_SPIN, CLUSTER_VARIABLES_HIA, sub_folders=True, sample_interval='None', quality_directory=LUNA_CLUS_DIR_HIAQ, quality_variables=CLUSTER_VARIABLES_HIA_QUALITY, year=year)

# %% Combine_spin
for year in range(2001,2022):
    combine_spin_data(PROC_CLUS_DIR_SPIN, PROC_CLUS_DIR_FGM, year=year)

# %% MSH_filter
for year in range(2001,2022):
    filter_spin_data(PROC_CLUS_DIR_SPIN, region='msh', year=year)

# %%

