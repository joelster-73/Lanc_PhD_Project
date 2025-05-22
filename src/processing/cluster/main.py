# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""

from ...src.config import LUNA_CLUS_DIR, PROC_CLUS_DIR, SAMPLE_SIZE, LUNA_CLUS_DIR_5VPS
from .config import CLUSTER_VARIABLES, CLUSTER_VARIABLES_5VPS


# Process all the data from the CDF files and save to a new CDF file
from .handling import process_cluster_files

process_cluster_files(LUNA_CLUS_DIR, PROC_CLUS_DIR, CLUSTER_VARIABLES, SAMPLE_SIZE)

process_cluster_files(LUNA_CLUS_DIR_5VPS, PROC_CLUS_DIR, CLUSTER_VARIABLES_5VPS, SAMPLE_SIZE,
                      sub_folders=True, year=2001)
