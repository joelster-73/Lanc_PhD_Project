# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""

from src.processing.cluster.config import CLUSTER_PLASMA
from src.processing.cluster.handling import process_cluster_files, update_fgm_data, resample_cluster_files
from src.processing.updating import update_plasma_data

# %% Position

for sc in CLUSTER_PLASMA:

    process_cluster_files(sc, 'state', '5VPS')
    process_cluster_files(sc, 'state', 'SPIN', sample_intervals=('1min','5min','15min'))

# %% Field

for sc in CLUSTER_PLASMA:

    process_cluster_files(sc, 'fgm', '5VPS')
    process_cluster_files(sc, 'fgm', 'SPIN')

# %% Update_fgm

for sc in CLUSTER_PLASMA:

    update_fgm_data(sc, 'raw') # GSE to GSM

    resample_cluster_files(sc, 'fgm', 'spin', sample_intervals=('1min','5min','15min'))

# %% Plasma

for sc in CLUSTER_PLASMA:

    process_cluster_files(sc, 'hia', 'moments')

# %% Update_hia

for sc in CLUSTER_PLASMA:

    update_plasma_data(sc, 'fgm', 'hia', 'omni', ('sw','msh'), convert_fields=('V',), field_res='spin')

    for region in ('sw','msh'):
        resample_cluster_files(sc, region, 'spin', sample_intervals=('1min','5min','15min'))

# %% Resample-only

resample_cluster_files('c1', 'state', 'spin', sample_intervals=('1min','5min','15min'))
resample_cluster_files('c1', 'fgm', 'spin', sample_intervals=('1min','5min','15min'))
resample_cluster_files('c1', 'sw', 'spin', sample_intervals=('1min','5min','15min'))
resample_cluster_files('c1', 'msh', 'spin', sample_intervals=('1min','5min','15min'))