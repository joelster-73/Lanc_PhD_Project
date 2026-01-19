# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""

from src.processing.cluster.handling import process_cluster_files, update_fgm_data, resample_cluster_files
from src.processing.updating import update_plasma_data

# %% Field

process_cluster_files('c1', 'fgm', '5VPS')
process_cluster_files('c1', 'fgm', 'SPIN')

# %% Position

process_cluster_files('c1', 'state', '5VPS')
process_cluster_files('c1', 'state', 'SPIN')

resample_cluster_files('c1', 'state', 'spin', sample_intervals=('1min','5min'))

# %% Update_fgm

update_fgm_data('c1', 'raw')

resample_cluster_files('c1', 'fgm', 'spin', sample_intervals=('1min','5min'))

# %% Plasma

process_cluster_files('c1', 'hia', 'moments')

# %% Update_hia

update_plasma_data('c1', 'fgm', 'hia', 'omni', ('sw','msh'), convert_fields=('V',), field_res='spin')

resample_cluster_files('c1', 'sw', 'spin', sample_intervals=('1min','5min'))
resample_cluster_files('c1', 'msh', 'spin', sample_intervals=('1min','5min'))


