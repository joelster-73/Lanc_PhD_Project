# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:01:45 2025

@author: richarj2
"""

CLUSTER_VARIABLES_SPIN = {
    'epoch' : 'time_tags__C1_CP_FGM_SPIN',      # epoch time in milliseconds (scalar)
    'r'     : 'sc_pos_xyz_gse__C1_CP_FGM_SPIN', # position in km from centre of Earth (vector)
    'B_avg' : 'B_mag__C1_CP_FGM_SPIN',          # fgm data in nT (magnitude)
    'B'     : 'B_vec_xyz_gse__C1_CP_FGM_SPIN',  # fgm data in nT (vector)
}

CLUSTER_VARIABLES_5VPS = {
    'epoch' : 'time_tags__C1_CP_FGM_5VPS',      # epoch time in milliseconds (scalar)
    'r'     : 'sc_pos_xyz_gse__C1_CP_FGM_5VPS', # position in km from centre of Earth (vector)
    'B_avg' : 'B_mag__C1_CP_FGM_5VPS',          # fgm data in nT (magnitude)
    'B'     : 'B_vec_xyz_gse__C1_CP_FGM_5VPS',  # fgm data in nT (vector)
}

CLUSTER_VARIABLES_5VPS_POS = {
    'epoch' : 'epoch',
    'r_x'   : 'r_x_GSE',
    'r_y'   : 'r_y_GSE',
    'r_z'   : 'r_z_GSE',
}