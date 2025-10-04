# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:01:45 2025

@author: richarj2
"""

MMS_VARIABLES = {
    # Vectors are 4 columns - magntiude and components
    # Field data is 8/16 Hz
    'epoch'     : 'Epoch',
    'B_gse'     : 'mms1_fgm_b_gse_srvy_l2',
    'B_gsm'     : 'mms1_fgm_b_gsm_srvy_l2',
    'B_flag'    : 'mms1_fgm_flag_srvy_l2',
    # State data is every 30s
    'epoch_pos' : 'Epoch_state',
    'r_gse'     : 'mms1_fgm_r_gse_srvy_l2', # km
}
