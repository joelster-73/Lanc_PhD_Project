# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:01:45 2025

@author: richarj2
"""

CLUSTER_SPACECRAFT = ('C1', 'C2', 'C3', 'C4')
MODES              = ('5VPS', 'SPIN')

# %% STATE

# STATE data is in the FGM files

STATE_VARIABLES_TEMPLATE = {
    'epoch' : 'time_tags__{sc}_CP_FGM_{mode}',      # epoch time in milliseconds (scalar)
    'r'     : 'sc_pos_xyz_gse__{sc}_CP_FGM_{mode}', # position in km from centre of Earth (vector)
}

CLUSTER_VARIABLES_STATE = {
    sc: {
        mode: {
            key: value.format(sc=sc, mode=mode)
            for key, value in STATE_VARIABLES_TEMPLATE.items()
        }
        for mode in MODES
    }
    for sc in CLUSTER_SPACECRAFT
}


# %% FGM

# STATE data is in the FGM files

FGM_VARIABLES_TEMPLATE = {
    'epoch' : 'time_tags__{sc}_CP_FGM_{mode}',      # epoch time in milliseconds (scalar)
    'B_avg' : 'B_mag__{sc}_CP_FGM_{mode}',          # fgm data in nT (magnitude)
    'B'     : 'B_vec_xyz_gse__{sc}_CP_FGM_{mode}',  # fgm data in nT (vector)
}

CLUSTER_VARIABLES_FGM = {
    sc: {
        mode: {
            key: value.format(sc=sc, mode=mode)
            for key, value in FGM_VARIABLES_TEMPLATE.items()
        }
        for mode in MODES
    }
    for sc in CLUSTER_SPACECRAFT
}

# %% HIA

HIA_VARIABLES_TEMPLATE = {
    'epoch' : 'time_tags__{sc}_CP_CIS-HIA_ONBOARD_MOMENTS',      # epoch time in milliseconds (scalar)
    'V'     : 'velocity_gse__{sc}_CP_CIS-HIA_ONBOARD_MOMENTS',   # plasma velocity in km/s (vector)
    'T_ion' : 'temperature__{sc}_CP_CIS-HIA_ONBOARD_MOMENTS',    # temperature (MK)
    'N_ion' : 'density__{sc}_CP_CIS-HIA_ONBOARD_MOMENTS',        # density (n/cc)
    'P_ion' : 'pressure__{sc}_CP_CIS-HIA_ONBOARD_MOMENTS',       # thermal pressure (nPa)
    'mode'  : 'cis_mode__{sc}_CP_CIS-HIA_ONBOARD_MOMENTS',       # instrument mode
}

CLUSTER_VARIABLES_HIA = {
    sc: {
        key: value.format(sc=sc)
        for key, value in HIA_VARIABLES_TEMPLATE.items()
    }
    for sc in CLUSTER_SPACECRAFT
}

# %% QUALITY

QUALITY_VARIABLES_TEMPLATE = {
    'epoch'   : 'time_tags__{sc}_CP_CIS-HIA_QUALITY',      # epoch time in milliseconds (scalar)
    'quality' : 'quality_MOM__{sc}_CP_CIS-HIA_QUALITY',    # quality = 3,4 is good
}

CLUSTER_VARIABLES_QUALITY = {
    sc: {
        key: value.format(sc=sc)
        for key, value in QUALITY_VARIABLES_TEMPLATE.items()
    }
    for sc in CLUSTER_SPACECRAFT
}

VARIABLES_DICT = {'fgm': CLUSTER_VARIABLES_FGM, 'hia': CLUSTER_VARIABLES_HIA, 'quality': CLUSTER_VARIABLES_QUALITY, 'state': CLUSTER_VARIABLES_STATE}