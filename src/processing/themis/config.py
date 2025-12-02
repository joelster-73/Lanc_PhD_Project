# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:48:07 2025

@author: richarj2
"""
# config.py

THEMIS_SPACECRAFT = ('tha', 'thb', 'thc', 'thd', 'the')

STATE_VARIABLES_TEMPLATE = { # 1 minute resolution
    'time': '{sc}_state_time',
    'r':    '{sc}_pos_gse'
}

THEMIS_VARIABLES_STATE = {
    sc: {key: value.format(sc=sc) for key, value in STATE_VARIABLES_TEMPLATE.items()}
    for sc in THEMIS_SPACECRAFT
}


FGM_VARIABLES_TEMPLATE = {
    'time':  '{sc}_{suffix}_time',
    'B_avg': '{sc}_{suffix}_btotal',
    'B_GSE': '{sc}_{suffix}_gse',
    'B_GSM': '{sc}_{suffix}_gsm',
    'quality': '{sc}_fgm_{suffix}_quality'
}

FGM_SUFFIXES = ('fgs','fgl','fgh','fge')

THEMIS_VARIABLES_FGM = {
    sc: {
        suffix: {key: value.format(sc=sc, suffix=suffix) for key, value in FGM_VARIABLES_TEMPLATE.items()}
        for suffix in FGM_SUFFIXES
    }
    for sc in THEMIS_SPACECRAFT
}


PLASMA_VARIABLES_TEMPLATE = {
    'time':         '{sc}_peim_time',
    'time_flag':    '{sc}_iesa_solarwind_flag_time',
    'flag':         '{sc}_iesa_solarwind_flag',     # 0 = not in solar wind, 1 = in solar wind
    'quality':      '{sc}_peim_data_quality',       # 0 = good, >1 = bad
    'N_ion':        '{sc}_peim_density',            # n/cc
    'P_ion':        '{sc}_peim_ptot',               # eV/cc
    'T_vec':        '{sc}_peim_t3_mag',             # Field aligned temperature (scalar = mean of components), eV
    'V_GSE':        '{sc}_peim_velocity_gse',       # km/s
    'V_GSM':        '{sc}_peim_velocity_gsm'

}

THEMIS_VARIABLES_PEIM = {
    sc: {key: value.format(sc=sc) for key, value in PLASMA_VARIABLES_TEMPLATE.items()}
    for sc in THEMIS_SPACECRAFT
}
