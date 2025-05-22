# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:48:07 2025

@author: richarj2
"""
# config.py

from src.config import LUNA_THEMIS_DIR, PROC_THEMIS_DIR

THEMIS_SPACECRAFT = ['tha', 'thb', 'thc', 'thd', 'the']

LUNA_THEMIS_DIRECTORIES = {sc: f'{LUNA_THEMIS_DIR}/{sc}/' for sc in THEMIS_SPACECRAFT}

PROC_THEMIS_DIRECTORIES = {sc: f'{PROC_THEMIS_DIR}/{sc}/' for sc in THEMIS_SPACECRAFT}


STATE_VARIABLES_TEMPLATE = {
    'time': '{sc}_state_time',
    'r': '{sc}_pos_gse'
}

THEMIS_VARIABLES_STATE = {
    sc: {key: value.format(sc=sc) for key, value in STATE_VARIABLES_TEMPLATE.items()}
    for sc in THEMIS_SPACECRAFT
}


FGM_VARIABLES_TEMPLATE = {
    'time': '{sc}_{suffix}_time',
    'B_avg': '{sc}_{suffix}_btotal',
    'B_GSE': '{sc}_{suffix}_gse',
    'B_GSM': '{sc}_{suffix}_gsm'
}

FGM_SUFFIXES = ['fgh', 'fgl', 'fge', 'fgs']

THEMIS_VARIABLES_FGM = {
    sc: {
        suffix: {key: value.format(sc=sc, suffix=suffix) for key, value in FGM_VARIABLES_TEMPLATE.items()}
        for suffix in FGM_SUFFIXES
    }
    for sc in THEMIS_SPACECRAFT
}

