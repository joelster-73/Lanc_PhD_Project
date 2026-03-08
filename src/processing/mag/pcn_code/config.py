# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 12:52:37 2025

@author: richarj2
"""
import os
import calendar

BASE_DIR    = 'Y:/Processed_Data/GROUND/STAUNING/'

DATA_DIR    = os.path.join(BASE_DIR, 'data')
DIST_DIR    = os.path.join(BASE_DIR, 'STEP1_getdist')
FI_DIR      = os.path.join(BASE_DIR, 'STEP2_fi')
AB_DIR      = os.path.join(BASE_DIR, 'STEP3_ab')
COEFF_DIR   = os.path.join(BASE_DIR, 'STEP4_coeff')
OUT_DIR     = os.path.join(BASE_DIR, 'output')

STAUN_OMNI_DIR = os.path.join(OUT_DIR, '1_use_stauning_omni')

UPDATED_OMNI_DIR = os.path.join(OUT_DIR, '1_use_updated_omni')

STAUN_PHI_DIR = os.path.join(STAUN_OMNI_DIR, '2_use_stauning_phi')

RECREATED_PHI_DIR = os.path.join(STAUN_OMNI_DIR, '2_use_recreated_phi')

STAUN_PROJ_DIR = os.path.join(STAUN_OMNI_DIR, '2_use_stauning_proj')

UPDATED_PHI_DIR = os.path.join(UPDATED_OMNI_DIR, '2_use_updated_phi')


DIRECTORIES = {'out': OUT_DIR, 'staun_omni': STAUN_OMNI_DIR, 'updated_omni': UPDATED_OMNI_DIR, 'staun_phi': STAUN_PHI_DIR, 'recreated_phi': RECREATED_PHI_DIR, 'updated_phi': UPDATED_PHI_DIR, 'staun_proj': STAUN_PROJ_DIR}

for _, path in DIRECTORIES.items():

    os.makedirs(path, exist_ok=True)

import matplotlib.pyplot as plt
plt.style.use('dark_background')


PLOT_LABELS_SHORT = {'f': r'$\varphi$ [$^\circ$]', 'a': r'$\alpha$ [mV/m / nT]', 'b': r'$\beta$ [nT]'}
PLOT_LABELS_LONG  = {'f': r'direction ($\varphi$) [$^\circ$]', 'a': r'slope ($\alpha$) [mV/m / nT]', 'b': r'interept ($\beta$) [nT]'}

LIST_OF_MONTHS = [mon[:3] for mon in list(calendar.month_name)[1:]]
