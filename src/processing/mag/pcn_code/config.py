# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 12:52:37 2025

@author: richarj2
"""
import os
import calendar

BASE_DIR    = 'Y:/Processed_Data/GROUND/STAUNING/'

PRELIM_DIR  = os.path.join(BASE_DIR, 'prelim')
IAGA_DIR    = os.path.join(BASE_DIR, 'iaga')
DATA_DIR    = os.path.join(BASE_DIR, 'data')
ANALY_DIR   = os.path.join(BASE_DIR, 'analysis')
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


DIRECTORIES = {'prelim': PRELIM_DIR, 'data': DATA_DIR, 'iaga': IAGA_DIR, 'dist': DIST_DIR, 'analysis': ANALY_DIR, 'phi': FI_DIR, 'ab': AB_DIR, 'coeff': COEFF_DIR, 'out': OUT_DIR, 'staun_omni': STAUN_OMNI_DIR, 'updated_omni': UPDATED_OMNI_DIR, 'staun_phi': STAUN_PHI_DIR, 'recreated_phi': RECREATED_PHI_DIR, 'updated_phi': UPDATED_PHI_DIR, 'staun_proj': STAUN_PROJ_DIR}

for _, path in DIRECTORIES.items():

    os.makedirs(path, exist_ok=True)

import matplotlib.pyplot as plt
plt.style.use('dark_background')


PLOT_LABELS_SHORT = {'phi': r'$\varphi$ [$^\circ$]',    'phi_unc': r'$\sigma_\varphi$ [$^\circ$]',
                     'f': r'$\varphi$ [$^\circ$]',
                     'a': r'$\alpha$ [mV/m / nT]',      'a_unc': r'$\sigma_\alpha$ [mV/m / nT]',
                     'b': r'$\beta$ [nT]',              'b_unc': r'$\sigma_\beta$ [nT]'
                     }
PLOT_LABELS_LONG  = {'phi': r'direction ($\varphi$) [$^\circ$]',    'phi_unc': r'direction unc ($\sigma_\varphi$) [$^\circ$]',
                     'f': r'direction ($\varphi$) [$^\circ$]',
                     'a': r'slope ($\alpha$) [mV/m / nT]',          'a_unc': r'slope unc ($\sigma_\alpha$) [mV/m / nT]',
                     'b': r'interept ($\beta$) [nT]',               'b_unc': r'interept unc ($\sigma_\beta$) [nT]'
                     }

LIST_OF_MONTHS = [mon[:3] for mon in list(calendar.month_name)[1:]]
