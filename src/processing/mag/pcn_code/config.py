# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 12:52:37 2025

@author: richarj2
"""
import os
import calendar
import matplotlib.pyplot as plt

OMNI_DIR    = 'Z:/omni/omni_min_def'
BASE_DIR    = 'Y:/Processed_Data/GROUND/STAUNING/'

IAGA_DIR    = os.path.join(BASE_DIR, 'iaga')
DATA_DIR    = os.path.join(BASE_DIR, 'data')
ANALY_DIR   = os.path.join(BASE_DIR, 'analysis')
DIST_DIR    = os.path.join(BASE_DIR, 'STEP1_getdist')
FI_DIR      = os.path.join(BASE_DIR, 'STEP2_fi')
AB_DIR      = os.path.join(BASE_DIR, 'STEP3_ab')
COEFF_DIR   = os.path.join(BASE_DIR, 'STEP4_coeff')
INP_DIR     = os.path.join(BASE_DIR, 'input')
OUT_DIR     = os.path.join(BASE_DIR, 'output')

PRELIM_DIR        = os.path.join(ANALY_DIR,        'prelim')
STAUN_OMNI_DIR    = os.path.join(OUT_DIR,          '1_use_stauning_omni')
UPDATED_OMNI_DIR  = os.path.join(OUT_DIR,          '1_use_updated_omni')
STAUN_PHI_DIR     = os.path.join(STAUN_OMNI_DIR,   '2_use_stauning_phi')
RECREATED_PHI_DIR = os.path.join(STAUN_OMNI_DIR,   '2_use_recreated_phi')
STAUN_PROJ_DIR    = os.path.join(STAUN_OMNI_DIR,   '2_use_stauning_proj')
UPDATED_PHI_DIR   = os.path.join(UPDATED_OMNI_DIR, '2_use_updated_phi')

DIRECTORIES = {'omni': OMNI_DIR, 'data': DATA_DIR, 'iaga': IAGA_DIR, 'dist': DIST_DIR, 'analysis': ANALY_DIR, 'prelim': PRELIM_DIR, 'phi': FI_DIR, 'ab': AB_DIR, 'coeff': COEFF_DIR, 'in': INP_DIR, 'out': OUT_DIR, 'staun_omni': STAUN_OMNI_DIR, 'updated_omni': UPDATED_OMNI_DIR, 'staun_phi': STAUN_PHI_DIR, 'recreated_phi': RECREATED_PHI_DIR, 'updated_phi': UPDATED_PHI_DIR, 'staun_proj': STAUN_PROJ_DIR}

for _, path in DIRECTORIES.items():
    os.makedirs(path, exist_ok=True)

dark_mode = False
if dark_mode:
    plt.style.use('dark_background')

PLOT_RANGE_CBAR   = {'phi': (8,80), 'f': (8,80), 'a': (0,80), 'b': (-45,5), 'covar': (-30,0)}

phi_unit = r'$^\circ$'
a_unit   = r'nT / (mV m$^{-1}$)'
b_unit   = 'nT'
ab_unit  = r'nT$^2$ / (mV m$^{-1}$)'

PLOT_LABELS_SHORT = {'f':           f'$\\varphi$ [{phi_unit}]',
                     'phi':         f'$\\varphi$ [{phi_unit}]',
                     'phi_unc':     f'$\\sigma_\\varphi$ [{phi_unit}]',
                     'a':           f'$\\alpha$ [{a_unit}]',
                     'a_unc':       f'$\\sigma_\\alpha$ [{a_unit}]',
                     'b':           f'$-\\beta$ [{b_unit}]',
                     'b_unc':       f'$\\sigma_\\beta$ [{b_unit}]',
                     'covar':       f'$-\\sigma_{{\\alpha\\beta}}$ [{ab_unit}]',
                     'gamma':       f'$\\gamma$ [{phi_unit}]',
                     }
PLOT_LABELS_LONG  = {'f':           f'direction ($\\varphi$) [{phi_unit}]',
                     'phi':         f'direction ($\\varphi$) [{phi_unit}]',
                     'phi_unc':     f'direction unc ($\\sigma_\\varphi$) [{phi_unit}]',
                     'a':           f'slope ($\\alpha$) [{a_unit}]',
                     'a_unc':       f'slope unc ($\\sigma_\\alpha$) [{a_unit}]',
                     'b':           f'intercept ($-\\beta$) [{b_unit}]',
                     'b_unc':       f'intercept unc ($\\sigma_\\beta$) [{b_unit}]',
                     'covar':       f'covariance ($-\\sigma_{{\\alpha\\beta}}$) [{ab_unit}]',
                     'gamma':       f'geomagnetic frame direction ($\\gamma$) [{phi_unit}]',
                     }

PLOT_LABELS_SCALED  = {'a_unc':       r'$\sigma_\alpha / \alpha$ [%]',
                       'b_unc':       r'$\sigma_\beta / \beta$ [%]',
                       'covar':       r'$\sigma_{\alpha\beta} / (\alpha\beta)$ [%]',
                       }

LIST_OF_MONTHS = [mon[:3] for mon in list(calendar.month_name)[1:]]
