# -*- coding: utf-8 -*-
'''
Created on Sat Feb 28 14:22:13 2026

@author: richarj2
'''

import os
import numpy as np
import pandas as pd

import scipy.io




BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, 'STEP1_getdist')
FI_DIR   = os.path.join(BASE_DIR, 'STEP2_fi')
OUT_DIR  = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)


def phi_import(var, source='npz'):
    """
    Load Phi data from .npz or .mat files.
    var: 'Phi_2d' or 'Phi_year'
    year: integer year required if loading Phi_year from MATLAB
    source: 'npz' -> loads from OUT_DIR
            'mat' -> loads from FI_DIR
    Returns: NumPy array
    """
    print(f'\nImporting {var} for {source}.\n')
    if source.lower() == 'npz':
        path = os.path.join(OUT_DIR, f'Phi_{var}.npz')
        data = np.load(path)
        if isinstance(var,int):
            var = 'Phi'
        else:
            var = f'Phi_{var}'
        struc = data[var]

    elif source.lower() == 'mat':
        if isinstance(var,int):
            var = f'F_{var}'
        else:
            var = f'Fi_{var}'
        path = os.path.join(FI_DIR, f'{var}.mat')
        mat = scipy.io.loadmat(path)
        struc = mat[var]

    else:
        raise ValueError('source must be "npz" or "mat"')

    print('Shape:',struc.shape)
    return struc

def phi_analyse(source='npz'):
    """
    Analyse Phi data and produce Table 6 style optimum correlations.
    Automatically loads Phi_year from .npz or MATLAB files.
    """
    # Load Phi_year arrays for 1997-2009
    Phi_list = []
    for year in range(1997, 2010):
        try:
            Phi_year = phi_import(var=year, source=source)
            Phi_list.append(Phi_year)
        except FileNotFoundError:
            print(f'Warning: Phi_{year} not found, skipping.')

    if not Phi_list:
        raise ValueError('No Phi_year data available.')

    phi_stack = np.stack(Phi_list, axis=2)
    phi_avr = np.nanmean(phi_stack, axis=2)

    SEASONS = {
        'WINTER': [10,11,0,1],
        'EQUINOX': [2,3,8,9],
        'SUMMER': [4,5,6,7]
    }
    UT_INTERVALS = {
        '01-07': list(range(12, 84)),
        '07-13': list(range(84, 156)),
        '13-19': list(range(156, 228)),
        '19-01': list(range(228, 288)) + list(range(0,12))
    }

    rows = []
    for season_name, months in SEASONS.items():
        for ut_name, bins in UT_INTERVALS.items():
            vals = phi_avr[months,:][:, bins]
            vals_mean = np.nanmean(vals, axis=0)
            oc, od = parabolic_fit_max(vals_mean)
            rows.append([season_name, ut_name, oc, od])

    df_table6 = pd.DataFrame(rows, columns=['Season','UT','OptCorr','OptDelay_min'])
    print('\nOptimum Correlation Table (Table 6 style):')
    print(df_table6)
    return df_table6

def parabolic_fit_max(y):

    idx = np.nanargmax(y)
    if idx < 2:
        idx_range = slice(0,5)
    elif idx > len(y)-3:
        idx_range = slice(len(y)-5, len(y))
    else:
        idx_range = slice(idx-2, idx+3)

    x_fit = np.arange(5)
    y_fit = y[idx_range]
    coeff = np.polyfit(x_fit, y_fit, 2)

    x_vertex = -coeff[1]/(2*coeff[0])
    y_vertex = np.polyval(coeff, x_vertex)

    return y_vertex, (idx-2 + x_vertex)*5


