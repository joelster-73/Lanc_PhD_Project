# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:44:28 2026

@author: richarj2
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .stauning_imports import import_phi, import_ab, import_hproj, import_dist, import_data, import_er
from .stauning_plots import print_coeffs_monthly_ut

def compare_phi(year, source='staun_omni'):

    if source.lower() == 'updated_omni':
        warnings.warn('Cannot compare npz as it fixes bug.')
        return

    phi_np  = import_phi(year, source=source)['phi'].ravel()
    phi_mat = import_phi(year, source='original')['phi'].ravel()

    if phi_np.shape!=phi_mat.shape:
        print(f'Shapes do not match\n   np: {phi_np.shape}\n  mat: {phi_mat.shape}')
        return

    diff     = np.abs(phi_mat - phi_np)
    not_nan  = ~np.isnan(diff)
    diff     = diff[not_nan]
    rel_diff = (diff / (np.abs(phi_mat[not_nan]) + 1e-10)) * 100  # avoid div by zero

    if year in ('2d','year'):
        title = 'Structure'
    else:
        title = 'Year'

    print(f'{title} {year} vs MATLAB:\n'
          f'  mean abs diff = {diff.mean():.4f}°\n'
          f'  max abs diff  = {diff.max():.4f}°\n'
          f'  mean rel diff = {rel_diff.mean():.4f}%\n'
          f'  within 0.1°   = {(diff < 0.1).mean()*100:.1f}%\n')

    if year in ('2d','year'):

        for source in ('original', source):
            df = import_data('phi', year, source)
            df_2d = print_coeffs_monthly_ut(df)

            save_dir = import_phi(year, source, True)[1]
            df_2d.to_csv(os.path.join(save_dir,f'phi_{year}.txt'), sep='\t', index=True, float_format='%.3g')


def compare_ab(year, source='staun_proj'):

    ab_mat = import_ab(year)
    ab_np  = import_ab(year, source)

    for key, unit in zip(('a','b'),('mV/m/nT','nT')):
        key_mat = ab_mat[key]
        key_np  = ab_np[key]
        diff    = np.abs(key_mat - key_np)
        diff    = diff[~np.isnan(diff)]

        print(f'Year {year} vs MATLAB for {key}:')
        print(f'  mean | max abs diff  = {diff.mean():.4f} | {diff.max():.4f} {unit}')
        print(f'  within 0.1 {unit:<9} = {(diff < 0.1).mean()*100:.1f}%')
        print(f'  within 0.5 {unit:<9} = {(diff < 0.5).mean()*100:.1f}%')
        print(f'  > 0.5 count          = {(diff > 0.5).sum()} / {diff.size}')

        if (diff > 0.5).any():
            idx = np.argwhere(diff > 0.5)
            print('  Large diff locations (month, time):')
            for i, j in idx[:5]:
                print(f'    [{i:02d},{j:03d}] MATLAB={key_mat[i,j]:.4f}  Python={key_np[i,j]:.4f}  diff={diff[i,j]:.4f}')
        print()

def compare_hproj(year, source='staun_phi'):

    hproj_mat = import_hproj(year, source='original')['hproj'].ravel()
    hproj_np  = import_hproj(year, source=source)['hproj'].ravel()

    if hproj_np.shape!=hproj_mat.shape:
        print(f'Shapes do not match\n   np: {hproj_np.shape}\n  mat: {hproj_mat.shape}')
        return

    diff     = np.abs(hproj_mat - hproj_np)
    not_nan  = ~np.isnan(diff)
    diff     = diff[not_nan]
    rel_diff = (diff / (np.abs(hproj_mat[not_nan]) + 1e-10)) * 100

    print(f'Year {year} vs MATLAB:\n'
          f'  mean abs diff = {diff.mean():.4f} nT\n'
          f'  max abs diff  = {diff.max():.4f} nT\n'
          f'  mean rel diff = {rel_diff.mean():.4f}%\n'
          f'  within 1 nT   = {(diff < 1).mean()*100:.1f}%\n')


def compare_ekl(year, source='updated_phi'):

    ekl_mat = import_er(year, source='original')
    ekl_np  = import_er(year, source=source)

    E_R_mat = ekl_mat['E_R'].ravel()
    E_R_np = ekl_np['E_R']

    if len(E_R_mat)!=len(E_R_np):
        print(f'Shapes do not match\n   np: {len(E_R_np)}\n  mat: {len(E_R_mat)}')
        return

    diff     = np.abs(E_R_mat - E_R_np)
    not_nan  = ~np.isnan(diff)
    diff     = diff[not_nan]
    rel_diff = (diff / (np.abs(E_R_mat[not_nan]) + 1e-6)) * 100

    print(f'Year {year} vs MATLAB:\n'
          f'  mean abs diff = {diff.mean():.4f} mV/m\n'
          f'  max abs diff  = {diff.max():.4f} mV/m\n'
          f'  mean rel diff = {rel_diff.mean():.4f}%\n'
          f'  within 1 mV/m = {(diff < 1).mean()*100:.1f}%\n')

def compare_dist(year):

    mat = import_dist(year, 'original')
    npz = import_dist(year, 'input')

    dist_mat_x = mat['dist_x']
    dist_mat_y = mat['dist_y']

    dist_np_x  = npz['dist_x']
    dist_np_y  = npz['dist_y']

    for comp, mat_arr, np_arr in zip(('X','Y'), (dist_mat_x,dist_mat_y), (dist_np_x,dist_np_y)):
        print(f'Year {year} {comp} vs MATLAB:')
        diff = np_arr - mat_arr
        not_nan  = ~np.isnan(diff)
        diff     = diff[not_nan]
        print(f'  mean diff     = {diff.mean():.4f} nT')  # systematic bias
        print(f'  std diff      = {diff.std():.4f} nT')   # random scatter

        fig, ax = plt.subplots()
        ax.plot(diff[:1440*30])  # first 30 days
        plt.title(f'{comp} Signed difference first 30 days')
        plt.show()
        plt.close()

        diff_daily_mean = diff.reshape(-1, 288).mean(axis=1)  # 288 = 1440/5
        fig, ax = plt.subplots()
        ax.plot(diff_daily_mean)
        plt.title(f'{comp} Daily mean signed difference')
        plt.show()
        plt.close()

        diff     = np.abs(diff)
        rel_diff = (diff / (np.abs(mat_arr[not_nan]) + 1e-3)) * 100
        print(f'  mean abs diff = {diff.mean():.3g} nT\n'
              f'  max abs diff  = {diff.mean():.3g} nT\n'
              f'  med rel diff  = {np.median(rel_diff):.3g}%\n'
              f'  within 1 nT   = {(diff < 1).mean()*100:.1f}%\n'
              f'  within 5 nT   = {(diff < 5).mean()*100:.1f}%\n')

def counts_above_levels(df_pcn, df_stats=None):

    #     PC < 2 => Quiet
    # 2 < PC < 5 => Moderate
    # 5 < PC     => Strong
    # PC > 10 for 1 hour => threat to power grids

    #       unc < 0.1 => insig.
    # 0.2 < unc < 1.0 => troublesome/minor
    # 2.0 < unc       => significant

    if df_stats is None:
        df_stats = pd.DataFrame(columns=['Quiet','Moderate','Strong','Severe','Small','Trouble','Large'])

    quiet    = (df_pcn['pcn']<2)
    moderate = (df_pcn['pcn']>=2) & (df_pcn['pcn']<5)
    strong   = (df_pcn['pcn']>=5) & (df_pcn['pcn']<10)
    severe   = (df_pcn['pcn']>=10)

    small    = (df_pcn['pcn_unc']<0.15)
    minor    = (df_pcn['pcn_unc']>=0.15) & (df_pcn['pcn_unc']<1.5)
    large    = (df_pcn['pcn_unc']>=1.5)

    index = df_pcn.index.year[0]

    df_stats.loc[index] = [np.sum(mask)/1440 for mask in (quiet,moderate,strong,severe,small,minor,large)]

    return df_stats