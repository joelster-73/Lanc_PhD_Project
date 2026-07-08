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

from scipy.stats import circmean

from .config import LIST_OF_MONTHS
from .stauning_imports import import_er, import_data, DIRECTORIES

def compare_coeff(source='staun_omni'):

    coeff_np = import_data('coeff', var=None, source=source)
    coeff_mat = import_data('coeff', var=None, source='original')

    save_dir = os.path.join(DIRECTORIES.get('analysis'), source)

    print(f'---{source}---')

    col_stats = {}

    for col in coeff_mat:
        mat_series = coeff_mat[col]
        np_series  = coeff_np[col]

        col_label = f'-{col}' if col == 'b' else col
        if col == 'b':
            mat_series = mat_series * -1
            np_series  = np_series  * -1

        if col == 'phi':
            avg_np   = np.degrees(circmean(np.radians(np_series)))
            avg_mat  = np.degrees(circmean(np.radians(mat_series)))
            avg_diff = np.degrees(circmean(np.radians(mat_series) - np.radians(np_series)))
            avg_diff = (avg_diff + 180) % 360 - 180
        else:
            avg_np   = np.mean(np_series)
            avg_mat  = np.mean(mat_series)
            avg_diff = np.mean(mat_series - np_series)

        col_stats[col_label] = {
            'Original':   [mat_series.min(), np.median(mat_series), mat_series.max(), avg_mat],
            'Recreated':  [np_series.min(),  np.median(np_series),  np_series.max(),  avg_np],
            'Difference': [mat_series.min() - np_series.min(),
                                                  np.median(mat_series) - np.median(np_series),
                                                  mat_series.max() - np_series.max(),
                                                  avg_diff],
        }

    index = ['Minimum', 'Median', 'Maximum', 'Average']

    all_data = {}
    for col_label, stats in col_stats.items():
        for stat_name, values in stats.items():
            all_data[f'{col_label} {stat_name}'] = values

    df = pd.DataFrame(all_data, index=index)
    df.index.name = 'Stat'

    filepath = os.path.join(save_dir, 'coeff_compare.txt')
    df.to_csv(filepath, sep='\t', index=True, float_format='%.1f')
    print(f'Saved to {filepath}')


def compare_phi(year, source='staun_omni'):

    if source.lower() == 'updated_omni':
        warnings.warn('Cannot compare npz as it fixes bug.')
        return

    phi_np  = import_data('phi', year, source=source)['phi'].ravel()
    phi_mat = import_data('phi', year, source='original')['phi'].ravel()

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

            save_dir = import_data('phi', year, source, True)[1]
            df_2d.to_csv(os.path.join(save_dir,f'phi_{year}.txt'), sep='\t', index=True, float_format='%.3g')


def compare_ab(year, source='staun_proj'):

    ab_mat = import_data('ab', year)
    ab_np  = import_data('ab', year, source)

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
            for (i,) in idx[:5]:
                print(f'    flat index {i}  diff={diff[i]:.4f}')
        print()

def compare_hproj(year, source='staun_phi'):

    hproj_mat = import_data('hproj', year, source='original')['hproj'].ravel()
    hproj_np  = import_data('hproj', year, source=source)['hproj'].ravel()

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

    mat = import_data('dist', year, 'original')
    npz = import_data('dist', year, 'input')

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

def print_coeffs_monthly_ut(df):

    df_2d = df.copy()

    index_names = df.index.names

    time_vals = df.index.get_level_values('time')
    hours = time_vals.map(lambda t: t.hour)

    if 'month' in index_names:
        months = df.index.get_level_values('month')

    elif 'doy' in index_names:
        doy = df.index.get_level_values('doy')
        months = doy.map(lambda d: pd.Timestamp('2000-01-01') + pd.Timedelta(days=int(d)-1)).month

    df_2d = df.iloc[:, 0].groupby([months, hours]).mean().unstack(level=0)
    df_2d.index.name   = 'hour'
    df_2d.columns.name = 'month'


    df_2d.columns = LIST_OF_MONTHS


    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_2d)

    return df_2d

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