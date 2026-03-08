# -*- coding: utf-8 -*-
'''
Created on Sat Feb 28 14:22:13 2026

@author: richarj2
'''

import os
import numpy as np

import scipy.io
import matplotlib.pyplot as plt

from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.interpolate import RectBivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

from config import DATA_DIR, FI_DIR, DIRECTORIES


# %% step2

def phi_step2(source='original'):
    """
    --- MATLAB: Stacks for all years matrix with phi-angles for all months and all UT-times
       and smoothes this matrix such that hour 23 and hour 00 (and month 12 and month 1) have smooth boundary.
       Saves matrix in Fi_2d.mat and plots matrix.
       Calls function to calculate phi every 5 minutes for the year and saves that as Fi_year.mat.---

    Average Phi arrays, smooth and interpolate to 1D yearly array.
    """
    print('\n\nLoading yearly Phi arrays.')

    fa = []
    for year in range(1997, 2010):
        fa.append(phi_import(year, source=source))  # shape: (rows, cols)

    fa = np.stack(fa, axis=2)  # shape: (rows, cols, 13)

    # Average across years (ignoring NaNs)
    f_avr = np.nanmean(fa, axis=2)  # shape: (rows, cols)

    # Tile into 3x3 block
    F_e = np.tile(f_avr, (3, 3))  # shape: (rows*3, cols*3)

    # Stack 3 copies along a new 3rd dimension
    F_ex = np.stack([F_e, F_e, F_e], axis=2)  # shape: (rows*3, cols*3, 3)

    # smooth3 gaussian equivalent (sigma derived from MATLAB's [3, 0.8] args)
    # MATLAB smooth3 gaussian: kernel size=3, std=0.8, applied to all 3 dims
    F_s = gaussian_filter(F_ex, sigma=[0.8, 0.8, 0.8], truncate=1.0)

    # smooth3 box equivalent: kernel size 3 in all dims = uniform filter of size 3
    F_n = uniform_filter(F_s, size=3)

    # Extract middle slice (MATLAB 1-based [13:24, 289:576, 3])
    # Converted to 0-based: rows 12:24, cols 288:576, layer index 2
    Fi_2d = F_n[12:24, 288:576, 2]  # shape: (12, 288)

    if source=='staun_omni':
        out_dir = DIRECTORIES.get('recreated_phi')
    elif source=='original':
        out_dir = DIRECTORIES.get('staun_phi')

    np.savez_compressed(os.path.join(out_dir,'Phi_2d.npz'), Phi=Fi_2d)

    # Call coeff_for_year and transpose
    Fi_year = coeff_for_year(Fi_2d.T)  # transpose input & output to match MATLAB

    # MATLAB smooth(..., 24, 'lowess') → locally weighted linear regression
    # Best Python equivalent is Savitzky-Golay or a manual LOWESS
    x = np.arange(len(Fi_year))
    # MATLAB 'lowess' span=24 means 24/N fraction of total points
    frac = 24 / len(Fi_year)
    smoothed = lowess(Fi_year, x, frac=frac, return_sorted=False)
    Fi_year = smoothed

    np.savez_compressed(os.path.join(out_dir,'Phi_year.npz'), Phi=Fi_year)

def coeff_for_year(f_avr):
    """
    --- MATLAB: Function to calculate phi every 5 minutes for the year by interpolation: ---

    Interpolate 2D Phi array to 1D yearly vector (MATLAB coeff_for_year equivalent).
    """
    days_in_year = 366

    # Tile f_avr into 3x3 block (equivalent to MATLAB's repmat-style construction)
    f_tmp = np.tile(f_avr, (3, 3))  # shape: (rows*3, cols*3)

    rows = f_avr.shape[0]  # 288
    #cols = f_avr.shape[1]  # days_in_year (12 months)

    # MATLAB meshgrid source points
    x_src = np.arange(15, days_in_year * 3 - 15 + 1, days_in_year / 12)  # length = cols*3
    y_src = np.arange(1, rows * 3 + 1)                                     # length = rows*3

    # MATLAB meshgrid target points
    x_tgt = np.arange(1, days_in_year * 3 + 1)  # length = days_in_year*3
    y_tgt = np.arange(1, rows * 3 + 1)           # length = rows*3

    # Interpolate (note: f_tmp is transposed in MATLAB call, so we match that)
    # RectBivariateSpline expects z[i, j] = f(x[i], y[j])
    interp_func = RectBivariateSpline(y_src, x_src, f_tmp)

    fi_tmp1 = interp_func(y_tgt, x_tgt)  # shape: (rows*3, days_in_year*3)

    # Extract the middle block (MATLAB 1-based: [289:576, 367:732])
    fi_avr = fi_tmp1[rows:rows*2, days_in_year:days_in_year*2]  # shape: (288, 366)

    # Flatten column by column into output vector
    fi_year = fi_avr.T.flatten()  # shape: (days_in_year * 288,)

    return fi_year

def phi_import(var, source='original'):

    print(f'\nImporting {var} for data using {source} input.')

    if source == 'original':
        if isinstance(var,int):
            var = f'F_{var}'
        else:
            var = f'Fi_{var}'
        path = os.path.join(FI_DIR, f'{var}.mat')
        mat = scipy.io.loadmat(path)
        struc = mat[var]

    else:
        in_dir = DIRECTORIES.get(source)

        path = os.path.join(in_dir, f'Phi_{var}.npz')
        data = np.load(path)
        var = 'Phi'
        struc = data[var]

    return struc


def phi_plot(var='2d', source='original', phi_data=None):
    """
    Load Phi data (.npz or .mat) and generate a contour plot.
    var: 'Phi_2d' or 'Phi_year'
    """
    if phi_data is None:
        phi_data = phi_import(var=var, source=source)

    # For Phi_year, reshape to 12x288 for plotting
    if var == 'year':
        phi_data = phi_data.reshape(-1, 288)

    fig, ax = plt.subplots(figsize=(10,6), dpi=300)
    cb = plt.contourf(phi_data, 12, cmap='rainbow')
    _ = plt.contour(phi_data, 12, colors='black', linewidths=0.5)
    cbar = plt.colorbar(cb)
    cbar.ax.set_ylabel('Phi')

    ax.set_title(f'{var.capitalize()} distribution ({source})')
    ax.set_xlabel('UT hour')
    ax.set_ylabel('Month')

    # Time ticks every hour
    time_ticks = np.arange(0, phi_data.shape[1]+1, 12)
    ax.set_xticks(time_ticks)
    ax.set_xticklabels([str(t//12) for t in time_ticks])

    # Time ticks every hour
    if var == 'year':
        time_ticks = np.arange(0, phi_data.shape[0]+1, phi_data.shape[0]//12)
        ax.set_yticks(time_ticks)
        ax.set_yticklabels([str(t//30) for t in time_ticks])

    if source=='original':
        save_dir = FI_DIR
    else:
        save_dir = DIRECTORIES.get(source)
    plt.savefig(os.path.join(save_dir, f'phi_{var}_{source}.png'), dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()

    return phi_data

def make_hproj(source='staun_phi'):
    """
    --- MATLAB: Calculates H_proj for the years using optimal phi (Fi_year) ---
    Compute and save H_proj for 1997-2009 using the optimal phi angle.

    NOTE: MATLAB hardcodes 1:105120 (365 days), dropping Dec 31st for leap years.
    source='mat' replicates this bug; source='npz' uses the full year.
    """
    phi_year = phi_import('year', source)
    out_dir  = DIRECTORIES.get(source)

    yeartime = scipy.io.loadmat(os.path.join(DATA_DIR, 'yeartime.mat'))['yeartime']
    hour   = yeartime[0].astype(np.float32)
    minute = yeartime[1].astype(np.float32)
    UT = hour * 15.0 + minute * 0.25

    for year in range(1997, 2010):
        print(year)
        dist = scipy.io.loadmat(os.path.join(DATA_DIR, f'dist_{year}.mat'))[f'dist_{year}']

        if source in ('staun_phi', 'recreated_phi'):
            # Replicate MATLAB bug: hardcode 365 days, drops Dec 31st on leap years
            n = 105120
        else:
            # Fix: use full year length
            n = dist['x'][0, 0].ravel().shape[0]

        dist_x = dist['x'][0, 0].ravel()[:n].astype(np.float32)
        dist_y = dist['y'][0, 0].ravel()[:n].astype(np.float32)
        phi    = phi_year[:n]
        ut     = UT[:n]

        y = ut + phi + 291.0
        H_proj = dist_x * np.sin(np.deg2rad(y)) - dist_y * np.cos(np.deg2rad(y))

        np.savez_compressed(os.path.join(out_dir, f'Hproj_{year}.npz'), **{f'Hproj_{year}': H_proj})
# %% manual

def compare_hproj(year, source='staun_phi'):
    hproj_mat = scipy.io.loadmat(os.path.join(DATA_DIR, f'Hproj_{year}.mat'))[f'Hproj_{year}'].astype(np.float32).ravel()

    compare_dir = DIRECTORIES.get(source)
    hproj_np  = np.load(os.path.join(compare_dir, f'Hproj_{year}.npz'))[f'Hproj_{year}'].astype(np.float32).ravel()

    diff     = np.abs(hproj_mat - hproj_np)
    rel_diff = (diff / (np.abs(hproj_mat) + 1e-10)) * 100

    print(f'Year {year} vs MATLAB:\n'
          f'  mean abs diff = {diff.mean():.4f} nT\n'
          f'  max abs diff = {diff.max():.4f} nT\n'
          f'  mean rel diff = {rel_diff.mean():.4f}%\n'
          f'  within 1 nT = {(diff < 1).mean()*100:.1f}%\n')

# %%

def main(source='original'):
    print('Calculates the best phi for each UT of every month to use for any year')
    print(f'Uses the {source.upper()} best monthly phi every 5-minutes of every year')
    if source=='original':
        print('    I.e. phi calculated by Stauning')
    else:
        print('    I.e. phi calculated by me')
    print('A smoothing is then applied to determine the best direction at every UT for each month')


    phi_step2(source)


if __name__ == '__main__':

    # # uses MATLAB phi angles for every year to calculate phi coefficients
    main('original')
    make_hproj('staun_phi')

    # uses manually calculated phi angles for every year to calculate phi coefficients
    main('staun_omni')
    make_hproj('recreated_phi')

    # uses updated omni and manually calculated angles
    #main('updated_omni')
    #make_hproj('updated_phi')


    for struc in ('2d','year'):
        phi_plot(struc, source='original') # original mat data
        phi_plot(struc, source='staun_phi') # using mat angles to create best
        phi_plot(struc, source='recreated_phi') # best directions from recreated phi
        #phi_plot(struc, source='updated_phi') # using updated omni

    print('Using the stauning phi data files to construct projections.')
    for year in range(1997, 2010):
        compare_hproj(year)

    print('Using the recreated phi data files to construct projections.')
    for year in range(1997, 2010):
        compare_hproj(year, source='recreated_phi')