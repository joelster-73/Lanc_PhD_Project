# -*- coding: utf-8 -*-
'''
Created on Sat Feb 28 14:22:13 2026

@author: richarj2
'''

import os
import numpy as np

import scipy.io

from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.interpolate import RectBivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

from config import DIRECTORIES
from stauning_imports import import_phi
from stauning_plots import plot_phi
from stauning_compares import compare_hproj

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

    # Stack yearly phi arrays: (12, 288, 13) then average across years
    strucs     = [import_phi(year, source=source) for year in range(1997, 2010)]
    phi_yearly = np.stack([s['phi'] for s in strucs], axis=2)
    n_valid    = np.sum(~np.isnan(phi_yearly), axis=2)
    phi_mean   = np.nanmean(phi_yearly, axis=2)

    if all('phi_var' in s for s in strucs):
        phi_var_yearly = np.stack([s['phi_var'] for s in strucs], axis=2)
        phi_mean_var   = np.nansum(phi_var_yearly, axis=2) / np.where(n_valid > 0, n_valid**2, np.nan)
    else:
        # Approximate: inter-annual variance of phi across years
        phi_mean_var = np.nanvar(phi_yearly, axis=2, ddof=1) / np.where(n_valid > 0, n_valid, np.nan)

    def _smooth(arr):
            tiled   = np.tile(arr, (3, 3))
            stacked = np.stack([tiled, tiled, tiled], axis=2)
            gauss   = gaussian_filter(stacked, sigma=[0.8, 0.8, 0.8], truncate=1.0)
            smooth  = uniform_filter(gauss, size=3)
            return smooth[12:24, 288:576, 2]

    #---2D structure---#

    phi_2d     = _smooth(phi_mean)
    phi_2d_var = _smooth(phi_mean_var) if phi_mean_var is not None else None

    out_dir = DIRECTORIES.get('recreated_phi' if source == 'staun_omni' else 'staun_phi')
    save_dict = {'Phi': phi_2d}
    if phi_2d_var is not None:
        save_dict['Phi_var'] = phi_2d_var
    np.savez_compressed(os.path.join(out_dir, 'Phi_2d.npz'), **save_dict)

    #---Flat structure---#

    phi_year, phi_year_var = coeff_for_year(phi_2d.T, phi_2d_var.T if phi_2d_var is not None else None)

    x    = np.arange(len(phi_year))
    frac = 24 / len(phi_year)
    phi_year = lowess(phi_year, x, frac=frac, return_sorted=False)
    if phi_year_var is not None:
        phi_year_var = lowess(phi_year_var, x, frac=frac, return_sorted=False)

    save_dict = {'Phi': phi_year}
    if phi_year_var is not None:
        save_dict['Phi_var'] = phi_year_var
    np.savez_compressed(os.path.join(out_dir, 'Phi_year.npz'), **save_dict)



def coeff_for_year(phi_mean, phi_mean_var=None):
    """
    --- MATLAB: Function to calculate phi every 5 minutes for the year by interpolation: ---

    Same spline applied to variance array if provided.

    Interpolate 2D Phi array to 1D yearly vector (MATLAB coeff_for_year equivalent).

    """
    days_in_year = 366
    n_ut  = phi_mean.shape[0]  # 288 (MATLAB: rows)

    phi_tiled = np.tile(phi_mean, (3, 3))  # MATLAB: f_tmp

    # MATLAB meshgrid source points (month centres, spaced days_in_year/12 apart)
    x_src = np.arange(15, days_in_year * 3 - 15 + 1, days_in_year / 12)
    y_src = np.arange(1, n_ut * 3 + 1)

    # MATLAB meshgrid target points (every day across 3x tiled year)
    x_tgt = np.arange(1, days_in_year * 3 + 1)
    y_tgt = np.arange(1, n_ut * 3 + 1)

    # Bicubic interpolation (MATLAB interp2 default)
    phi_interp = RectBivariateSpline(y_src, x_src, phi_tiled)(y_tgt, x_tgt)  # MATLAB: fi_tmp1
    phi_centre = phi_interp[n_ut:n_ut*2, days_in_year:days_in_year*2]        # MATLAB: fi_avr

    if phi_mean_var is not None:
        phi_var_tiled  = np.tile(phi_mean_var, (3, 3))
        phi_var_interp = RectBivariateSpline(y_src, x_src, phi_var_tiled)(y_tgt, x_tgt)
        phi_var_centre = phi_var_interp[n_ut:n_ut*2, days_in_year:days_in_year*2]
        return phi_centre.T.flatten(), phi_var_centre.T.flatten()  # MATLAB: fi_year

    return phi_centre.T.flatten(), None  # MATLAB: fi_year

def make_hproj(source='staun_phi'):
    """
    --- MATLAB: Calculates H_proj for the years using optimal phi (Fi_year) ---
    Compute and save H_proj for 1997-2009 using the optimal phi angle.

    NOTE: MATLAB hardcodes 1:105120 (365 days), dropping Dec 31st for leap years.
    source='mat' replicates this bug; source='npz' uses the full year.
    """
    phi_year = import_phi('year', source)
    phis     = phi_year['phi']
    phi_var  = phi_year.get('phi_var', None)
    out_dir  = DIRECTORIES.get(source)


    yeartime = scipy.io.loadmat(os.path.join(DIRECTORIES.get('data'), 'yeartime.mat'))['yeartime']
    hour   = yeartime[0].astype(np.float32)
    minute = yeartime[1].astype(np.float32)
    UT = hour * 15.0 + minute * 0.25

    for year in range(1997, 2010):
        print(year)
        dist = scipy.io.loadmat(os.path.join(DIRECTORIES.get('data'), f'dist_{year}.mat'))[f'dist_{year}']

        if source in ('staun_phi', 'recreated_phi'):
            # Replicate MATLAB bug: hardcode 365 days, drops Dec 31st on leap years
            n = 105120
        else:
            # Fix: use full year length
            n = dist['x'][0, 0].ravel().shape[0]

        dist_x = dist['x'][0, 0].ravel()[:n].astype(np.float32)
        dist_y = dist['y'][0, 0].ravel()[:n].astype(np.float32)
        phi    = phis[:n]
        ut     = UT[:n]

        y_rad  = np.deg2rad(ut + phi + 291.0)
        H_proj = dist_x * np.sin(y_rad) - dist_y * np.cos(y_rad)

        save_dict = {f'Hproj_{year}': H_proj}

        #---JER manual additions---#
        if phi_var is not None:
            # dH/dphi = dist_x * cos(y) + dist_y * sin(y)
            dH_dphi    = dist_x * np.cos(y_rad) + dist_y * np.sin(y_rad)
            H_proj_var = dH_dphi**2 * phi_var[:n] * (np.pi / 180)**2

            save_dict[f'Hproj_{year}_var'] = H_proj_var

        np.savez_compressed(os.path.join(out_dir, 'hprojs', f'Hproj_{year}.npz'), **save_dict)
# %%

def main(source='original'):
    print()
    print('-----------------------------------')
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

    if False:

        for struc in ('2d','year'):
            plot_phi(struc, source='original') # original mat data
            plot_phi(struc, source='staun_phi') # using mat angles to create best
            plot_phi(struc, source='recreated_phi') # best directions from recreated phi
            #plot_phi(struc, source='updated_phi') # using updated omni

        print('Using the stauning phi data files to construct projections.')
        for year in range(1997, 2010):
            compare_hproj(year)

        print('Using the recreated phi data files to construct projections.')
        for year in range(1997, 2010):
            compare_hproj(year, source='recreated_phi')