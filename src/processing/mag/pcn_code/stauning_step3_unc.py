# -*- coding: utf-8 -*-
'''
Created on Sat Feb 28 14:22:13 2026

@author: richarj2
'''

import os
import numpy as np

import scipy.io

from scipy.ndimage import gaussian_filter, uniform_filter

from scipy.signal import resample_poly

from stauning_imports import import_phi, import_ab
from stauning_step3 import coeff_for_year, smooth_coeff, smooth_lowess_year

from config import DIRECTORIES

# %% step1

def ab_step1(source='staun_phi'):
    """
    --- MATLAB: Function manually started, to call function
        to calculate regression coefficients a (slope)  and b (intercept) between.---

    Calculate parameters a and b for 1997 - 2009
    """
    for year in range(1997, 2010):
        print(f'\nProcessing year {year}')
        ab_regress(year, source)

def ab_regress(year, source='staun_phi'):
    """
    --- MATLAB: Function to calculate regression coefficients a (slope) and b (intercept)
        and save those in matrix for each year in files ab_YYYY.mat. ---

    Find best linear fit of Ekl and H_proj
    """
    # Load Ekl file
    if source in ('staun_proj', 'staun_phi', 'recreated_phi'):
        ekl = scipy.io.loadmat(os.path.join(DIRECTORIES.get('data'), f'ekls_{year}.mat'))[f'ekls_{year}']

        yeartime = scipy.io.loadmat(os.path.join(DIRECTORIES.get('data'), 'yeartime.mat'))['yeartime']
        time_a = yeartime.T # transpose to match MATLAB's yeartime'

        if source=='staun_proj':
            in_dir = DIRECTORIES.get('data')
            hproj_data = scipy.io.loadmat(os.path.join(in_dir, f'Hproj_{year}.mat'))
        else:
            in_dir = DIRECTORIES.get(source)
            hproj_data = np.load(os.path.join(in_dir, 'hprojs', f'Hproj_{year}.npz'))

        H_proj = hproj_data[f'Hproj_{year}'].flatten()

    elif source=='updated_phi':
        raise ValueError('Not implemented.')

    # Initialise regression coefficient arrays
    # MATLAB: ab.a(mn, (hr*12)+(min+5)/5) — 1-indexed, so max col = 23*12 + 60/5 = 288
    ab_a = np.full((12, 288), np.nan)
    ab_b = np.full((12, 288), np.nan)

    #---JER manual additions---#
    ab_a_var = np.full((12, 288), np.nan)
    ab_b_var = np.full((12, 288), np.nan)
    ab_covar = np.full((12, 288), np.nan)

    for mn in range(1, 13):
        print(f'  Month {mn:02d}')
        for hr in range(24):
            for minute in range(0, 60, 5):

                day = 16 + 30 * (mn - 1)
                day_min = day - 15
                day_max = day + 15

                min5_min = day_min * 288
                min5_max = day_max * 288

                # Slice arrays (MATLAB is 1-indexed, Python is 0-indexed)
                temp1 = H_proj[min5_min:min5_max + 1]
                temp_dt = time_a[min5_min:min5_max + 1, :]
                temp_ace = ekl[0, min5_min:min5_max + 1]

                # Find indices where hour and minute match
                temp_pos = np.where((temp_dt[:, 0] == hr) & (temp_dt[:, 1] == minute))[0]

                tcorr_Esw = temp_ace[temp_pos].astype(float)
                tcorr_Hproj = temp1[temp_pos].astype(float)

                # Remove NaNs from either array (mutual masking)
                valid = ~np.isnan(tcorr_Esw) & ~np.isnan(tcorr_Hproj)
                tcorr_Esw = tcorr_Esw[valid]
                tcorr_Hproj = tcorr_Hproj[valid]

                # Linear regression (equivalent to polyfit degree 1)
                if len(tcorr_Esw) >= 2:
                    #---JER manual addition, changed cov=True---#
                    #---if include weights as 1/std, changed to cov=unscaled---#
                    popt, pcov = np.polyfit(tcorr_Esw, tcorr_Hproj, 1, cov=True)

                    # MATLAB col index: (hr*12) + (min+5)/5, 1-indexed → subtract 1
                    col = hr * 12 + (minute + 5) // 5 - 1
                    ab_a[mn - 1, col] = popt[0]  # slope
                    ab_b[mn - 1, col] = popt[1]  # intercept
                    ab_a_var[mn - 1, col] = pcov[0,0]
                    ab_b_var[mn - 1, col] = pcov[1,1]
                    ab_covar[mn - 1, col] = pcov[0,1]

    np.savez_compressed(os.path.join(DIRECTORIES.get(source), 'abs', f'ab_{year}.npz'), a=ab_a, b=ab_b, a_var=ab_a_var, b_var=ab_b_var, covar=ab_covar)
    print(f'Saved ab_{year}.npz')

# %% step2
def ab_step2(source='original'):
    """
    --- MATLAB: Stacks for all years matrix with a and b coefficients for all months and all UT-times
        and smoothes this matrix such that hour 23 and hour 00 (and month 12 and month 1) have smooth boundary.
        Saves matrix in ab_2d.mat and plots matrix. Calls function to calculate a and b every 5 minutes
        for the year and saves that as ab_year.mat.---

    Average a and b coefficients
    """

    #---JER manual additions---#
    #---Any references to variance etc.---#

    a_years = []
    b_years = []
    a_var_years = []
    b_var_years = []
    covar_years = []

    for year in range(1997, 2010):
        data = import_ab(year, source)
        a_years.append(data['a'])
        b_years.append(data['b'])
        if 'a_var' in data:
            a_var_years.append(data['a_var'])
            b_var_years.append(data['b_var'])
            covar_years.append(data['covar'])

    has_unc = len(a_var_years) == len(a_years)

    a_years = np.stack(a_years, axis=2)
    b_years = np.stack(b_years, axis=2)

    n_valid = np.sum(~np.isnan(a_years), axis=2)
    a_avr   = np.nanmean(a_years, axis=2)
    b_avr   = np.nanmean(b_years, axis=2)

    if has_unc:
        a_var_years = np.stack(a_var_years, axis=2)
        b_var_years = np.stack(b_var_years, axis=2)
        covar_years = np.stack(covar_years, axis=2)
        a_var_avr   = np.nansum(a_var_years, axis=2) / n_valid**2
        b_var_avr   = np.nansum(b_var_years, axis=2) / n_valid**2
        covar_avr   = np.nansum(covar_years, axis=2) / n_valid**2

    ab_2d_a = smooth_coeff(a_avr, gaussian_sigma=0.8, box_size=5)
    ab_2d_b = smooth_coeff(b_avr, gaussian_sigma=0.1, box_size=7)

    if has_unc:
        ab_2d_a_var  = smooth_unc(a_var_avr,  gaussian_sigma=0.8, box_size=5)
        ab_2d_b_var  = smooth_unc(b_var_avr,  gaussian_sigma=0.1, box_size=7)
        ab_2d_covar = smooth_unc(covar_avr, gaussian_sigma=0.8, box_size=5)

    ab_year_a = smooth_lowess_year(coeff_for_year(ab_2d_a))
    ab_year_b = smooth_lowess_year(coeff_for_year(ab_2d_b))

    if has_unc:
        ab_year_a_var  = smooth_lowess_year(coeff_for_year(ab_2d_a_var))
        ab_year_b_var  = smooth_lowess_year(coeff_for_year(ab_2d_b_var))
        ab_year_covar = smooth_lowess_year(coeff_for_year(ab_2d_covar))

    if source == 'original':
        save_dir = DIRECTORIES.get('ab')
    else:
        save_dir = DIRECTORIES.get(source)

    ab_2d_save   = dict(a=ab_2d_a,   b=ab_2d_b)
    ab_year_save = dict(a=ab_year_a, b=ab_year_b)
    if has_unc:
        ab_2d_save.update(  a_var=ab_2d_a_var,    b_var=ab_2d_b_var,   covar=ab_2d_covar)
        ab_year_save.update(a_var=ab_year_a_var,  b_var=ab_year_b_var, covar=ab_year_covar)

    np.savez_compressed(os.path.join(save_dir, 'ab_2d.npz'),   **ab_2d_save)
    np.savez_compressed(os.path.join(save_dir, 'ab_year.npz'), **ab_year_save)
    print(f'Saved ab_2d.npz and ab_year.npz (uncertainty{"" if has_unc else " not"} included)')


def smooth_unc(c_var, gaussian_sigma, box_size):
    """
    Called by ab_step2()
    """

    ones = np.ones_like(c_var)
    def _filter(x):
        x_e  = np.tile(x, (3, 3))
        x_ex = np.stack([x_e] * 5, axis=2)
        x_s  = gaussian_filter(x_ex.astype(float),
                               sigma=[gaussian_sigma, gaussian_sigma, 0],
                               truncate=gaussian_sigma)
        x_n  = uniform_filter(x_s, size=[box_size, box_size, 1])
        return x_n[12:24, 288:576, 2]
    return _filter(c_var) * _filter(ones)

def make_coeff_1min(source='original'):
    """
    --- MATLAB: Make 1-min file of coefficients from 5-min averaged files. ---

    Phi, a and b are interpolated to 1-minute resolution and combined in one file.
    """
    # load coefficient files
    phi = import_phi('year', source)
    ab  = import_ab('year', source)

    # interpolate coefficients to 1-minute resolution (factor 5)
    coeff = {
        'phi': resample_poly(phi, 5, 1),
        'a':   resample_poly(ab['a'], 5, 1),
        'b':   resample_poly(ab['b'], 5, 1),
    }

    if 'phi_var' in phi:
        coeff['phi_var'] = phi['phi_var']

    for key in ('a_var','b_var','covar'):
        if key in ab:
            coeff[key] = resample_poly(ab[key], 5, 1)

    # save all coeff to file
    if source=='original':
        save_dir = DIRECTORIES.get('coeff')
    else:
        save_dir = DIRECTORIES.get(source)

    out_path = os.path.join(save_dir, 'coeff.npz')
    np.savez_compressed(out_path, **coeff)
    print(f'Saved 1-min coefficients to {out_path}')

    return coeff
# %% main


def main(source='staun_proj', full=False):
    print()
    print('-----------------------------------')
    print('Calculates the best a/b for each UT of every month to use for any year')
    print(f'Uses the {source.upper()} best monthly a/b every 5-minutes of every year')
    if source=='original':
        print('    I.e. a/b calculated by Stauning')
    else:
        print('    I.e. a/b calculated by me')
    print('A smoothing is then applied to determine the best coeffs at every UT for each month')

    if full:
        if source != 'original':
            ab_step1(source)
        ab_step2(source)

    make_coeff_1min(source)


if __name__ == '__main__':

    # # uses MATLAB a/b for every year to calculate a/b coefficients
    main('original')

    # uses MATLAB H projections to calculate a/b for every year then stack
    main('staun_proj')

    # use MATLAB phi to calculate H projections and thus a/b
    main('staun_phi')

    # use recreated phi to calculated H projections and thus a/b
    main('recreated_phi')

    # use updated phi to calcuate H projections and this a/b
    #main('updated_phi')

