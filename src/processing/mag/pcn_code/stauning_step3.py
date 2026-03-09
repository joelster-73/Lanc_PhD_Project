# -*- coding: utf-8 -*-
'''
Created on Sat Feb 28 14:22:13 2026

@author: richarj2
'''

import os
import numpy as np

import scipy.io

from scipy.ndimage import gaussian_filter, uniform_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import RectBivariateSpline
from scipy.signal import resample_poly

from stauning_imports import import_phi, import_ab
from stauning_plots import plot_ab

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
                temp_pos = np.where(
                    (temp_dt[:, 0] == hr) & (temp_dt[:, 1] == minute)
                )[0]

                tcorr_Esw = temp_ace[temp_pos].astype(float)
                tcorr_Hproj = temp1[temp_pos].astype(float)

                # Remove NaNs from either array (mutual masking)
                valid = ~np.isnan(tcorr_Esw) & ~np.isnan(tcorr_Hproj)
                tcorr_Esw = tcorr_Esw[valid]
                tcorr_Hproj = tcorr_Hproj[valid]

                # Linear regression (equivalent to polyfit degree 1)
                if len(tcorr_Esw) >= 2:
                    R = np.polyfit(tcorr_Esw, tcorr_Hproj, 1)
                    # MATLAB col index: (hr*12) + (min+5)/5, 1-indexed → subtract 1
                    col = hr * 12 + (minute + 5) // 5 - 1
                    ab_a[mn - 1, col] = R[0]  # slope
                    ab_b[mn - 1, col] = R[1]  # intercept

    np.savez_compressed(os.path.join(DIRECTORIES.get(source), 'abs', f'ab_{year}.npz'), a=ab_a, b=ab_b)
    print(f'Saved ab_{year}.npz')

# %% step2
def ab_step2(source='original', show_plot=False):
    """
    --- MATLAB: Stacks for all years matrix with a and b coefficients for all months and all UT-times
        and smoothes this matrix such that hour 23 and hour 00 (and month 12 and month 1) have smooth boundary.
        Saves matrix in ab_2d.mat and plots matrix. Calls function to calculate a and b every 5 minutes
        for the year and saves that as ab_year.mat.---

    Average a and b coefficients
    """
    # Load all ab files from 1997 to 2009
    a_years = []
    b_years = []

    for year in range(1997, 2010):
        data = import_ab(year, source)
        a_years.append(data['a'])
        b_years.append(data['b'])

    a_years = np.stack(a_years, axis=2)  # shape: (12, 288, 13)
    b_years = np.stack(b_years, axis=2)

    # Average across years (ignoring NaNs)
    a_avr = np.nanmean(a_years, axis=2)  # shape: (12, 288)
    b_avr = np.nanmean(b_years, axis=2)

    ab_2d_a = smooth_coeff(a_avr, gaussian_size=7, gaussian_sigma=0.8, box_size=5)
    ab_2d_b = smooth_coeff(b_avr, gaussian_size=9, gaussian_sigma=0.1, box_size=7)

    ab_year_a = smooth_lowess_year(coeff_for_year(ab_2d_a))
    ab_year_b = smooth_lowess_year(coeff_for_year(ab_2d_b))

    # Save
    if source=='original':
        save_dir = DIRECTORIES.get('ab')
    else:
        save_dir = DIRECTORIES.get(source)

    np.savez_compressed(os.path.join(save_dir,'ab_2d.npz'),   a=ab_2d_a,   b=ab_2d_b)
    np.savez_compressed(os.path.join(save_dir,'ab_year.npz'), a=ab_year_a, b=ab_year_b)
    print('Saved ab_2d.npz and ab_year.npz')

def smooth_coeff(c_avr, gaussian_sigma, box_size):
    """
    Called by ab_step2()
    """
    # Tile 3x3 in 2D, then stack 5 times in 3rd dim
    c_e = np.tile(c_avr, (3, 3))                      # (36, 864)
    c_ex = np.stack([c_e] * 5, axis=2)                # (36, 864, 5)

    # smooth3 gaussian: scipy's gaussian_filter applies per-axis sigma
    c_s = gaussian_filter(c_ex.astype(float),
                          sigma=[gaussian_sigma, gaussian_sigma, 0],
                          truncate=gaussian_sigma)
    # smooth3 box: uniform_filter equivalent
    c_n = uniform_filter(c_s, size=[box_size, box_size, 1])

    # Extract centre tile: MATLAB c_n(13:12*2, 289:288*2, 3) — 1-indexed

    return c_n[12:24, 288:576, 2]

def smooth_lowess_year(data, span=24):
    """
    Called by ab_step2()
    """

    extended = np.tile(data, 3)
    n = len(data)
    smoothed = lowess(extended, np.arange(len(extended)),
                      frac=span / len(extended), it=0, return_sorted=False)
    return smoothed[n:2*n]

def coeff_for_year(f_avr):
    """
    --- MATLAB: Function to calculate a and b every 5 minutes for the year by interpolation ---

    Interpolate 1-D array from 2-D array of coefficients matrix
    """

    days_in_year = 366

    # Tile 3x3
    f_tmp = np.tile(f_avr, (3, 3))  # shape: (36, 864)

    # MATLAB meshgrid: X varies along columns, Y along rows
    x_points = np.arange(15, days_in_year * 3 - 15 + 1, days_in_year / 12)  # 36 points
    y_points = np.arange(1, 288 * 3 + 1)                                     # 864 points

    XI = np.arange(1, days_in_year * 3 + 1)   # 1098 points
    YI = np.arange(1, 288 * 3 + 1)            # 864 points

    # interp2 in MATLAB takes (X, Y, Z, XI, YI) where Z is f_tmp'
    # scipy's RectBivariateSpline takes (x, y, z) where z[i,j] = f(x[i], y[j])

    spline  = RectBivariateSpline(y_points, x_points, f_tmp.T)
    fi_tmp1 = spline(YI, XI)  # shape: (864, 1098)

    # Extract centre: MATLAB fi_tmp1([289:576], [367:732]) 1-indexed
    fi_avr = fi_tmp1[288:576, days_in_year:days_in_year*2]  # (288, 366)

    # Flatten column by column into 1D array (288 samples per day)
    fi_year = fi_avr.T.flatten()  # shape: (366*288,)

    return fi_year


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
        'a': resample_poly(ab['a'], 5, 1),
        'b': resample_poly(ab['b'], 1, 1),
    }

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


def main(source='staun_proj'):
    print('Calculates the best a/b for each UT of every month to use for any year')
    print(f'Uses the {source.upper()} best monthly a/b every 5-minutes of every year')
    if source=='original':
        print('    I.e. a/b calculated by Stauning')
    else:
        print('    I.e. a/b calculated by me')
    print('A smoothing is then applied to determine the best coeffs at every UT for each month')

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


    for struc in ('2d','year'):
        plot_ab(struc, source='original') # original mat data
        plot_ab(struc, source='staun_proj')
        plot_ab(struc, source='staun_phi') # using mat angles to create best
        plot_ab(struc, source='recreated_phi') # best directions from recreated phi
        #phi_plot(struc, source='updated_phi') # using updated omni

    for source in ('staun_proj', 'staun_phi', 'recreated_phi'):
        print(f'Using the {source} data files to construct projections.')
        for year in range(1997, 2010):
            compare_ab(year, source)
        break
