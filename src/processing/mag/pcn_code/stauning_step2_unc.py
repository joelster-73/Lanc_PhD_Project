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

from src.processing.mag.pcn_code.config import DIRECTORIES
from src.processing.mag.pcn_code.stauning_imports import import_phi, import_er
from src.processing.mag.pcn_code.stauning_plots import monthly_phi_plot, plot_phi
from src.processing.mag.pcn_code.stauning_compares import compare_phi, compare_hproj

# %% step1

def phi_step1(source='staun_omni', show_plot=True):
    """
    ---MATLAB: Function manually called to call functions below:---

    Compute monthly Phi arrays for 1997-2009.
    """
    for year in range(1997, 2010):
        print(f'\nProcessing year {year}')
        Hproj_arr = phi_arr(year, source)
        phi_corr(year, Hproj_arr, source, show_plot)

def phi_arr(year, source):
    """
    --- MATLAB: Function calls function to calculate projections (5 degree steps)
        of disturbance every 5 minutes (fi stands for angle phi) ---
    --- Combined with `hproj` function ---
    Compute H_proj for all phi for a given year.

    NOTE: MATLAB hardcodes 1:105120 (365 days), dropping Dec 31st for leap years.
    source='mat' replicates this bug; source='npz' uses the full year.
    """
    print('Loading mag files.')
    dist = scipy.io.loadmat(os.path.join(DIRECTORIES.get('data'), f'dist_{year}.mat'))[f'dist_{year}']
    yeartime = scipy.io.loadmat(os.path.join(DIRECTORIES.get('data'), 'yeartime.mat'))['yeartime']

    hour   = yeartime[0].astype(np.float32)
    minute = yeartime[1].astype(np.float32)

    if source == 'staun_omni':
        # Replicate MATLAB bug: hardcode 365 days, drops Dec 31st on leap years
        n = 105120
    else:
        # Fix: use full year length (105120 for 365 days, 105408 for 366 days)
        n = dist['x'][0,0].ravel().shape[0]

    dist_x = dist['x'][0,0].ravel()[:n].astype(np.float32)
    dist_y = dist['y'][0,0].ravel()[:n].astype(np.float32)
    reps = int(np.ceil(n / 288))
    hour_full   = np.tile(hour,   reps)[:n]
    minute_full = np.tile(minute, reps)[:n]

    UT = (hour_full * 15.0 + minute_full * 0.25)

    Hproj_arr = np.empty((72, n), dtype=np.float32)
    print('Projecting every phi.')
    for k, phi in enumerate(range(0, 360, 5)):
        y = UT + phi + 291.0
        Hproj_arr[k] = dist_x * np.sin(np.deg2rad(y)) - dist_y * np.cos(np.deg2rad(y))

    return Hproj_arr

def phi_corr(year, Hproj_arr, source, show_plot=False):
    """
    --- MATLAB: Function to determine optimal correlation between projection of disturbance
        and EKL and saves optimal phi-angel in yearly files F_YYYY.m, calling for this function makerr.m: ---

    Compute correlations between H_proj and Ekl, save Phi_YEAR arrays.

    NOTE: The electric field in ekls is already time-shifted by 15-mins.
    """
    print('Loading sw files.')

    # Load Ekl data and the time array for the year

    field_dict = import_er(year, source)
    ekl        = field_dict['E_R']
    yeartime   = field_dict['t']

    if source=='updated_phi':
        yeartime = np.array([[dt.hour, dt.minute] for dt in yeartime.flatten()])

    out_dir  = DIRECTORIES.get(source)
    save_dir = os.path.join(out_dir, 'phis')
    os.makedirs(save_dir, exist_ok=True)

    fig_dir  = os.path.join(out_dir, 'contours')
    os.makedirs(fig_dir, exist_ok=True)

    # Transpose for time along rows, variables along columns
    H_proj  = Hproj_arr.T
    time_a  = yeartime.T
    Phi     = np.empty((12, 288), dtype=np.float32)  # store monthly phi results every 5 mins
    Phi_var = np.empty((12, 288), dtype=np.float32)

    print('Looping months.')
    for month in range(12):
        print(f'  Month {month+1:02d}')
        # Initialise correlation array for 288 time steps and 72 phi bins
        R = np.full((288, 72), np.nan, dtype=np.float32)
        R_se = np.full((288, 72), np.nan, dtype=np.float32) # standard error

        # Compute central day of the month and 30-day window around it
        day = 16 + 30 * month
        dmin, dmax = day - 15, day + 15
        imin, imax = dmin * 288, dmax * 288 + 1

        # Slice H_proj, time, and Ekl for the month window
        temp1    = H_proj[imin:imax]
        temp_date = time_a[imin:imax]
        ekl_window = ekl[imin:imax]        # MATLAB: temp_ace

        # Loop over hours and 5-minute intervals
        for hr in range(24):
            for minute in range(0, 60, 5):
                # Mask for current hour and 5-minute slot
                mask = ((temp_date[:,0]==hr) & (temp_date[:,1]==minute))
                ekl_vals  = ekl_window[mask]   # MATLAB: tabcorr_Esw   = temp_ace(temp_pos)

                H = temp1[mask]                # MATLAB: tabcorr_Hproj = temp1(temp_pos,:)
                ok = ~np.isnan(ekl_vals) & ~np.isnan(H[:,0])  # valid points
                ekl_vals, H = ekl_vals[ok], H[ok]

                idx = hr*12 + minute//5  # index in 288-bin day
                if ekl_vals.size > 3:
                    # Remove mean for correlation
                    ekl_anom = ekl_vals - ekl_vals.mean()   # MATLAB: Ec = E - mean(E)
                    Hc = H - H.mean(axis=0)
                    # Compute correlation numerator and denominator
                    num = np.sum(ekl_anom[:,None]*Hc, axis=0)
                    den = np.sqrt(np.sum(ekl_anom**2)*np.sum(Hc**2, axis=0))
                    R[idx] = num / den  # store correlations

                    #---JER manual additions---#
                    # Add uncertainty estimate
                    n = ekl_vals.size
                    R_se[idx] = 1.0 / np.sqrt(n - 3)  # standard error via Fisher z-transform

        # Smooth correlations and get best phi per time bin
        best_indices, smooth_R, phi_uncertainty = compute_best_phi(R, R_se)
        Phi_var[month] = phi_uncertainty**2
        if show_plot:
            monthly_phi_plot(smooth_R, best_indices, year, month+1, fig_dir)  # optional plotting
        Phi[month] = best_indices*5 - 180  # convert index to degrees

    np.savez_compressed(os.path.join(save_dir,f'Phi_{year}.npz'), Phi=Phi, Phi_var=Phi_var)

def compute_best_phi(R, R_se, smooth_level=90):
    """
    --- MATLAB: Function (called fi_corr.m) to find optimum correlation:---
    --- named `makerr` in MATLAB ---

    Smooth correlations with 5th-degree polynomial and return smoothed best phi indices.
    NOTE: Has been vectorised
    Fully replicates MATLAB 'makerr' function.
    """
    x = np.arange(1, 73, dtype=np.float32)

    def fit_row(row):
        coeffs = np.polyfit(x, row, 5)
        return np.polyval(coeffs, x)

    smooth_R = np.apply_along_axis(fit_row, 1, R)
    best_indices = np.nanargmin(smooth_R, axis=1).astype(np.float32) + 1 # MATLAB find() is 1-based, so +1 to match

    # Replicate MATLAB: smooth([ff ff ff], 90, 'lowess') then take middle third
    # 90 / (288*3) — span in points
    # MATLAB lowess does 0 robustness iterations
    extended = np.tile(best_indices, 3)
    smoothed = lowess(extended, np.arange(len(extended)), frac=smooth_level / len(extended), it=0, return_sorted=False)
    best_indices_smoothed = smoothed[288:288*2]

    #---JER manual additions---#
    # Uncertainty: std of top-N candidates within a threshold of the minimum
    phi_uncertainty = np.full(smooth_R.shape[0], np.nan)
    for i, row in enumerate(smooth_R):
        se      = np.nanmean(R_se[i])
        min_val = np.nanmin(row)
        within  = np.where(row <= min_val + se)[0]

        phi_uncertainty[i] = (within[-1] - within[0]) * 5

    return best_indices_smoothed, smooth_R, phi_uncertainty

# %% step2

def phi_step2(source='original', exc2003=True):
    """
    --- MATLAB: Stacks for all years matrix with phi-angles for all months and all UT-times
       and smoothes this matrix such that hour 23 and hour 00 (and month 12 and month 1) have smooth boundary.
       Saves matrix in Fi_2d.mat and plots matrix.
       Calls function to calculate phi every 5 minutes for the year and saves that as Fi_year.mat.---

    Average Phi arrays, smooth and interpolate to 1D yearly array.

    In 120-page documentation, they say they exclude 2003 data as there's no Vostok data (even for Thule coefficients). This is not in their code, but implemented as such below if the flag is true.
    """
    print('\n\nLoading yearly Phi arrays.')

    # Stack yearly phi arrays: (12, 288, 13) then average across years
    years      = [y for y in range(1997, 2010) if not (exc2003 and y == 2003)]
    strucs     = [import_phi(year, source=source) for year in years]
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

    out_dir = DIRECTORIES.get({'staun_omni': 'recreated_phi', 'updated_omni': 'updated_phi'}.get(source,'staun_phi'))
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


        dist_x = dist['x'][0,0].ravel()[:n].astype(np.float32)
        dist_y = dist['y'][0,0].ravel()[:n].astype(np.float32)
        phi    = phis[:n]

        reps = int(np.ceil(n / 288))
        hour_full   = np.tile(hour,   reps)[:n]
        minute_full = np.tile(minute, reps)[:n]

        UT = (hour_full * 15.0 + minute_full * 0.25)
        y_rad  = np.deg2rad(UT + phi + 291.0)

        H_proj = dist_x * np.sin(y_rad) - dist_y * np.cos(y_rad)

        save_dict = {f'Hproj_{year}': H_proj}

        #---JER manual additions---#
        if phi_var is not None:
            # dH/dphi = dist_x * cos(y) + dist_y * sin(y)
            dH_dphi    = dist_x * np.cos(y_rad) + dist_y * np.sin(y_rad)
            H_proj_var = dH_dphi**2 * phi_var[:n] * (np.pi / 180)**2

            save_dict[f'Hproj_{year}_var'] = H_proj_var

        np.savez_compressed(os.path.join(out_dir, 'hprojs', f'Hproj_{year}.npz'), **save_dict)

# %% mains

def main_step1(source='staun_omni', show_plot=True):
    print()
    print('-----------------------------------')
    print('Calculates the best phi for every month over 1999 to 2009')
    print('Using the magnetometer disturbances from Stauning (I.e. their QDC corrected data)')
    print('A fixed 15-minute time lag between BSN and PC is assumed')
    print('Projected along various phi angles (from 0 to 360)')
    print(f'Uses the {source.upper()} OMNI electric fields')
    if source=='staun_omni':
        print('    I.e. E_R calculated by Stauning using old OMNI')
    elif source=='updated_omni':
        print('    I.e. E_R calculated by me using new OMNI')
    print('A smoothing is then applied to determine the best phi for every five minutes of a day for each month')

    phi_step1(source, show_plot)

def main_step2(source='original'):
    print()
    print('-----------------------------------')
    print('Calculates the best phi for each UT of every month to use for any year')
    print(f'Uses the {source.upper()} best monthly phi every 5-minutes of every year')
    if source=='original':
        print('    I.e. phi calculated by Stauning')
    else:
        print('    I.e. phi calculated by me')
    print('A smoothing is then applied to determine the best direction at every UT for each month')


    phi_step2(source, True)

# %% main

if __name__ == '__main__':

    if True:

        #main_step1('staun_omni', False)
        main_step1('updated_omni', False)

        for year in range(1997,2010):
            compare_phi(year, 'staun_omni')
            compare_phi(year, 'updated_omni')

        # # uses MATLAB phi angles for every year to calculate phi coefficients
        main_step2('original')
        make_hproj('staun_phi')

        # uses manually calculated phi angles for every year to calculate phi coefficients
        main_step2('staun_omni')
        make_hproj('recreated_phi')

        # uses updated omni and manually calculated angles
        main_step2('updated_omni')
        make_hproj('updated_phi')

    if True:

        for struc in ('2d','year'):
            plot_phi(struc, source='original') # original mat data
            plot_phi(struc, source='staun_phi') # using mat angles to create best
            plot_phi(struc, source='recreated_phi') # best directions from recreated phi
            plot_phi(struc, source='updated_phi') # using updated omni

            compare_phi(struc, 'staun_phi')
            compare_phi(struc, 'recreated_phi')
            compare_phi(struc, 'updated_phi')

        print('Using the stauning phi data files to construct projections.')
        for year in range(1997, 2010):
            compare_hproj(year)

        print('Using the recreated phi data files to construct projections.')
        for year in range(1997, 2010):
            compare_hproj(year, source='recreated_phi')
