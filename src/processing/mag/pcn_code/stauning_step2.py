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
from stauning_plots import monthly_phi_plot, plot_phi
from stauning_compares import compare_phi, compare_hproj


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
    UT = (hour * 15.0 + minute * 0.25)[:n]

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

    if source=='staun_omni':
        ekl = scipy.io.loadmat(os.path.join(DIRECTORIES.get('data'), f'ekls_{year}.mat'))[f'ekls_{year}'][0]
        yeartime = scipy.io.loadmat(os.path.join(DIRECTORIES.get('data'), 'yeartime.mat'))['yeartime']
    elif source=='updated_omni':
        raise ValueError(f'"{source}" not implemented.')

    out_dir  = DIRECTORIES.get(source)
    save_dir = os.path.join(out_dir, 'phis')
    os.makedirs(save_dir, exist_ok=True)

    fig_dir  = os.path.join(out_dir, 'contours')
    os.makedirs(fig_dir, exist_ok=True)


    # Transpose for time along rows, variables along columns
    H_proj = Hproj_arr.T
    time_a = yeartime.T
    Phi = np.empty((12, 288), dtype=np.float32)  # store monthly phi results every 5 mins

    print('Looping months.')
    for month in range(12):
        # Initialise correlation array for 288 time steps and 72 phi bins
        R = np.full((288, 72), np.nan, dtype=np.float32)

        # Compute central day of the month and 30-day window around it
        day = 16 + 30 * month
        dmin, dmax = day - 15, day + 15
        imin, imax = dmin * 288, dmax * 288  # convert to 5-minute time bins

        # Slice H_proj, time, and Ekl for the month window
        temp1 = H_proj[imin:imax]
        temp_date = time_a[imin:imax]
        temp_ace = ekl[imin:imax]

        # Loop over hours and 5-minute intervals
        for hr in range(24):
            for minute in range(0, 60, 5):
                # Mask for current hour and 5-minute slot
                mask = ((temp_date[:,0]==hr) & (temp_date[:,1]==minute))
                E = temp_ace[mask]  # Ekl values
                H = temp1[mask]     # H_proj values
                ok = ~np.isnan(E) & ~np.isnan(H[:,0])  # valid points
                E, H = E[ok], H[ok]

                idx = hr*12 + minute//5  # index in 288-bin day
                if E.size > 3:
                    # Remove mean for correlation
                    Ec = E - E.mean()
                    Hc = H - H.mean(axis=0)
                    # Compute correlation numerator and denominator
                    num = np.sum(Ec[:,None]*Hc, axis=0)
                    den = np.sqrt(np.sum(Ec**2)*np.sum(Hc**2, axis=0))
                    R[idx] = num / den  # store correlations

        # Smooth correlations and get best phi per time bin
        best_indices, smooth_R = compute_best_phi(R)
        if show_plot:
            monthly_phi_plot(smooth_R, best_indices, year, month+1, fig_dir)  # optional plotting
        Phi[month] = best_indices*5 - 180  # convert index to degrees
        print(f'  Month {month+1:02d}')


    np.savez_compressed(os.path.join(save_dir,f'Phi_{year}.npz'), Phi=Phi)
    # np.savetxt(os.path.join(OUT_DIR,f'Phi_{year}.csv'), Phi, delimiter=',')  # optional CSV


def compute_best_phi(R, smooth_level=90):
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

    # MATLAB find() is 1-based, so +1 to match
    best_indices = np.nanargmin(smooth_R, axis=1).astype(np.float32) + 1

    # Replicate MATLAB: smooth([ff ff ff], 90, 'lowess') then take middle third
    extended = np.tile(best_indices, 3)
    smoothed = lowess(
        extended,
        np.arange(len(extended)),
        frac=smooth_level / len(extended),  # 90 / (288*3) — span in points
        it=0,                                # MATLAB lowess does 0 robustness iterations
        return_sorted=False
    )
    best_indices_smoothed = smoothed[288:288*2]

    return best_indices_smoothed, smooth_R


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
    phi_mean   = np.nanmean(phi_yearly, axis=2)  # MATLAB: f_avr

    phi_tiled = np.tile(phi_mean, (3, 3))  # MATLAB: F_e
    phi_3d    = np.stack([phi_tiled, phi_tiled, phi_tiled], axis=2)  # MATLAB: F_ex

    # MATLAB smooth3 gaussian and box: kernel size=3, std=0.8
    phi_gauss  = gaussian_filter(phi_3d, sigma=[0.8, 0.8, 0.8], truncate=1.0)  # MATLAB: F_s
    phi_smooth = uniform_filter(phi_gauss, size=3)  # MATLAB: F_n

    # Extract middle block (MATLAB 1-based [13:24, 289:576, 3] → 0-based [12:24, 288:576, 2])
    phi_2d = phi_smooth[12:24, 288:576, 2]  # MATLAB: Fi_2d

    out_dir = DIRECTORIES.get('recreated_phi' if source == 'staun_omni' else 'staun_phi')
    np.savez_compressed(os.path.join(out_dir, 'Phi_2d.npz'), Phi=phi_2d)

    # Interpolate to 1D yearly vector (transpose to match MATLAB calling convention)
    # MATLAB smooth(..., 24, 'lowess'): span=24 points
    phi_year = coeff_for_year(phi_2d.T)  # MATLAB: Fi_year
    phi_year = lowess(phi_year, np.arange(len(phi_year)), frac=24/len(phi_year), return_sorted=False)

    np.savez_compressed(os.path.join(out_dir, 'Phi_year.npz'), Phi=phi_year)

def coeff_for_year(phi_mean):
    """
    --- MATLAB: Function to calculate phi every 5 minutes for the year by interpolation: ---

    Interpolate 2D Phi array to 1D yearly vector (MATLAB coeff_for_year equivalent).
    """
    days_in_year = 366
    n_ut = phi_mean.shape[0]

    # Tile for edge-safe interpolation)
    phi_tiled = np.tile(phi_mean, (3, 3))  # MATLAB: f_tmp

    # MATLAB meshgrid source points (month centres, spaced days_in_year/12 apart)
    x_src = np.arange(15, days_in_year * 3 - 15 + 1, days_in_year / 12)
    y_src = np.arange(1, n_ut * 3 + 1)

    # MATLAB meshgrid target points (every day across 3x tiled year)
    x_tgt = np.arange(1, days_in_year * 3 + 1)
    y_tgt = np.arange(1, n_ut * 3 + 1)

    # Bicubic interpolation (MATLAB interp2 default)
    phi_interp = RectBivariateSpline(y_src, x_src, phi_tiled)(y_tgt, x_tgt)  # MATLAB: fi_tmp1

    # Extract middle block (MATLAB 1-based [289:576, 367:732] → 0-based)
    phi_centre = phi_interp[n_ut:n_ut*2, days_in_year:days_in_year*2]  # MATLAB: fi_avr

    # Flatten column-by-column to yearly vector
    return phi_centre.T.flatten()  # MATLAB: fi_year


def make_hproj(source='staun_phi'):
    """
    --- MATLAB: Calculates H_proj for the years using optimal phi (Fi_year) ---
    Compute and save H_proj for 1997-2009 using the optimal phi angle.

    NOTE: MATLAB hardcodes 1:105120 (365 days), dropping Dec 31st for leap years.
    source='mat' replicates this bug; source='npz' uses the full year.
    """
    phis = import_phi('year', source)['phi']
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

        np.savez_compressed(os.path.join(out_dir, 'hprojs', f'Hproj_{year}.npz'), **{f'Hproj_{year}': H_proj})

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


    phi_step2(source)

# %% main

if __name__ == '__main__':

    main_step1('staun_omni', False)

    #main('update_omni') # not implemented

    for year in range(1997,2010):
        compare_phi(year, 'staun_omni')

    # # uses MATLAB phi angles for every year to calculate phi coefficients
    main_step2('original')
    make_hproj('staun_phi')

    # uses manually calculated phi angles for every year to calculate phi coefficients
    main_step2('staun_omni')
    make_hproj('recreated_phi')

    # uses updated omni and manually calculated angles
    #main('updated_omni')
    #make_hproj('updated_phi')


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