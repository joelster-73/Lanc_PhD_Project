# -*- coding: utf-8 -*-
'''
Created on Sat Feb 28 14:22:13 2026

@author: richarj2
'''

import os
import numpy as np

import scipy.io
import matplotlib.pyplot as plt

from statsmodels.nonparametric.smoothers_lowess import lowess

from config import DATA_DIR, FI_DIR, DIRECTORIES


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
    dist = scipy.io.loadmat(os.path.join(DATA_DIR, f'dist_{year}.mat'))[f'dist_{year}']
    yeartime = scipy.io.loadmat(os.path.join(DATA_DIR, 'yeartime.mat'))['yeartime']

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
        ekl = scipy.io.loadmat(os.path.join(DATA_DIR, f'ekls_{year}.mat'))[f'ekls_{year}'][0]
        yeartime = scipy.io.loadmat(os.path.join(DATA_DIR, 'yeartime.mat'))['yeartime']
    elif source=='updated_omni':
        raise ValueError(f'"{source}" not implemented.')

    out_dir = DIRECTORIES.get(source)

    fig_dir = os.path.join(out_dir, 'contours')
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
            plot_phi(smooth_R, best_indices, year, month+1, fig_dir)  # optional plotting
        Phi[month] = best_indices*5 - 180  # convert index to degrees
        print(f'  Month {month+1:02d}')


    np.savez_compressed(os.path.join(out_dir,f'Phi_{year}.npz'), Phi=Phi)
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


def plot_phi(smooth_R, best_indices, year, month, save_dir):
    """
    In MATLAB, this is done by makerr directly.

    Plot smoothed correlation R and best phi direction ff.
    """
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)

    # Contour plot
    c = ax.contourf(smooth_R, 10, cmap='bwr_r', vmin=-1, vmax=1)

    # Overlay best direction
    ax.plot(best_indices, np.arange(288), c='green', linewidth=2.5)

    # Time ticks every hour
    time_ticks = np.arange(0, 288+1, 12)
    ax.set_yticks(time_ticks)
    ax.set_yticklabels([str(t//12) for t in time_ticks])

    # Phi ticks every 30 degrees
    ax.set_xticks(np.arange(0, 72, 6))  # every 6 indices → 30 degrees
    ax.set_xticklabels([str(i*5 - 180) for i in np.arange(0, 72, 6)])

    ax.set_title(f'Correlation contour ({year}-{month:02d})')
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Time (hours)')

    plt.colorbar(c, ax=ax, label='R')
    plt.savefig(os.path.join(save_dir, f'{year}_{month:02d}.png'), dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()


# %% manual

def compare_phi(year, source='staun_omni'):

    if source.lower() == 'updated_omni':
        raise ValueError ('Cannot compare npz as it fixes bug.')
    in_dir = DIRECTORIES.get(source)
    phi_np = np.load(os.path.join(in_dir, f'Phi_{year}.npz'))['Phi'].astype(np.float32)

    phi_mat = scipy.io.loadmat(os.path.join(FI_DIR, f'F_{year}.mat'))[f'F_{year}'].astype(np.float32)

    diff     = np.abs(phi_mat - phi_np)
    rel_diff = (diff / (np.abs(phi_mat) + 1e-10)) * 100  # avoid div by zero

    print(f'Year {year} vs MATLAB:\n'
          f'  mean abs diff = {diff.mean():.4f}°\n'
          f'  max abs diff = {diff.max():.4f}°\n'
          f'  mean rel diff = {rel_diff.mean():.4f}%\n'
          f'  within 0.1° = {(diff < 0.1).mean()*100:.1f}%\n'
          f'  [numerical precision only — LOWESS interpolation differences]\n')

# %%

def main(source='staun_omni', show_plot=True):
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



if __name__ == '__main__':
    main('staun_omni', True)

    #main('update_omni') # not implemented

    for year in range(1997,2010):
        compare_phi(year, 'staun_omni')


