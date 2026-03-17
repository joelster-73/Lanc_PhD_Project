# -*- coding: utf-8 -*-
'''
Created on Sat Feb 28 14:22:13 2026

@author: richarj2
'''


import os
import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.processing.mag.pcn_code.config import DIRECTORIES
from src.processing.mag.pcn_code.stauning_compares import compare_dist, compare_ekl
from src.processing.mag.pcn_code.stauning_imports import import_def_omni

# %% disturbances

def _robust_loess(y, span, it=3):
    """
    Robust local quadratic smoother matching MATLAB's smooth(..., span, 'rloess').
    span : number of points in the smoothing window (odd; incremented if even)
    it   : number of robust iterations
    """
    from scipy.signal import savgol_filter
    y = np.array(y, dtype=np.float64)
    n = len(y)
    w = span if span % 2 == 1 else span + 1
    w = min(w, n if n % 2 == 1 else n - 1)  # must not exceed array length

    result = savgol_filter(y, window_length=w, polyorder=2)

    for _ in range(it):
        residuals = y - result
        mad = np.median(np.abs(residuals))
        if mad < 1e-10:
            break
        u = np.clip(residuals / (6.0 * mad), -1.0, 1.0)
        robust_w = (1.0 - u ** 2) ** 2   # bisquare weights

        # weighted savgol: apply weights by scaling y then normalising
        denom = savgol_filter(robust_w, window_length=w, polyorder=2)
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        result = savgol_filter(y * robust_w, window_length=w, polyorder=2) / denom

    return result


def ss_interp(var):
    """
    --- MATLAB: Function to calculate sector structure and explanation of Matlab function 'smooth': ss_interp.m ---

    Compute sector structure (SS) for a 1-minute array covering exactly one year.
    """
    var = np.array(var, dtype=np.float64)
    day_points = 1440
    ndays = len(var) // day_points

    # daily median
    day_med = np.array([
        np.nanmedian(var[i * day_points:(i + 1) * day_points])
        for i in range(ndays)
    ])

    # smooth medians: 7-day moving average then 7-day robust loess (quadratic, matches 'rloess')
    w = pd.Series(day_med).rolling(7, center=True, min_periods=1).mean().values
    w = _robust_loess(w, span=7, it=3)

    # spline interpolation to 1-minute resolution (matches interp1 'spline')
    day_idx = np.arange(1, ndays + 1)           # 1-based, matches MATLAB
    min_idx = np.linspace(1, ndays, ndays * day_points)
    cs  = CubicSpline(day_idx, w, extrapolate=True)
    out = cs(min_idx)

    return out


def q_day(arr):
    """
    --- MATLAB: Function to calculate actual QDC for a 30 days period: PC_DB_init/q_day.m ---

    Calculate Quiet Day Curve (QDC) and Actual Quiet Day for a 30-day period (older DB-init version).
    """
    arr  = np.array(arr, dtype=np.float64)
    n    = len(arr)
    step = 2
    a    = n // 1440   # number of days

    # 120-point moving average trend (matches smooth(arr, 120) - 'moving')
    arr_s = pd.Series(arr).rolling(120, center=True, min_periods=1).mean().values

    # gradient
    arr_g = np.gradient(arr)

    divq  = 0
    day   = None          # 2-D quiet day matrix, shape (1440, accumulating)
    act   = np.full(50000, np.nan)
    ac    = 0
    p     = np.zeros(1)
    flag  = True

    while np.min(p) < 120:
        divq += step
        p = np.zeros(1320)

        q = np.full(n, np.nan)

        # older inline loop (matches original for i=31:len-31)
        for i in range(30, n - 30):
            if (np.max(np.abs(arr[i - 30:i + 31] - arr_s[i - 30:i + 31])) < divq and
                    np.max(np.abs(arr_g[i - 30:i + 31])) < divq):
                q[i] = arr[i]
                if ac < len(act):
                    act[ac] = i
                    ac += 1

        # append quiet data to day matrix (1440 x days)
        q_mat = q.reshape(a, 1440).T   # (1440, a)
        day   = q_mat if day is None else np.hstack([day, q_mat])

        # number of quiet points within each 2-hour window
        for i in range(1320):
            p[i] = np.sum(~np.isnan(day[i:i + 121, :]))

        # original escape condition
        actual = step / divq
        if actual <= 0.05:
            flag = False
            break

    if flag:
        # MATLAB: qqq=[day;day;day] → (3*1440, ncols), nanmean(qqq') → (3*1440,)
        triple = np.vstack([day, day, day])           # (3*1440, ncols)
        qqq    = np.nanmean(triple, axis=1)           # (3*1440,)

        qqq_s = pd.Series(qqq).rolling(240, center=True, min_periods=1).mean().values
        qqq_s = lowess(qqq_s, np.arange(len(qqq_s)),
                       frac=240 / len(qqq_s), it=3, return_sorted=False)

        qday   = qqq_s[1439:2879]

        ActDay = int(np.round(np.nanmedian(act[:ac]) / 1440))
    else:
        qday   = np.full(1440, np.nan)
        ActDay = np.nan

    return qday, ActDay


def qdc_interp(in_arr):
    """
    --- MATLAB: Function to interpolation/extrapolation of QDC for all days from actual QDC: PC_DB_init/qdc_interp.m ---

    Interpolate / extrapolate QDC for every day from the sparse Actual Day array.
    """
    in_arr = np.array(in_arr, dtype=np.float64)
    ndays  = len(in_arr) // 1440

    # STEP 1 - build 2-D array of Actual Days
    qDay = []
    da   = []
    for i in range(ndays):
        arr_day = in_arr[i * 1440:(i + 1) * 1440]
        if not np.all(np.isnan(arr_day)):
            qDay.append(arr_day)
            da.append(i + 1)   # 1-based day number, matches MATLAB

    if len(da) == 0:
        return in_arr   # nothing to interpolate

    qDay = np.array(qDay).T   # (1440, n_actual)
    da   = np.array(da)
    all_days = np.arange(1, ndays + 1)

    # STEP 2 - nearest-neighbour interpolation + lowess(60) per minute
    from scipy.interpolate import interp1d
    interp_arr = np.full((1440, ndays), np.nan)
    for i in range(1440):
        nn = interp1d(da, qDay[i, :], kind='nearest',
                      fill_value='extrapolate')(all_days)
        interp_arr[i, :] = lowess(nn, all_days,
                                  frac=60 / ndays, it=1, return_sorted=False)

    # STEP 3 - flatten to 1-D then smooth with loess(120)
    l_arr  = interp_arr.T.flatten()   # column-major to match MATLAB reshape
    out    = lowess(l_arr, np.arange(len(l_arr)),
                    frac=120 / len(l_arr), it=1, return_sorted=False)

    return out


def avr5min(arr):
    """
    --- MATLAB: The function avarages 1-minute data to 5 minute: avr5min.m ---

    5-minute block average of a 1-minute array.
    """
    arr   = np.array(arr, dtype=np.float64)
    n5    = len(arr) // 5
    out   = np.array([np.nanmean(arr[i * 5:(i + 1) * 5]) for i in range(n5)])
    return out


def dist_5m(year, station, iaga_dir):
    """
    --- MATLAB: Function to calcuate disturbances in observatory data and 5 minute averaging: STEP1_getdist/dist_5m.m ---

    Compute 5-minute averaged X/Y disturbances for a single year (no overlap).
    """
    start = datetime(year, 1, 1)
    end   = datetime(year, 12, 31)
    ndays = (end - start).days + 1
    n_min = ndays * 1440

    x = np.full(n_min, np.nan)
    y = np.full(n_min, np.nan)

    # read IAGA files (no overlap)
    for d in range(ndays):
        actual = start + timedelta(days=d)
        fname  = os.path.join(iaga_dir, str(actual.year),
                              f'{station}{actual.strftime("%Y%m%d")}dmin.min')
        if not os.path.exists(fname):
            print(f'Warning: No file {os.path.basename(fname)}. Skipping.')
            continue

        with open(fname, 'r') as f:
            for line in f:
                if line.startswith('DATE'):
                    break
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    dt  = datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S')
                    xv  = float(line[30:40])
                    yv  = float(line[40:50])
                    idx = int((dt - start).total_seconds() // 60)
                    if 0 <= idx < n_min:
                        x[idx] = np.nan if xv >= 88888 else float(round(xv))
                        y[idx] = np.nan if yv >= 88888 else float(round(yv))
                except (ValueError, IndexError):
                    continue

    # SS - rounded immediately to match DB write rounding in ss_bd_interp / dbset
    ss_x = np.round(ss_interp(x)).astype(np.float64)
    ss_y = np.round(ss_interp(y)).astype(np.float64)

    # QDC (Actual Days)
    # step through year in 10-day steps, 30-day windows (matches year_actual_qdc / qday_db)
    qdc_x_1min = np.full(n_min, np.nan)
    qdc_y_1min = np.full(n_min, np.nan)

    for s_day in range(0, ndays + 1, 10):
        s_idx = s_day * 1440
        e_idx = min(s_idx + 31 * 1440, n_min)

        by_x = x[s_idx:e_idx] - ss_x[s_idx:e_idx]
        by_y = y[s_idx:e_idx] - ss_y[s_idx:e_idx]

        if len(by_x) < 1440:
            continue

        qday_x, ActDay_x = q_day(by_x)
        qday_y, ActDay_y = q_day(by_y)

        if not np.isnan(ActDay_x):
            j = s_idx + int(ActDay_x - 1) * 1440
            if j + 1440 <= n_min:
                qdc_x_1min[j:j + 1440] = np.round(qday_x)

        if not np.isnan(ActDay_y):
            j = s_idx + int(ActDay_y - 1) * 1440
            if j + 1440 <= n_min:
                qdc_y_1min[j:j + 1440] = np.round(qday_y)

    # QDC interpolation - rounded to match DB write rounding in qdc_db_interp / dbset
    qdc_x = np.round(qdc_interp(qdc_x_1min)).astype(np.float64)
    qdc_y = np.round(qdc_interp(qdc_y_1min)).astype(np.float64)

    # disturbances
    dist_x = x - ss_x - qdc_x
    dist_y = y - ss_y - qdc_y

    # 5-minute averaging
    dist_x_5m = avr5min(dist_x)
    dist_y_5m = avr5min(dist_y)

    dt_5min = pd.date_range(start=start, periods=len(dist_x_5m), freq='5min')

    return {
        'time': np.array(dt_5min),
        'x':    dist_x_5m,
        'y':    dist_y_5m,
    }


def make_dist_5min(years, station, iaga_dir, out_dir, compare=False):
    """
    --- MATLAB: Script to call function for averaging disturbances in observatory data: STEP1_getdist/make_dist_5min.m ---

    Generate and save 5-minute disturbance files for a list of years.
    """
    os.makedirs(out_dir, exist_ok=True)
    for year in years:
        print(f'\n{year}')
        dist = dist_5m(year, station, iaga_dir)
        out_path = os.path.join(out_dir, f'dist_{year}.npz')
        np.savez_compressed(out_path, **dist)
        print(f'Saved {out_path}')

        compare_dist(year)


# %% field

def make_ekl_5min(years, compare=True):

    df_next = pd.DataFrame({'E_R': [np.nan, np.nan, np.nan]},
                       index=pd.date_range(end=f'{years[0]}-01-01 00:00', periods=3, freq='5min'))

    out_dir = DIRECTORIES.get('in')

    for year in years:

        df = import_def_omni(year)
        df.set_index('epoch', inplace=True)

        # Mask fill values before using E_R
        mask = (df['B_y_GSM'] < 100) & (df['B_z_GSM'] < 100) & (df['V_flow'] < 5000)
        df.loc[~mask, 'E_R'] = np.nan

        full_index = pd.date_range(start=f'{year}-01-01 00:00', end=f'{year}-12-31 23:55', freq='5min')

        df = df[['E_R']].resample('5min').mean().reindex(full_index)

        # Shift timestamps forward 15 min
        df.index = df.index + pd.Timedelta(minutes=15)

        in_year = df.index.year == year
        df_year = df.loc[in_year]

        # Prepend the last 15 min of previous year
        df_year = pd.concat([df_next, df_year])

        # Carry forward the 3 bins that shifted into next year
        df_next = df.loc[~in_year]

        out_path = os.path.join(out_dir, f'ekls_{year}.npz')
        np.savez_compressed(out_path, times= df_year.index, E_R=df_year['E_R'].values)

        if compare:
            compare_ekl(year)



# %% main

def main(compare=False):

    if False:

        iaga_dir = DIRECTORIES.get('iaga')
        out_dir  = DIRECTORIES.get('in')

        make_dist_5min(range(1997,2010), 'thl', iaga_dir, out_dir, compare)

    make_ekl_5min(range(1997,2010), compare)

    return

if __name__=='__main__':

    main(True)