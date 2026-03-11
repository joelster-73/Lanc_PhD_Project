# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:21:59 2026

@author: richarj2
"""

import os
import numpy as np
import pandas as pd

from statsmodels.nonparametric.smoothers_lowess import lowess

import calendar
from datetime import datetime, timedelta
from config import DIRECTORIES

from stauning_imports import import_coeff

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='numpy')

# %% process_mag_data

def process_mag_data(station='thl', t1=1997, t2=2009, overlap=31):
    """
    --- MATLAB: Definitive_PC_Index.m (SS and QDC sections) ---
    Precompute SS and QDC from IAGA data for each year.
    Run once — independent of coefficients.
    Saves SS and QDC to DIRECTORIES.get('prelim')/ss/ and DIRECTORIES.get('prelim')/qdc/
    """
    ss_dir  = os.path.join(DIRECTORIES.get('prelim'), 'ss')
    qdc_dir = os.path.join(DIRECTORIES.get('prelim'), 'qdc')
    dist_dir = os.path.join(DIRECTORIES.get('prelim'), 'dist')
    os.makedirs(ss_dir,  exist_ok=True)
    os.makedirs(qdc_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)

    for year in range(t1, t2 + 1):
        print(f'\nProcessing mag data for {year}')

        ndays   = sum(calendar.monthrange(year, m)[1] for m in range(1, 13)) + overlap * 2
        t_start = datetime(year, 1, 1) - timedelta(days=overlap)
        dt_arr  = pd.date_range(start=t_start, periods=ndays * 1440, freq='1min')

        data = {
            'x': np.full(ndays * 1440, np.nan),
            'y': np.full(ndays * 1440, np.nan),
        }

        # read IAGA files
        df = read_iaga(station, year, DIRECTORIES.get('iaga'), overlap)
        for col in ('x', 'y'):
            idx   = np.searchsorted(dt_arr, df.index)
            valid = (idx >= 0) & (idx < len(dt_arr))
            data[col][idx[valid]] = df[col].values[valid]

        # calculate SS
        ss = {}
        for col in ('x', 'y'):
            arr     = data[col].reshape(ndays, 1440)
            day_med = np.nanmedian(arr, axis=1)

            # 7-day moving average
            w = pd.Series(day_med).rolling(7, center=True, min_periods=1).mean().values
            # 7-day robust loess
            w = lowess(w, np.arange(ndays), frac=7/ndays, it=3, return_sorted=False)

            # interpolate to 1-minute resolution
            day_idx = np.arange(ndays)
            min_idx = np.linspace(0, ndays - 1, ndays * 1440)
            ss[f'ss_{col}'] = np.interp(min_idx, day_idx, w)

        np.savez_compressed(os.path.join(ss_dir, f'ss_{year}.npz'),
                 time=np.array(dt_arr),
                 **ss)
        print(f'Saved SS for {year}')

        # calculate QDC
        year_start_idx = overlap * 1440
        year_end_idx   = (ndays - overlap) * 1440
        qdc = {
            'qdc_x': np.full(ndays * 1440, np.nan),
            'qdc_y': np.full(ndays * 1440, np.nan),
        }

        for col in ('x', 'y'):
            qdc_col = f'qdc_{col}'

            for s in range(0, year_end_idx - year_start_idx, 10 * 1440):
                abs_s = year_start_idx - overlap * 1440 + s
                abs_e = min(abs_s + 30 * 1440, len(data[col]))

                by = data[col][abs_s:abs_e] - ss[f'ss_{col}'][abs_s:abs_e]
                if len(by) < 1440:
                    continue

                qday, ActDay = q_day(by)

                if not np.isnan(ActDay):
                    j_start = abs_s + int(ActDay - 1) * 1440
                    j_end   = j_start + 1440
                    if j_end <= len(qdc[qdc_col]):
                        qdc[qdc_col][j_start:j_end] = qday

            # interpolate QDC across all days
            qdc_days = qdc[qdc_col].reshape(ndays, 1440)
            ActDays  = np.where(~np.all(np.isnan(qdc_days), axis=1))[0]

            if len(ActDays) == 0:
                continue

            qdays_mat  = qdc_days[ActDays, :]
            interp_arr = np.full((ndays, 1440), np.nan)
            all_days   = np.arange(ndays)

            for i in range(1440):
                interp_arr[:, i] = np.interp(all_days, ActDays, qdays_mat[:, i])
                interp_arr[:, i] = lowess(interp_arr[:, i], all_days,
                                          frac=60/ndays, it=1, return_sorted=False)

            l_arr = interp_arr.flatten()
            qdc[qdc_col] = lowess(l_arr, np.arange(len(l_arr)),
                                  frac=120/len(l_arr), it=1, return_sorted=False)

        np.savez_compressed(os.path.join(qdc_dir, f'qdc_{year}.npz'),
                 time=np.array(dt_arr),
                 **qdc)
        print(f'Saved QDC for {year}')

        # disturbance data

        dist = {}
        for col in ('x', 'y'):
            dist[f'dist_{col}'] = data[col] - qdc[f'qdc_{col}'] - ss[f'ss_{col}']

        np.savez(os.path.join(dist_dir, f'dist_{year}.npz'),
                 time=np.array(dt_arr),
                 **dist)
        print(f'Saved disturbance for {year}')

# %% q_day

def q_day(arr):
    """
    --- MATLAB: q_day.m ---
    Calculate Quiet Day Curve (QDC) and Actual Quiet Day for a 30-day period.
    arr - 1-minute values for 30 days = 43200 elements
    Returns:
        qday   - 1440 element array of QDC values for the actual quiet day
        ActDay - index of the actual quiet day within the 30-day period
    """
    arr = np.array(arr, dtype=np.float64)

    # absolute deviations from 120-min moving average trend
    trend = pd.Series(arr).rolling(120, center=True, min_periods=1).mean().values
    arr_d = np.abs(arr - trend)

    # absolute gradients
    arr_g = np.abs(np.gradient(arr))

    # shift 61-min window by 1 min, find max deviations and gradients
    n = len(arr)
    max_ds = np.full(n - 60, np.nan)
    max_g  = np.full(n - 60, np.nan)
    for i in range(30, n - 30):
        max_ds[i - 30] = np.max(np.abs(arr_d[i - 30:i + 31]))
        max_g[i - 30]  = np.max(np.abs(arr_g[i - 30:i + 31]))

    # initialise quiet limit search
    divq = 0
    day  = None
    act  = np.full(len(arr) * 30, np.nan)
    ac   = 0
    p    = np.zeros(1)
    flag = True

    ndays = n // 1440

    while np.min(p) < 120:
        divq += 2
        if divq >= 40:
            flag = False
            break

        p = np.zeros(1320)

        # find data within quiet limit
        q = np.full(n, np.nan)
        idx = np.where((max_ds < divq) & (max_g < divq))[0]
        q[idx + 30] = arr[idx + 30]

        # append quiet data to day matrix
        q_mat = q.reshape(ndays, 1440).T  # shape (1440, ndays)
        day = q_mat if day is None else np.hstack([day, q_mat])

        # quiet minutes
        act[ac:ac + len(idx)] = idx + 30
        ac += len(idx)

        # find number of quiet points within 2-hour periods
        for i in range(1320):
            p[i] = np.sum(~np.isnan(day[i:i + 121, :]))

    if flag:
        # repeat quiet day and calculate mean for every quiet minute
        with np.errstate(all='ignore'):
            qqq = np.nanmean(np.vstack([day.T, day.T, day.T]), axis=0)  # (1440,)

        # smooth QDC
        qqq_s = pd.Series(qqq).rolling(240, center=True, min_periods=1).mean().values
        qqq_s = lowess(qqq_s, np.arange(len(qqq_s)), frac=240/len(qqq_s),
                       it=1, return_sorted=False)

        # QDC covering one day
        qday = qqq_s[1439:2879]

        # actual quiet day
        ActDay = int(np.round(np.nanmedian(act[:ac]) / 1440))
    else:
        qday   = np.full(1440, np.nan)
        ActDay = np.nan

    return qday, ActDay


# %% read_iaga

def read_iaga(station, year, iaga_dir, overlap=31):
    """
    --- MATLAB: Definitive_PC_Index.m (IAGA reading section) ---
    Read IAGA 2002 .min files for a year +/- overlap days into a DataFrame.
    Values >= 88888 are set to NaN (missing data flag).
    """
    start = datetime(year, 1, 1) - timedelta(days=overlap)
    end   = datetime(year, 12, 31) + timedelta(days=overlap)
    ndays = (end - start).days + 1

    records = []
    for d in range(ndays):
        actual = start + timedelta(days=d)
        fname  = os.path.join(DIRECTORIES.get('iaga'), str(actual.year),
                              f'{station}{actual.strftime("%Y%m%d")}dmin.min')
        if not os.path.exists(fname):
            print(f'Warning: No file {os.path.basename(fname)}. Skipping.')
            continue

        with open(fname, 'r') as f:
            # skip header until DATE line
            for line in f:
                if line.startswith('DATE'):
                    break
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    dt  = datetime.strptime(line[:19], '%Y-%m-%d %H:%M:%S')
                    x   = float(line[30:40])
                    y   = float(line[40:50])
                    records.append({
                        'time': dt,
                        'x':    np.nan if x >= 88888 else x,
                        'y':    np.nan if y >= 88888 else y,
                    })
                except (ValueError, IndexError):
                    continue

    df = pd.DataFrame(records).set_index('time')
    return df

# %% calc_def_PC

def calc_def_PC(station='thl', year=1997, coeff=None, overlap=31, source='original'):
    """
    --- MATLAB: Definitive_PC_Index.m (PC index section) ---
    Load precomputed SS and QDC, calculate PC index using coeff.
    Run multiple times with different coeff realisations for uncertainty quantification.
    Saves PC index to pcn/ folder.
    """
    dist_dir = os.path.join(DIRECTORIES.get('prelim'), 'dist')

    # output path
    if source == 'original':
        path = DIRECTORIES.get('coeff')
    else:
        path = DIRECTORIES.get(source)

    pcn_dir = os.path.join(path, 'pcn')
    os.makedirs(pcn_dir, exist_ok=True)

    # load coefficients if not passed in
    if coeff is None:
        coeff = np.load(os.path.join(path, 'coeff.npz'))

    lon = {'thl': 291.0, 'vos': 106.9}[station]

    print(f'\nCalculating PC index ({station}) for {year}')

    ndays = sum([calendar.monthrange(year, m)[1] for m in range(1, 13)]) + overlap * 2
    t_start = datetime(year, 1, 1) - timedelta(days=overlap)
    dt_arr  = pd.date_range(start=t_start, periods=ndays * 1440, freq='1min')

    # load precomputed SS, QDC, and disturbance
    dist = np.load(os.path.join(dist_dir, f'dist_{year}.npz'))
    dist_x = dist['dist_x']
    dist_y = dist['dist_y']

    # year slice only (no overlap)
    i           = np.arange(overlap * 1440, (ndays - overlap) * 1440)
    hours       = dt_arr[i].hour
    minutes     = dt_arr[i].minute
    min_in_year = np.arange(1, len(i) + 1)
    index       = min_in_year - 1

    UT        = hours * 15.0 + minutes * 0.25
    angle     = np.deg2rad(lon + coeff['phi'][index] + UT)
    H_proj    = dist_x[i] * np.sin(angle) - dist_y[i] * np.cos(angle)

    pc     = (H_proj - coeff['b'][index]) / coeff['a'][index]
    pc     = np.round(pc, 2)
    pc_unc = calc_pc_unc(index, dist_x[i], dist_y[i], angle, H_proj, coeff)

    # save
    out_path = os.path.join(pcn_dir, f'pc_{year}.npz')
    np.savez_compressed(out_path,
             time    = np.array(dt_arr[i]),
             dist_x  = dist_x[i],
             dist_y  = dist_y[i],
             pcn     = pc,
             pcn_unc = pc_unc)

    print(f'Saved PC index to {out_path}')

    print_pc(pc, pc_unc)

def calc_pc_unc(index, dist_x, dist_y, angle_rad, H_proj, coeff):

    a = coeff['a'][index]

    n = len(a)
    zeros = np.zeros(n)

    a_var = coeff['a_var'][index] if 'a_var' in coeff else zeros
    b_var = coeff['b_var'][index] if 'b_var' in coeff else zeros
    covar = coeff['cov_ab'][index] if 'cov_ab' in coeff else zeros
    p_var = coeff['phi_var'][index] if 'phi_var' in coeff else zeros  # degrees²

    # if no uncertainty fields at all, return zeros rather than running formula
    if not any(k in coeff for k in ('a_var', 'b_var', 'cov_ab', 'phi_var')):
        return zeros

    dH_dphi  = dist_x * np.cos(angle_rad) + dist_y * np.sin(angle_rad)
    dPC_dphi = dH_dphi / a
    dPC_db   = -1.0 / a
    dPC_da   = -H_proj / a**2

    var_pc = (dPC_dphi**2 * p_var * (np.pi / 180)**2
            + dPC_db**2   * b_var
            + dPC_da**2   * a_var
            + 2 * dPC_da  * dPC_db * covar)

    return np.sqrt(np.abs(var_pc))

def print_pc(pc, pc_unc):

    df = pd.DataFrame({'pcn': pc, 'unc': pc_unc, 'rel': pc_unc/np.abs(pc)*100})
    rows = [df.iloc[(df['pcn'] - q).abs().argmin()] for q in df['pcn'].quantile([0, 0.25, 0.5, 0.75, 1])]
    print(pd.DataFrame(rows, index=['min','Q25','med','Q75','max']))
    print(f'Median relative uncertainty:    {df["rel"].median():.3g} %')
    print(f'Maximum absolute uncertainty:   {df["unc"].max():.3g} mV/m')

# %% main

def main(year=None, source='staun_proj'):
    print()
    print('-----------------------------------')
    print('Calculates the PCN index using the phi, a, b coefficients')
    print(f'Uses the {source.upper()} best monthly a/b/phi every 1-minute of every year')
    if source=='original':
        print('    I.e. a/b/phi calculated by Stauning')
    else:
        print('    I.e. a/b/phi calculated by me')
    print()

    coeff = import_coeff(None, source)

    print(pd.DataFrame(coeff))

    if year is None:
        year_range = range(1997,2022)
    else:
        year_range = range(year,year+1)

    for year in year_range:

        calc_def_PC(station='thl', year=year, coeff=coeff, overlap=31, source=source)


if __name__ == '__main__':

    if True:
        process_mag_data(station='thl', t1=1997, t2=2021, overlap=31)

    # # uses MATLAB a/b/phi for every year to calculate pcn
    if True:
        main(source='original')

    # uses MATLAB H projections to calculate a/b for every year then stack
    main(source='staun_proj')

    # use MATLAB phi to calculate H projections and thus a/b
    main(source='staun_phi')

    # use recreated phi to calculated a/b/phi and thus pcn and thus a/b
    main(source='recreated_phi')

    # use updated phi to calculated a/b/phi and thus pcn and thus a/b
    #main('updated_phi')


