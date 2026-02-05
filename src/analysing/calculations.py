# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:11:13 2025

@author: richarj2
"""
import numpy as np
import pandas as pd

from uncertainties import unumpy as unp
from uncertainties import ufloat

import scipy.stats as stats

from ..coordinates.spatial import cartesian_to_spherical, spherical_to_cartesian

# %% Weighted

def average_of_averages(series, series_uncs=None, series_counts=None, mask=None, unit=None):
    """
    Level 1 dispatcher.
    - Standard scalar
    - Circular scalar
    - Vector components
    """

    if isinstance(series, pd.DataFrame) and series.shape[1] == 3:
        return vector_average_of_averages(
            series,
            series_uncs=series_uncs,
            series_counts=series_counts,
            mask=mask
        )

    if unit is None:
        unit = series.attrs.get('units',{}).get(series.name,'')

    if unit in ('rad', 'deg', '°'):
        return circular_average_of_averages(
            series,
            series_uncs=series_uncs,
            series_counts=series_counts,
            mask=mask
        )

    return standard_average_of_averages(
        series,
        series_uncs=series_uncs,
        series_counts=series_counts,
        mask=mask
    )


def standard_average_of_averages(series, series_uncs=None, series_counts=None, mask=None):
    """
    Level 2
    Average of data with counts and/or uncertainties
    """

    if mask is None:
        mask = np.ones(len(series), dtype=bool)

    for data in (series,series_uncs,series_counts):
        if data is not None:
            mask &= ~data.isna()

    if series_counts is not None:
        mask &= series_counts>0

    # Case 1: No counts provided (or all zero)
    if series_counts is None or np.nansum(series_counts.loc[mask]) == 0:
        ufloats = None
        if series_uncs is not None:
            ufloats = series.loc[mask].combine(series_uncs.loc[mask], lambda v, u: ufloat(v, u))
        else:
            ufloats = series.loc[mask]
        return calc_simple_mean_error(ufloats)

    # Extract arrays
    means  = np.array(series.loc[mask], dtype=float)
    counts = np.array(series_counts.loc[mask], dtype=float)

    valid = counts >= 2
    if series_uncs is not None:
        sems = np.array(series_uncs.loc[mask], dtype=float)
        eps = 1e-8
        sems = np.maximum(sems, eps)

    means  = means[valid]
    counts = counts[valid]
    if series_uncs is not None:
        sems = sems[valid]

    if means.size == 0:
        return ufloat(np.nan, np.nan)

    # Case 2: Counts provided, SEMs missing
    if series_uncs is None:
        overall_mean = np.average(means, weights=counts)

        N = np.sum(counts)
        if N <= 1:
            return ufloat(np.nan, np.nan)
        numerator = np.nansum(counts * (means - overall_mean) ** 2)
        pooled_var = numerator / (N - 1)
        overall_sem = np.sqrt(pooled_var / N)
        return ufloat(overall_mean, overall_sem)

    # Case 3: Counts and SEMs both provided
    overall_mean = np.average(means, weights=counts)

    N = np.sum(counts)
    if N <= 1:
        return ufloat(np.nan, np.nan)
    s_i = sems * np.sqrt(counts)   # convert SEMs back to SDs
    numerator = np.nansum((counts - 1) * (s_i ** 2) + counts * (means - overall_mean) ** 2)
    pooled_var = numerator / (N - 1)
    overall_sem = np.sqrt(pooled_var / N)
    return ufloat(overall_mean, overall_sem)


def circular_average_of_averages(series, series_uncs=None, series_counts=None, mask=None):
    """
    Level 2
    Average of circular data with counts and/or uncertainties
    """
    if mask is None:
        mask = np.ones(len(series), dtype=bool)

    for data in (series, series_uncs, series_counts):
        if data is not None:
            mask &= ~data.isna()

    if series_counts is not None:
        mask &= series_counts > 0

    # Extract masked arrays
    angles = np.array(series.loc[mask], dtype=float)
    counts = np.array(series_counts.loc[mask], dtype=float) if series_counts is not None else None
    sems   = np.array(series_uncs.loc[mask], dtype=float) if series_uncs is not None else None

    # Case 1: No counts provided → simple circular mean/error
    if counts is None or np.nansum(counts) == 0:
        ufloats = unp.uarray(angles, sems if sems is not None else np.zeros_like(angles))
        return calc_circular_mean_error(ufloats)

    # Case 2: Counts provided, no SEMs
    if sems is None:
        mean = circular_avg(angles, weights=counts)
        sem  = circular_sem(angles, weights=counts)
        return ufloat(mean, sem)

    # Case 3: Counts and SEMs both provided
    stds = sems * np.sqrt(counts)
    eps = 1e-8
    weights = 1.0 / (stds**2 + eps)
    weights = np.where(np.isfinite(weights), weights, 0.0)

    mean = circular_avg(angles, weights=weights)
    sem  = circular_sem(angles, weights=weights)
    return ufloat(mean, sem)

# %% Mean_error

def calc_mean_error(series, start=None, end=None, unit=None):
    """
    Level 2
    Mean and sem of data (wrapper)
    Decides if circular or standard functions
    """
    if start:
        series = series[series.index >= pd.Timestamp(start)]
    if end:
        series = series[series.index <= pd.Timestamp(end)]

    if unit is None:
        unit = series.attrs.get('units', {}).get(series.name, None)

    series = series.dropna().to_numpy()

    if len(series)==0:
        return np.nan

    if unit in ('rad', 'deg', '°'):
        if unit in ('deg', '°'):
            series = np.radians(series)

        val_unc = calc_circular_mean_error(series)

        if unit in ('deg', '°'):
            return ufloat(np.degrees(val_unc.n),np.degrees(val_unc.s))

        return val_unc

    return calc_simple_mean_error(series)

def calc_simple_mean_error(data):
    """
    Level 3
    Mean and sem of non-angular data
    Decides if with or without uncertainties
    """
    data = np.asarray(data)
    if data.size == 0:
        return ufloat(np.nan, np.nan)

    try:  # check if data has errors
        errors = unp.std_devs(data)

        if np.all(errors == 0):
            data = unp.nominal_values(data)
            raise Exception

    except:
        mean = np.mean(data)
        err  = sem(data) if data.size > 0 else np.nan
        return ufloat(mean, err)

    data = unp.nominal_values(data)
    eps = 1e-8
    weights = 1.0 / (errors**2 + eps)
    weights = np.where(np.isfinite(weights), weights, 0.0)  # handle inf/nan

    if np.sum(weights) == 0:
        return ufloat(np.nan, np.nan)

    mean = np.average(data, weights=weights)
    err  = np.sqrt(1 / np.sum(weights))

    return ufloat(mean, err)

# %% Standard_deviation

def sem(series, nu=1):
    """
    Level 4
    sem of non-angular data
    """
    if len(series)<=nu:
        return 0

    # Standard Error of the Mean using sample std
    std = np.std(series, ddof=nu)
    sqrt_count = np.sqrt(np.size(series))
    return std / sqrt_count

def calc_sample_std(series, start=None, end=None, unit=None):
    """
    Level 2
    Sample standard deviation of data (wrapper)
    Decides if circular or standard functions
    """
    if start:
        series = series[series.index >= pd.Timestamp(start)]
    if end:
        series = series[series.index <= pd.Timestamp(end)]

    if unit is None:
        unit = series.attrs.get('units', {}).get(series.name, None)

    series = series.dropna().to_numpy()

    if len(series)<=1:
        return 0

    if unit in ('rad', 'deg', '°'):
        if unit in ('deg', '°'):
            series = np.radians(series)

        unc = calc_circular_std(series)

        if unit in ('deg', '°'):
            return np.degrees(unc)

        return unc

    return calc_weighted_std(series)

def calc_weighted_std(data):
    """
    Level 3
    Sample standard deviation of non-angular data
    Decides if with or without uncertainties
    """
    try: # check if data has errors
        errors  = unp.std_devs(data)

        if np.all(errors == 0): # All 0
            data = unp.nominal_values(data)
            raise Exception

    except: # No uncertainties
        return np.std(data, ddof=1)

    # Weighted
    data    = unp.nominal_values(data)
    safe_errors = np.where(errors == 0, np.nan, errors)
    weights = 1 / safe_errors**2
    weights = np.where(np.isfinite(weights), weights, 0.0)  # handle inf/nan

    mean  = np.average(data, weights=weights)
    var_b = np.average((data - mean)**2, weights=weights) # biased

    N_eff = (np.sum(weights))**2 / np.sum(weights**2)
    var = var_b * N_eff / (N_eff - 1)

    return np.sqrt(var)

def std_of_averages(series, series_uncs=None, series_counts=None, mask=None):
    """
    Level 1
    Sample standard deviation of data with counts
    """
    if mask is None:
        mask = np.ones(len(series), dtype=bool)

    for data in (series, series_uncs, series_counts):
        if data is not None:
            mask &= ~np.isnan(data)

    # Extract arrays
    values = np.array(series.loc[mask], dtype=float)
    counts = np.array(series_counts.loc[mask], dtype=float) if series_counts is not None else None
    sems   = np.array(series_uncs.loc[mask], dtype=float) if series_uncs is not None else None

    # Case 1: No counts → simple standard deviation
    if counts is None or np.nansum(counts) == 0:
        if sems is not None:
            ufloats = unp.uarray(values, sems)
            return calc_weighted_std(ufloats)
        return calc_sample_std(values)

    # Case 2: Counts provided, SEMs missing → weighted std
    if sems is None:
        return calc_weighted_std(unp.uarray(values, 1/np.sqrt(counts)))  # treat counts as inverse-variance weights

    # Case 3: Counts and SEMs both provided → combine uncertainties
    s_i = sems * np.sqrt(counts)   # convert SEMs to SDs
    ufloats = unp.uarray(values, s_i)
    return calc_weighted_std(ufloats)

# %% Circular statistics

def calc_circular_mean_error(data):
    """
    Level 3
    Mean and sem of angular data
    Decides if with or without uncertainties
    """
    try: # check if data has errors
        errors  = unp.std_devs(data)

        if np.all(errors == 0): # All 0
            data = unp.nominal_values(data)
            raise Exception

    except: # No uncertainties
        mean = circular_mean(data)
        err  = circular_stddev(data) / np.sqrt(len(data))

        return ufloat(mean, err)

    # Weighted
    data    = unp.nominal_values(data)
    safe_errors = np.where(errors == 0, np.nan, errors)
    weights = 1 / safe_errors**2
    weights = np.where(np.isfinite(weights), weights, 0.0)  # handle inf/nan

    mean = circular_avg(data, weights)
    err  = circular_sem(data, weights)

    return ufloat(mean, err)

def calc_circular_std(data):
    """
    Level 3
    Sample standard deviation of angular data
    Decides if with or without uncertainties
    """
    try: # check if data has errors
        errors  = unp.std_devs(data)

        if np.all(errors == 0): # All 0
            data = unp.nominal_values(data)
            raise Exception

    except: # No uncertainties
        return circular_stddev(data)

    data = unp.nominal_values(data)
    weights = 1 / errors**2
    weights = np.where(np.isfinite(weights), weights, 0.0)  # handle inf/nan

    return circular_std(data, weights)

def circular_avg(angles, weights=None):
    """
    Weighted average of circular data
    """
    if weights is None:
        weights = np.ones_like(angles)

    return np.arctan2(np.sum(weights*np.sin(angles)), np.sum(weights*np.cos(angles)))

def circular_sem(angles, weights=None):
    """
    Weighted sem of circular data
    """
    std = circular_std(angles, weights=weights)

    n_eff = (np.sum(weights)**2) / np.sum(weights**2)

    return std / np.sqrt(n_eff)

def circular_std(angles, weights=None):
    """
    Weighted sample standard deviation of circular data
    """
    if weights is None:
        weights = np.ones_like(angles)

    R = np.sqrt((np.sum(weights*np.cos(angles)))**2 + (np.sum(weights*np.sin(angles)))**2) / np.sum(weights)

    return np.sqrt(-2 * np.log(R))

def circular_mean(angles):

    return stats.circmean(angles, low=-np.pi, high=np.pi)

def circular_variance(angles):

    # n = len(angles)
    # R = np.sqrt((np.sum(np.cos(angles)) / n)**2 + (np.sum(np.sin(angles)) / n)**2)
    # return 1-R

    return stats.circvar(angles, low=-np.pi, high=np.pi)


def circular_stddev(angles):

    # n = len(angles)
    # R = np.sqrt((np.sum(np.cos(angles)) / n)**2 + (np.sum(np.sin(angles)) / n)**2)
    # return sqrt(-2lnR)

    return stats.circstd(angles, low=-np.pi, high=np.pi)

# %% Vectors

def vec_mag(vec):

    try:
        mag = np.linalg.norm(vec)
    except:
        mag = unp.sqrt(np.sum(vec**2))

    return mag.item() if hasattr(mag, 'item') else mag


def calc_average_vector(df_vec, param=None):

    if len(df_vec)==0:
        #return unp.uarray([np.nan, np.nan, np.nan], [0, 0, 0])
        return unp.uarray([], [])

    x_label = 'x'
    y_label = 'y'
    z_label = 'z'
    if param is not None:
        vec, coords = param.split('_')
        x_label = f'{vec}_x_{coords}'
        if x_label not in df_vec:
            x_label = f'{vec}_x_GSE'
        y_label = f'{vec}_y_{coords}'
        z_label = f'{vec}_z_{coords}'

    not_nan = np.isfinite(df_vec[[x_label, y_label, z_label]].to_numpy()).all(axis=1)

    if np.sum(not_nan)==0:
        return unp.uarray([], [])

    x_data = df_vec.loc[not_nan,x_label].to_numpy()
    y_data = df_vec.loc[not_nan,y_label].to_numpy()
    z_data = df_vec.loc[not_nan,z_label].to_numpy()

    if len(x_data)==1:
        return unp.uarray([x_data[0],y_data[0],z_data[0]], [0, 0, 0])

    r, theta, phi = cartesian_to_spherical(x_data, y_data, z_data)

    r_avg     = calc_simple_mean_error(r)
    theta_avg = calc_circular_mean_error(theta)
    phi_avg   = calc_circular_mean_error(phi)

    x, y, z = spherical_to_cartesian(r_avg, theta_avg, phi_avg)

    uarr = unp.uarray([x.n, y.n, z.n],[x.s, y.s, z.s])

    return uarr


def calc_angle_between_vecs(df, vec1, vec2):

    vec1_field, vec1_coords = vec1.split('_')[0], '_'.join(vec1.split('_')[1:])
    vec2_field, vec2_coords = vec2.split('_')[0], '_'.join(vec2.split('_')[1:])


    vec1_arr = df[[f'{vec1_field}_{comp}_{vec1_coords}' for comp in ('x','y','z')]].to_numpy()
    vec2_arr = df[[f'{vec2_field}_{comp}_{vec2_coords}' for comp in ('x','y','z')]].to_numpy()

    dot = np.einsum('ij,ij->i', vec1_arr, vec2_arr)

    vec1_norm = np.linalg.norm(vec1_arr, axis=1)
    vec2_norm = np.linalg.norm(vec2_arr, axis=1)

    return np.arccos(np.clip(dot / (vec1_norm * vec2_norm), -1.0, 1.0))

def vector_average_of_averages(series, series_uncs=None, series_counts=None, mask=None):
    """
    Level 2
    Average of vector data with counts and/or uncertainties
    """
    if mask is None:
        mask = np.ones(len(series), dtype=bool)

    # Mask invalid rows
    for data in (series, series_uncs, series_counts):
        if data is not None:
            mask &= ~series.isna().any(axis=1) if isinstance(series, pd.DataFrame) else ~data.isna()

    if series_counts is not None:
        mask &= series_counts > 0

    filtered_series = series.loc[mask]
    filtered_uncs = series_uncs.loc[mask] if series_uncs is not None else None
    filtered_counts = series_counts.loc[mask] if series_counts is not None else None

    if len(filtered_series) == 0:
        return unp.uarray([], [])

    # Convert to ufloats element-wise
    if filtered_uncs is not None:
        uarrays = [unp.uarray(filtered_series.iloc[:, i].values, filtered_uncs.iloc[:, i].values) for i in range(filtered_series.shape[1])]
    else:
        uarrays = [unp.uarray(filtered_series.iloc[:, i].values, np.zeros(filtered_series.shape[0])) for i in range(filtered_series.shape[1])]

    x_data, y_data, z_data = uarrays

    # Single vector → return as-is
    if len(x_data) == 1:
        return np.array([x_data[0], y_data[0], z_data[0]], dtype=object)

    # Convert to spherical coordinates (lists of ufloats)
    r = unp.sqrt(x_data**2 + y_data**2 + z_data**2)
    theta = unp.arccos(z_data / r)
    phi = unp.arctan2(y_data, x_data)

    r_nom, r_std            = unp.nominal_values(r), unp.std_devs(r)
    theta_nom, theta_std    = unp.nominal_values(theta), unp.std_devs(theta)
    phi_nom, phi_std        = unp.nominal_values(phi), unp.std_devs(phi)

    # Average with counts and uncertainties
    r_avg     = standard_average_of_averages(pd.Series(r_nom, index=filtered_series.index), series_uncs=pd.Series(r_std, index=filtered_series.index), series_counts=filtered_counts)
    theta_avg = circular_average_of_averages(pd.Series(theta_nom, index=filtered_series.index), series_uncs=pd.Series(theta_std, index=filtered_series.index), series_counts=filtered_counts)
    phi_avg   = circular_average_of_averages(pd.Series(phi_nom, index=filtered_series.index), series_uncs=pd.Series(phi_std, index=filtered_series.index), series_counts=filtered_counts)

    # Convert back to Cartesian with propagated uncertainties
    x_avg = r_avg * unp.sin(theta_avg) * unp.cos(phi_avg)
    y_avg = r_avg * unp.sin(theta_avg) * unp.sin(phi_avg)
    z_avg = r_avg * unp.cos(theta_avg)

    return unp.uarray([x_avg.n, y_avg.n, z_avg.n],
                      [x_avg.s, y_avg.s, z_avg.s])

# %% Statistics

def percentile_func(series, p=50):
    return np.percentile(series, p)

def mad(series):
    # Median of the absolute deviations from the median
    median_val = np.median(series)
    return np.median(np.abs(series - median_val))

def iqr(series):
    Q3 = percentile_func(series, 75)
    Q1 = percentile_func(series, 25)
    return Q3 - Q1

def kps(series, nu=1):
    # Karl Pearson Second Skewness Coefficient
    # skewness = 3 x (mean - median) / std
    mean = np.mean(series)
    median = np.median(series)
    std = np.std(series, ddof=nu)  # Using sample standard deviation (ddof=1)
    return 3 * (mean - median) / std

def median_with_counts(series, counts=None, mask=None):

    if mask is not None:
        try:
            series = series.loc[mask].to_numpy()
            if counts is not None:
                counts = counts.loc[mask].to_numpy()
        except:
            series = series[mask]
            if counts is not None:
                counts = counts[mask]

    if counts is None:
        q1 = np.percentile(series,25)
        median = np.percentile(series,50)
        q3 = np.percentile(series,75)

    else:
        idx = np.argsort(series)
        series = series[idx]
        counts = counts[idx]

        cum_counts = np.cumsum(counts)
        total = cum_counts[-1]

        def percentile(p):
            target = p * total
            return series[np.searchsorted(cum_counts, target)]

        median = percentile(0.5)
        q1 = percentile(0.25)
        q3 = percentile(0.75)

    return median, q1, q3