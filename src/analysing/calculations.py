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


# %% Mean_error

def average_of_averages(means, counts=None, sems=None):

    if counts is None:
        return calc_mean_error(means)

    means = np.array(means, dtype=float)
    sems = np.array(sems, dtype=float)
    counts = np.array(counts, dtype=float)

    overall_mean = np.average(means, weights=counts)

    if sems is None:
        return ufloat(overall_mean,0)

    N = counts.sum()
    s_i = sems * np.sqrt(counts)
    numerator = np.sum((counts - 1) * (s_i ** 2) + counts * (means - overall_mean) ** 2)
    pooled_var = numerator / (N - 1)

    overall_sem = np.sqrt(pooled_var / N)

    return ufloat(overall_mean, overall_sem)


def calc_mean_error(series, start=None, end=None, unit=None):

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

    try: # check if data has errors
        errors  = unp.std_devs(data)

        if np.all(errors == 0):
            mean    = np.mean(data)
            err     = sem(data)

        else:
            weights = 1 / errors**2
            weights = np.where(np.isfinite(weights), weights, 0.0)  # handle inf/nan

            data    = unp.nominal_values(data)
            err     = np.sqrt(1 / np.sum(weights))
            mean    = np.average(data, weights)

    except:
        mean    = np.mean(data)
        err     = sem(data)

    return ufloat(mean, err)

def calc_sample_std(series, start=None, end=None, unit=None):

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

        unc = circular_stddev(series)

        if unit in ('deg', '°'):
            return np.degrees(unc)

        return unc

    return np.std(series,ddof=1)

def sem(series, nu=1):

    if len(series)<=nu:
        return 0

    # Standard Error of the Mean using sample std
    std = np.std(series, ddof=nu)
    sqrt_count = np.sqrt(np.size(series))
    return std / sqrt_count

# %% Circular statistics

def calc_circular_mean_error(data):

    try: # check if data has errors
        errors  = unp.std_devs(data)

        if np.all(errors == 0):
            mean = stats.circmean(data)
            err  = circular_stddev(data) / np.sqrt(len(data))

        else:
            weights = 1 / errors**2
            weights = np.where(np.isfinite(weights), weights, 0.0)  # handle inf/nan

            data    = unp.nominal_values(data)

            mean = circular_mean(data, weights)
            err = circular_sem(data, weights)

    except:
        mean = stats.circmean(data)
        err  = circular_stddev(data) / np.sqrt(len(data))

    return ufloat(mean, err)

def circular_mean(angles, weights=None):

    if weights is None:
        weights = np.ones_like(angles)

    return np.arctan2(np.sum(weights*np.sin(angles)), np.sum(weights=np.cos(angles)))


def circular_sem(angles, weights=None):

    if weights is None:
        weights = np.ones_like(angles)

    R = np.sqrt((np.sum(weights*np.cos(angles)))**2 + (np.sum(weights*np.sin(angles)))**2) / np.sum(weights)

    std = np.sqrt(-2 * np.log(R))

    n_eff = (np.sum(weights)**2) / np.sum(weights**2)

    return std / np.sqrt(n_eff)


def circular_variance(angles):

    # n = len(angles)
    # R = np.sqrt((np.sum(np.cos(angles)) / n)**2 + (np.sum(np.sin(angles)) / n)**2)
    # return 1-R

    return stats.circvar(angles, low=-np.pi, high=np.pi)


def circular_stddev(angles):

    # n = len(angles)
    # R = np.sqrt((np.sum(np.cos(angles)) / n)**2 + (np.sum(np.sin(angles)) / n)**2)
    # return sqrt(1-2lnR)

    return stats.circstd(angles, low=-np.pi, high=np.pi)

# %% Vectors

def vec_mag(vec):

    try:
        mag = np.linalg.norm(vec)
    except:
        mag = unp.sqrt(np.sum(vec**2))

    return mag.item() if hasattr(mag, 'item') else mag


def calc_average_vector(df_vec, param=None):

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

    x_data = df_vec.loc[:,x_label].to_numpy()
    y_data = df_vec.loc[:,y_label].to_numpy()
    z_data = df_vec.loc[:,z_label].to_numpy()

    if len(df_vec)>1:

        r, theta, phi = cartesian_to_spherical(x_data, y_data, z_data)

        r_avg = calc_simple_mean_error(r)
        theta_avg = calc_circular_mean_error(theta)
        phi_avg = calc_circular_mean_error(phi)

        x, y, z = spherical_to_cartesian(r_avg, theta_avg, phi_avg)

        uarr = unp.uarray([x.n, y.n, z.n],
                      [x.s, y.s, z.s])

        return uarr

    return unp.uarray(df_vec[[x_label,y_label,z_label]].to_numpy(), [0, 0, 0])


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
