# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:11:13 2025

@author: richarj2
"""
import numpy as np
import pandas as pd

from uncertainties import unumpy as unp
from uncertainties import ufloat


def calc_mean_error(series, start=None, end=None):

    if start:
        series = series[series.index >= pd.Timestamp(start)]
    if end:
        series = series[series.index <= pd.Timestamp(end)]

    unit = series.attrs.get('units', {}).get(series.name, None)

    series = series.dropna().to_numpy()

    if unit in ('rad', 'deg', '°'):
        if unit in ('deg', '°'):
            series = np.radians(series)

        unc = calc_circular_mean_error(series)

        if unit in ('deg', '°'):
            return np.degrees(unc)

        return unc

    return calc_simple_mean_error(series)


def calc_simple_mean_error(data):

    try: # check if data has errors
        errors  = unp.std_devs(data)
        weights = 1 / errors**2
        weights = np.where(np.isfinite(weights), weights, 0.0)  # handle inf/nan

        data    = unp.nominal_values(data)
        err     = np.sqrt(1 / np.sum(weights))

    except:
        weights = np.ones_like(data)
        err     = sem(data)

    mean = np.average(data, weights)
    return ufloat(mean, err)

def calc_circular_mean_error(data):

    try: # check if data has errors
        errors  = unp.std_devs(data)
        weights = 1 / errors**2
        weights = np.where(np.isfinite(weights), weights, 0.0)  # handle inf/nan

        data    = unp.nominal_values(data)

    except:
        weights = np.ones_like(data)

    mean = circular_mean(data, weights)
    err = circular_sem(data, weights)

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


def circular_variance(angles, dof=1):
    n = len(angles) - dof
    R = np.sqrt((np.sum(np.cos(angles)) / n)**2 + (np.sum(np.sin(angles)) / n)**2)
    return 1 - R

def vec_mag(vec):

    try:
        mag = np.linalg.norm(vec)
    except:
        mag = unp.sqrt(np.sum(vec**2))

    return mag.item() if hasattr(mag, 'item') else mag


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

def sem(series, nu=1):
    # Standard Error of the Mean using sample std
    std = np.std(series, ddof=nu)
    sqrt_count = np.sqrt(np.size(series))
    return std / sqrt_count

def kps(series, nu=1):
    # Karl Pearson Second Skewness Coefficient
    # skewness = 3 x (mean - median) / std
    mean = np.mean(series)
    median = np.median(series)
    std = np.std(series, ddof=nu)  # Using sample standard deviation (ddof=1)
    return 3 * (mean - median) / std
