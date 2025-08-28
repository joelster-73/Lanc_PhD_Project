# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:11:13 2025

@author: richarj2
"""
import numpy as np
import pandas as pd

from uncertainties import unumpy as unp
from uncertainties import ufloat

from scipy.stats import circmean, circstd, circvar


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
            return np.degrees(val_unc)

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

def calc_circular_mean_error(data):

    try: # check if data has errors
        errors  = unp.std_devs(data)

        if np.all(errors == 0):
            mean = circmean(data)
            err  = circular_stddev(data) / np.sqrt(len(data))

        else:
            weights = 1 / errors**2
            weights = np.where(np.isfinite(weights), weights, 0.0)  # handle inf/nan

            data    = unp.nominal_values(data)

            mean = circular_mean(data, weights)
            err = circular_sem(data, weights)

    except:
        mean = circmean(data)
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

    return circvar(angles, low=-np.pi, high=np.pi)


def circular_stddev(angles):

    # n = len(angles)
    # R = np.sqrt((np.sum(np.cos(angles)) / n)**2 + (np.sum(np.sin(angles)) / n)**2)
    # return sqrt(1-2lnR)

    return circstd(angles, low=-np.pi, high=np.pi)


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

def kps(series, nu=1):
    # Karl Pearson Second Skewness Coefficient
    # skewness = 3 x (mean - median) / std
    mean = np.mean(series)
    median = np.median(series)
    std = np.std(series, ddof=nu)  # Using sample standard deviation (ddof=1)
    return 3 * (mean - median) / std
