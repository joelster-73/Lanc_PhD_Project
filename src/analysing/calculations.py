# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:11:13 2025

@author: richarj2
"""
import numpy as np
import pandas as pd

from uncertainties import unumpy as unp
from uncertainties import ufloat

def calc_mean_error(series,start=None,end=None):

    if start:
        series = series[series.index >= pd.Timestamp(start)]
    if end:
        series = series[series.index <= pd.Timestamp(end)]

    # Calculate mean and SEM
    try:
        mean = series.mean()
    except:
        mean = np.mean(series)

    err = sem(series)

    return ufloat(mean, err)

def calc_circular_mean_error(series,start=None,end=None,unit='rad'):

    if start:
        series = series[series.index >= pd.Timestamp(start)]
    if end:
        series = series[series.index <= pd.Timestamp(end)]

    # Calculate mean and SEM
    mean = circular_mean(series)
    err = circular_standard_deviation(series,1) / np.sqrt(np.size(series))

    if unit == 'deg':
        mean = np.degrees(mean)
        err = np.degrees(err)

    return ufloat(mean, err)


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


def circular_mean(angles):
    return np.arctan2(np.sum(np.sin(angles)), np.sum(np.cos(angles)))

def circular_variance(angles, dof=1):
    """
    Calculate the circular variance of an array of angles.

    Parameters:
    angles (array-like): Angles in radians.

    Returns:
    float: Circular variance (0 to 1).
    """
    n = len(angles) - dof
    R = np.sqrt((np.sum(np.cos(angles)) / n)**2 + (np.sum(np.sin(angles)) / n)**2)
    return 1 - R

def circular_standard_deviation(angles, dof=1):
    """
    Calculate the circular standard deviation of an array of angles.

    Parameters:
    angles (array-like): Angles in radians.

    Returns:
    float: Circular standard deviation.
    """
    n = len(angles) - dof
    R = np.sqrt((np.sum(np.cos(angles)) / n)**2 + (np.sum(np.sin(angles)) / n)**2)
    return np.sqrt(-2 * np.log(R))

def vec_mag(vec):

    try:
        return np.linalg.norm(vec)
    except:
        return unp.sqrt(np.sum(vec**2))
    return np.nan


def get_position_u(shock, sc):

    x = shock[f'{sc}_r_x_GSE']
    y = shock[f'{sc}_r_y_GSE']
    z = shock[f'{sc}_r_z_GSE']
    for comp in (x,y,z):
        if np.isnan(comp):
            return None

    x_u = shock[f'{sc}_r_x_GSE_unc']
    y_u = shock[f'{sc}_r_x_GSE_unc']
    z_u = shock[f'{sc}_r_x_GSE_unc']
    for unc in (x_u,y_u,z_u):
        if np.isnan(unc):
            unc = 0
    return unp.uarray([x,y,z],[x_u,y_u,z_u])