# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:48:48 2025

@author: richarj2
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from .config import optimal_parameters
from ...processing.speasy.retrieval import retrieve_data

def get_average_interval(time, where='up', shock_type='FF'):
    if where=='up' and shock_type=='FF':
        return time-timedelta(minutes=9), time-timedelta(minutes=1)
    elif where=='up' and shock_type=='FR':
        return time+timedelta(minutes=1), time+timedelta(minutes=9)
    elif where=='dw' and shock_type=='FF':
        return time+timedelta(minutes=2), time+timedelta(minutes=10)
    elif where=='dw' and shock_type=='FR':
        return time-timedelta(minutes=10), time-timedelta(minutes=2)


def sufficient_compression(parameter, detector, interceptor, shock_time, intercept_time, **kwargs):

    min_ratio_change = kwargs.get('min_ratio',optimal_parameters['min_ratio'])
    min_points       = kwargs.get('min_points',3)

    shock_type = 'FF'
    for i in range(2):
        data1_up_start, data1_up_end = get_average_interval(shock_time, 'up', shock_type)
        data1_dw_start, data1_dw_end = get_average_interval(shock_time, 'dw', shock_type)

        data1_up = retrieve_data(parameter, detector, data1_up_start, data1_up_end, add_omni_sc=False).dropna()

        data1_dw = retrieve_data(parameter, detector, data1_dw_start, data1_dw_end, add_omni_sc=False).dropna()

        if len(data1_up)<min_points or len(data1_dw)<min_points:
            return False

        # Use calc_mean_error and incorporate errors/uncertainties
        data1_up_avg = np.mean(data1_up.to_numpy())
        data1_dw_avg = np.mean(data1_dw.to_numpy())

        B1_ratio = data1_dw_avg/data1_up_avg

        if B1_ratio<1: # have an FR shock
            shock_type='FR'
        else:
            break

    data2_up_start, data2_up_end = get_average_interval(intercept_time, 'up', shock_type)
    data2_dw_start, data2_dw_end = get_average_interval(intercept_time, 'dw', shock_type)

    data2_up = retrieve_data(parameter, interceptor, data2_up_start, data2_up_end, add_omni_sc=False).dropna()

    data2_dw = retrieve_data(parameter, interceptor, data2_dw_start, data2_dw_end, add_omni_sc=False).dropna()

    if len(data2_up)<min_points or len(data2_dw)<min_points:
        return False

    data2_up_avg = np.mean(data2_up.to_numpy())
    data2_dw_avg = np.mean(data2_dw.to_numpy())

    B2_ratio = data2_dw_avg/data2_up_avg
    # Consider being less strict on OMNI and how to incorporate
    min_B_comp = max(1.2,min_ratio_change*B1_ratio)

    return (B2_ratio >= min_B_comp)


def find_peak_cross_corr(parameter, series1, series2, source1, source2, shock_time, lags, check_compression=True, overlap_mins=3, **kwargs):
    # lag > 0 implies source2 measures later than source1
    # lag < 0 implies source2 measures before source1

    correlations = []

    for lag in lags:

        # lag>0 means series2 is lag seconds ahead
        # need to use -lag to bring backwards
        series2_shifted = series2.shift(periods=-lag, freq='s')
        valid_indices = (~series1.isna()) & (~series2_shifted.isna())

        # Need at least 2 degrees of freedom

        # Try passing in overlap_mins into .corr()
        if np.sum(valid_indices) < overlap_mins:
            corr = np.nan
        else:
            series1_valid = series1[valid_indices]
            series2_valid = series2_shifted[valid_indices]
            corr = series1_valid.corr(series2_valid)

        correlations.append(corr)

    # Cross-correlation values
    corr_series = pd.Series(correlations, index=np.array(lags)).dropna()

    try:
        if check_compression:
            for lag, corr in corr_series.sort_values(ascending=False).items():
                lagged_time = shock_time + timedelta(seconds=lag)
                if sufficient_compression(parameter, source1, source2, shock_time, lagged_time, **kwargs):
                    return lag, corr

            return np.nan, np.nan

        else:
            best_lag = corr_series.idxmax()
            best_value = corr_series[best_lag]

            return best_lag, best_value

    except:
        return np.nan, np.nan


def find_step_direction(df, shock_time):
    differences = df.diff()

    floored_time = pd.Timestamp(shock_time).floor('1min')

    if floored_time in df.index:
        diff_before = differences.loc[floored_time].iloc[0]
        dir_before = 'inc' if diff_before>0 else 'dec'

        try:
            diff_after = differences.loc[floored_time + pd.Timedelta(minutes=1)].iloc[0]
        except:
            return dir_before
        dir_after  = 'inc' if diff_after>0  else 'dec'

        if dir_before == dir_after:
            return dir_before
        elif abs(diff_before) > abs(diff_after):
            return dir_before
        else:
            return dir_after

    else:
        return None

def find_discontinuity_approx(df, shock_direction='inc'):
    differences = df.diff() / df
    if shock_direction=='inc':
        time_guess = differences.stack().idxmax()
    elif shock_direction=='dec':
        time_guess = differences.stack().idxmin()

    return time_guess[0]

def find_closest(shocks):

    if 'closest' not in shocks.columns:
        shocks['closest'] = None

    sc_labels = [col.split('_')[0] for col in shocks if '_coeff' in col]

    for index, shock in shocks.iterrows():

        omni_pos = np.array([shock[f'OMNI_r_{comp}_GSE'] for comp in ('x','y','z')])
        for comp in omni_pos:
            if np.isnan(comp):
                continue

        sc_distances = {}
        for sc in sc_labels:
            if sc == 'OMNI' or sc==shock['spacecraft']:
                continue

            sc_pos = np.array([shock[f'{sc}_r_{comp}_GSE'] for comp in ('x','y','z')])
            for comp in sc_pos:
                if np.isnan(comp):
                    continue
            sc_distances[sc] = np.linalg.norm(sc_pos - omni_pos)

        shocks.at[index,'closest'] = min(sc_distances, key=sc_distances.get)