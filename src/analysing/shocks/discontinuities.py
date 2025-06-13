# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:48:48 2025

@author: richarj2
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from src.processing.speasy.retrieval import retrieve_data
from src.processing.speasy.config import speasy_variables

def find_time_lag(parameter, data1, data2, source1, source2, shock_time, resolution, sampling_interval, time_window_dw):

    step_dir = find_step_direction(data1, shock_time)
    approx_time = find_discontinuity_approx(data2, step_dir) # finding largest jump

    start_refined = approx_time-timedelta(minutes=time_window_dw)
    end_refined   = approx_time+timedelta(minutes=time_window_dw)

    param = parameter
    if 'GSE' in parameter:
        param = '_'.join(parameter.split('_')[:2])
    data2 = retrieve_data(param, source2, speasy_variables,
                              start_refined, end_refined, downsample=True, resolution=sampling_interval, add_omni_sc=False)
    data2 = data2[[parameter]]

    lag, unc, coeff = find_peak_cross_corr(parameter, data1, data2, source1, source2, shock_time, resolution)

    return lag, unc, coeff

def find_peak_cross_corr(parameter, data1, data2, source1, source2, shock_time, resolution):
    # lag > 0 implies source2 measures later than source1
    # lag < 0 implies source2 measures before source1

    aligned = pd.merge(data1[[parameter]].rename(columns={parameter: source1}),
                       data2[[parameter]].rename(columns={parameter: source2}),
                       left_index=True, right_index=True, how='outer')

    series1 = aligned[source1]
    series2 = aligned[source2]

    start_lag = int(np.floor((data2.index.min() - data1.index.max()).total_seconds())/resolution)*resolution
    end_lag = int(np.floor((data2.index.max() - data1.index.min()).total_seconds())/resolution)*resolution
    lags = range(start_lag, end_lag + 1, resolution)

    correlations = []

    for lag in lags:

        # lag>0 means series2 is lag seconds ahead
        # need to use -lag to bring backwards
        series2_shifted = series2.shift(periods=-lag, freq='s')

        valid_indices = (~series1.isna()) & (~series2_shifted.isna())

        # Need at least 2 degrees of freedom
        if np.sum(valid_indices) < int(len(data1)/2):
            corr = np.nan
        else:
            series1_valid = series1[valid_indices]
            series2_valid = series2_shifted[valid_indices]
            corr = series1_valid.corr(series2_valid)

        correlations.append(corr)

    # Cross-correlation values
    corr_series = pd.Series(correlations, index=np.array(lags)/60).dropna()

    try:
        best_lag = corr_series.idxmax()
        best_value = corr_series[best_lag]

        # if best_value<0.6:
        #     print(f'No suitable lag found, best coeff: {best_value:.2f}')
        #     return np.nan, np.nan, best_value

        best_lag_time = int(60*best_lag) # seconds
        time_lag_unc = resolution

        print(f'Lag from {source1} to {source2} for {parameter}: {best_lag_time} s; coeff: {best_value:.2f}')

        return best_lag_time, time_lag_unc, best_value

    except:
        print(f'No shock front between {source1} and {source2}.')
        return np.nan, np.nan, np.nan


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