# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:48:48 2025

@author: richarj2
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from ...processing.speasy.retrieval import retrieve_data
from ...processing.speasy.config import speasy_variables

def find_step_direction(df, disc_time, shock_type=None):
    differences = df.diff()

    floored_time = pd.Timestamp(disc_time).floor('1min')

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
    elif shock_type is not None:
        if shock_type in ('FF','SR'):
            return 'inc'
        elif shock_type in ('FR','SR'):
            return 'dec'
    else:
        return None

def find_time_lag(parameter, source1, source2, source1_time, source2_time):
    # lag > 0 implies source2 measures later than source1
    # lag < 0 implies source2 measures before source1


    if set(['OMNI','DSC']).intersection([source1,source2]):
        resolution = 1
    elif set(['ACE','IMP8']).intersection([source1,source2]):
        resolution = 0.5
    else:
        resolution = 0.25

    # either smarter way
    # or check for up/downsample clever
    sampling = {0.25: '15s', 0.5: '30s', 1: '1min'}
    sampling_interval = sampling.get(resolution)
    res_s = int(60*resolution)

    # needs improving to take into account proximity to shock spacecraft
    buffer_size_up = 20
    window_start_up = source1_time - timedelta(minutes=buffer_size_up)
    window_end_up   = source1_time + timedelta(minutes=buffer_size_up)
    
    data1 = retrieve_data(parameter, source1, speasy_variables,
                              window_start_up, window_end_up, downsample=True, resolution=sampling_interval)
    
    buffer_size_dw = 80
    if source2 in ('WIND','ACE','DSC'):
        buffer_size_dw = 50

    window_start = source2_time - timedelta(minutes=buffer_size_dw)
    window_end   = source2_time + timedelta(minutes=buffer_size_dw)

    data2 = retrieve_data(parameter, source2, speasy_variables,
                              window_start, window_end, downsample=True, resolution=sampling_interval)

    if data1.empty or data2.empty:
        return np.nan, np.nan, np.nan

    aligned = pd.merge(data1[[parameter]].rename(columns={parameter: source1}),
                       data2[[parameter]].rename(columns={parameter: source2}),
                       left_index=True, right_index=True, how='outer')

    series1 = aligned[source1]
    series2 = aligned[source2]
    
    start_lag = int(np.floor((aligned.index.min() - source1_time).total_seconds())/res_s)*res_s
    end_lag = int(np.floor((aligned.index.max() - source1_time).total_seconds())/res_s)*res_s
    lags = range(start_lag, end_lag + 1, res_s)

    correlations = []
    
    for lag in lags:
        
        # lag>0 means series2 is lag seconds ahead
        # need to use -lag to bring backwards
        series2_shifted = series2.shift(periods=-lag, freq='s')

        valid_indices = (~series1.isna()) & (~series2_shifted.isna())
        
        # Need at least 2 degrees of freedom
        #print(series2_shifted)
        if np.sum(valid_indices) < buffer_size_up/resolution:
            corr = np.nan
        else:
            series1_valid = series1[valid_indices]
            series2_valid = series2_shifted[valid_indices]
    
            # Need at least 2 degrees of freedom
            corr = series1_valid.corr(series2_valid)
            
        
        correlations.append(corr)

    # Cross-correlation values
    corr_series = pd.Series(correlations, index=np.array(lags)/60).dropna()

    #############################
    #pdb.set_trace() ### remove
    #############################

    try:
        best_lag = corr_series.idxmax()
        best_value = corr_series[best_lag]

        # if best_value<0.6:
        #     print(f'No suitable lag found, best coeff: {best_value:.2f}')
        #     return np.nan, np.nan, best_value

        best_lag_time = int(60*best_lag) # seconds
        time_lag_unc = res_s

        # print(f'Lag from {source1} to {source2}: {best_lag_time} s; coeff: {best_value:.2f}')
        # print(source1_time,source2_time)
        # if source2=='C1':
        #     corr_series.plot()

        return best_lag_time, time_lag_unc, best_value

    except:
        #print(f'No shock front between {source1} and {source2}.')
        return np.nan, np.nan, np.nan

def find_time_lag_curr(parameter, source1, source2, source1_time, source2_time):
    # lag > 0 implies source2 measures later than source1
    # lag < 0 implies source2 measures before source1


    if set(['OMNI','DSC']).intersection([source1,source2]):
        resolution = 1
    elif set(['ACE','IMP8']).intersection([source1,source2]):
        resolution = 0.5
    else:
        resolution = 0.25

    # either smarter way
    # or check for up/downsample clever
    sampling = {0.25: '15s', 0.5: '30s', 1: '1min'}
    sampling_interval = sampling.get(resolution)

    # needs improving to take into account proximity to shock spacecraft
    if source2 in ('WIND','ACE','DSC'):
        source2_time = source1_time

    buffer_size = 20

    window_start = source1_time - timedelta(minutes=buffer_size)
    window_end   = source1_time + timedelta(minutes=buffer_size)

    data1 = retrieve_data(parameter, source1, speasy_variables,
                              window_start, window_end, downsample=True, resolution=sampling_interval)


    window_start = source2_time - timedelta(minutes=3*buffer_size)
    window_end   = source2_time + timedelta(minutes=4*buffer_size)

    data2 = retrieve_data(parameter, source2, speasy_variables,
                              window_start, window_end, downsample=True, resolution=sampling_interval)

    if data1.empty or data2.empty:
        return np.nan, np.nan, np.nan

    aligned = pd.merge(data1[[parameter]].rename(columns={parameter: source1}),
                       data2[[parameter]].rename(columns={parameter: source2}),
                       left_index=True, right_index=True, how='outer')

    series1 = aligned[source1]
    series2 = aligned[source2]

    start_lag = int((window_start-source1_time).total_seconds()/(60*resolution))
    end_lag   = int((window_end-source1_time).total_seconds()/(60*resolution))

    lags = range(start_lag, end_lag+1)
    correlations = []
    for lag in lags:
        corr = np.nan

        try:
            if lag < 0:
                series1_slice = series1[-lag:].values
                series2_slice = series2[:lag].values
            elif lag > 0:
                series1_slice = series1[:-lag].values
                series2_slice = series2[lag:].values
            else:
                series1_slice = series1.values
                series2_slice = series2.values
        except: # in case index out of range
            continue

        mask = ~np.isnan(series1_slice) & ~np.isnan(series2_slice)
        if np.sum(mask) < buffer_size/resolution:
            corr = np.nan
        elif mask.any():
            corr = np.corrcoef(series1_slice[mask], series2_slice[mask])[0, 1]
            if ~np.isfinite(corr):
                corr = np.nan

        correlations.append(corr)

    # Cross-correlation values
    corr_series = pd.Series(correlations, index=lags).dropna()

    #############################
    #pdb.set_trace() ### remove
    #############################

    try:
        best_lag = corr_series.idxmax()
        best_value = corr_series[best_lag]

        # if best_value<0.6:
        #     print(f'No suitable lag found, best coeff: {best_value:.2f}')
        #     return np.nan, np.nan, best_value

        best_lag_time = int(60*best_lag*resolution) # seconds
        time_lag_unc = 60*resolution

        #print(f'Lag: {best_lag_time} s; coeff: {best_value:.2f}')
        #corr_series.plot()

        return best_lag_time, time_lag_unc, best_value

    except:
        #print(f'No shock front between {source1} and {source2}.')
        return np.nan, np.nan, np.nan


def find_discontinuity(df_low, df_high=None, shock_direction='inc'):
    # Uses lower, 1-minute data to find approximate region
    # And then higher resolution for higher precision

    time_low, low_uncertainty, low_left_unc, low_right_unc = find_discontinuity_approx(df_low, shock_direction)
    if df_high is not None and len(df_high.dropna())>0:
        try:
            differences = df_high.diff() / df_high
            if shock_direction=='inc':
                time_guess = differences.stack().idxmax()
                order = False
            elif shock_direction=='dec':
                time_guess = differences.stack().idxmin()
                order = True
            row_time = time_guess[0]

            # Higher resolution time outside range for lower resolution
            # Likely detected noise
            lower_bound = time_low - timedelta(seconds=low_left_unc)
            upper_bound = time_low + timedelta(seconds=low_right_unc)

            if row_time < lower_bound or row_time > upper_bound:
                sorted_differences = differences.stack().sort_values(ascending=order)

                for timestamp, diff in sorted_differences.items():
                    new_time = timestamp[0]
                    if lower_bound <= new_time <= upper_bound:
                        row_time = new_time
                        break

            time_unc_left = (row_time - lower_bound).total_seconds()
            time_unc_right = (upper_bound - row_time).total_seconds()
            time_unc = np.sqrt(time_unc_left**2+time_unc_right**2)

            return row_time, time_unc, time_unc_left, time_unc_right
        except:
            return time_low, low_uncertainty, low_left_unc, low_right_unc

    return time_low, low_uncertainty, low_left_unc, low_right_unc

def find_discontinuity_approx(df, shock_direction='inc'):
    differences = df.diff() / df
    if shock_direction=='inc':
        time_guess = differences.stack().idxmax()
    elif shock_direction=='dec':
        time_guess = differences.stack().idxmin()

    row_time = time_guess[0]
    column = time_guess[1]
    column_differences = differences[column]

    prev_time = None
    try:
        ind = df.index.get_loc(row_time)
        prev_time = df.index[ind-1]
        is_peak = column_differences.iloc[ind] * column_differences.iloc[ind+1] < 0
        if is_peak:
            time = prev_time + (row_time - prev_time) / 2 # midpoint of highest jump
        else:
            time = row_time
    except:
        time = row_time
        ind = df.index.get_loc(row_time)

    for i in range(df.index.get_loc(row_time)-1, -1, -1): # Finds where conditions start to increase
        if column_differences.iloc[i] * column_differences.iloc[i + 1] < 0:
            prev_time = df.index[i]
            break
    else:
        prev_time = df.index[0]

    next_time = None
    for i in range(df.index.get_loc(row_time) + 1, len(df.index)): # Finds where conditions start to decrease
        if column_differences.iloc[i] * column_differences.iloc[i - 1] < 0:
            next_time = df.index[i-1] # -1 so when decreasing
            break
    else:
        next_time = df.index[len(df.index)-1]

    time_unc_left = (time - prev_time).total_seconds()
    time_unc_right = (next_time - time).total_seconds()
    time_unc = np.sqrt(time_unc_left**2+time_unc_right**2)

    return time, time_unc, time_unc_left, time_unc_right