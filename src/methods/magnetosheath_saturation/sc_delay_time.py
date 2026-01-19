# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 17:19:46 2025

@author: richarj2
"""
import numpy as np
import pandas as pd

from ...config import R_E
from ...coordinates.magnetic import convert_GSE_to_GSM_with_angles
from ...processing.mag.config import lagged_indices

from ...processing.reading import import_processed_data
from ...processing.dataframes import resample_data_weighted
from ...plotting.distributions import plot_freq_hist

def calc_bs_sc_delay(df, omni_key='sw', sc_key='sc', region='sw'):
    """
    Calculates time it takes solar wind to reach omni BSN, assuming planar solar wind propagation.
    t is the lag to add to the 17,53-minutes
      so t>0 implies the plasma has reached the spacecraft before arriving at the bow shock nose
      so the plasma has to travel this extra time, so is added onto 17
    """

    r_bs = df[[f'R_{comp}_BSN_{omni_key}' for comp in ('x','y','z')]].values * R_E
    r_sc = df[[f'r_{comp}_GSE_{sc_key}' for comp in ('x','y','z')]].values * R_E
    try:
        v_sw = df[[f'V_{comp}_GSE_{sc_key}' for comp in ('x','y','z')]].values
    except:
        rotated = convert_GSE_to_GSM_with_angles(df, [[f'V_{comp}_GSM_{sc_key}' for comp in ('x','y','z')]], coords_suffix='c1', inverse=True)
        v_sw = rotated[[f'V_{comp}_GSE_{sc_key}' for comp in ('x','y','z')]].values

    v_hat = v_sw / np.linalg.norm(v_sw, axis=1)[:, None]

    delta_r = -(r_sc - r_bs)
    t = np.einsum('ij,ij->i', delta_r, v_hat) / np.linalg.norm(v_sw, axis=1) # time = distance/speed

    df[f'prop_time_s_{sc_key}'] = t

def shift_sc_to_bs(df_sc, sample_interval, region='sw', max_delay=60):
    """
    Shifts the time index of the spacecraft data based on propagation to OMNI BSN.
    So comparing df_omni[t] with df_sc[t] is valid.
    Due to possible matching indices, weighted average of duplicate rows is done.
    Delays greater than 60 mins are set to-nan, and msh delays cannot be positive.
    """

    if 'prop_time_s' not in df_sc:
        print('Spacecraft to BSN propagation not in df.')
        return df_sc

    attrs = df_sc.attrs

    times = df_sc['prop_time_s'].copy()
    times[times.abs()>max_delay*60] = np.nan # delays greater than 60-minutes are NaN
    if region=='msh':
        times[times>0] = np.nan # delays should always be negative in MSH

    non_nan_lag = ~times.isna()
    df_sc = df_sc.loc[non_nan_lag]

    df_sc.index -= pd.to_timedelta(df_sc['prop_time_s'], unit='s')
    df_sc        = df_sc.infer_objects(copy=False)

    idx_min  = df_sc.index.floor(sample_interval)
    dup_mask = idx_min.duplicated(keep=False)

    if np.sum(dup_mask)==0:
        print('No duplicate indices.')

    else:
        df_duplicates = df_sc.loc[dup_mask].assign(index_minute=idx_min[dup_mask])

        df_resampled = resample_data_weighted(df_duplicates, time_col='index', sample_interval=sample_interval)

        df_sc = df_sc.loc[~dup_mask]
        df_sc = pd.concat([df_sc, df_resampled], axis=0)
        df_sc = df_sc.sort_index()

    df_sc.attrs = attrs

    return df_sc

def add_dynamic_index_lag(df_sc, df_pc, indices=lagged_indices):
    """
    Shifts the time index of the polar cap response based on BS to PC/AE time.
    And also accounts for the position of the spacecraft relative to the bow shock.
    So comparing df_sc[t] and df_pc[t] is the appropriate driver-response.
    """

    overlap = df_sc.index.intersection(df_pc.index)
    dt_variable = pd.to_timedelta(df_sc.loc[overlap, 'prop_time_s'], unit='s')

    for ind, lags in indices.items():
        if ind not in df_pc:
            print(ind, 'not in dataframe.')
            continue

        print('lagging', ind)

        for lag in lags:
            print(lag)

            dt_lag = pd.to_timedelta(lag, unit='m') + dt_variable

            target_index = overlap + dt_lag.values
            full_index = df_pc.index.union(target_index).sort_values()

            temp = df_pc[ind].reindex(full_index).interpolate(method='time')
            last_valid_time = df_pc[ind].last_valid_index()
            temp.loc[temp.index > last_valid_time] = np.nan
            aligned_values = temp.reindex(target_index).values

            column_name = f'{ind}_{lag}m_adj'
            if column_name not in df_pc:
                df_pc.insert(df_pc.columns.get_loc(f'{ind}_{lag}m') + 1, column_name, np.nan)
                df_pc.attrs.get('units',{}).update({column_name: df_pc.attrs.get('units',{}).get(ind)})

            df_pc.loc[overlap, column_name] = aligned_values


# %%% Delay Histograms

def plot_delay_hists(sc, region, data_pop='plasma', sample_interval='5min'):


    df = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_{sc}')

    plot_freq_hist(df['prop_time_s'], bin_width=60, data_name=f'Lag ({sc.upper()} to BS) [s]', brief_title=region.upper(), sub_directory='prop_hists', file_name=f'{region}_{sc}_{sample_interval}')

