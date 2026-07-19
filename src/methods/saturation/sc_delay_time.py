# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 17:19:46 2025

@author: richarj2
"""
import numpy as np
import pandas as pd


from ...config import R_E, get_proc_directory
from ...coordinates.magnetic import convert_GSE_to_GSM_with_angles
from ...coordinates.boundaries import bsn_jelinek2012

from ...processing.writing import write_to_cdf
from ...processing.reading import import_processed_data
from ...processing.dataframes import resample_data_weighted
from ...plotting.distributions import plot_freq_hist


def calc_flat_delay(df, region='sw', pos_col='r_x_GSE', pres_col='P_flow', vel_col='V_x_GSE', lag_col='prop_time_s'):
    """
    Approximates the time it takes solar wind to reach the BSN using just the x components
    This is then added to the lag time from BSN to geomagnetic effect
    """

    pressure = 2.056 # nPa
    speed    = {'sw': -440, 'msh': -110} # km/s

    # Position

    valid_positions = ~df[pos_col].isna()
    positions = df.loc[valid_positions,pos_col].to_numpy()

    # Bowshock

    if region=='sw':
        pressures = df.loc[valid_positions,pres_col].to_numpy() * 2 # x2 is because I define pressure with 1/2 prefactor

        pressures[np.isnan(pressures)] = pressure
        bowshocks = bsn_jelinek2012(pressures)

    else:
        bowshocks = bsn_jelinek2012(pressure)

    # Speed

    speeds = df.loc[valid_positions,vel_col].to_numpy()
    speeds[np.isnan(speeds)] = speed.get(region)

    df.loc[valid_positions,lag_col] = -(positions - bowshocks) * R_E / speeds

def shift_sc_to_bs(df_sc, sample_interval, region='sw', max_delay=60, write_to_file=False):
    """
    Shifts the time index of the spacecraft data based on propagation to OMNI BSN.
    So comparing df_omni[t] with df_sc[t] is valid.
    Due to possible matching indices, weighted average of duplicate rows is done.
    Delays greater than 60 mins are set to-nan, and msh delays cannot be positive.
    TL;DR this just shifts data to the BSN to create essentially a new OMNI but using Cluster, THEMIS, MMS.
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

    df_sc.index = df_sc.index.floor(sample_interval)
    dup_mask    = df_sc.index.duplicated(keep=False)

    if np.sum(dup_mask)==0:
        print('No duplicate times.')

    else:
        df_duplicates = df_sc.loc[dup_mask]

        df_resampled = resample_data_weighted(df_duplicates, time_col='index', sample_interval=sample_interval)

        df_sc = df_sc.loc[~dup_mask]
        df_sc = pd.concat([df_sc, df_resampled], axis=0)
        df_sc = df_sc.sort_index()

    df_sc.rename_axis('epoch', inplace=True)
    df_sc.index.name = 'epoch'

    df_sc.attrs = attrs
    df_sc.attrs['units'][f'sc_{region}'] = 'STRING'
    df_sc.attrs['units']['prop_time_s_unc'] = 's'
    df_sc.attrs['units']['prop_time_s_count'] = 'NUM'
    df_sc.attrs['units']['Delta B_z_unc'] = 'nT'
    df_sc.attrs['units']['Delta B_z_count'] = 'NUM'

    if write_to_file:
        print('Writing shifted to file...')
        DIR = get_proc_directory(region, dtype='plasma', resolution=sample_interval) # output directory
        write_to_cdf(df_sc, directory=DIR, file_name=f'{region}_times_shifted', reset_index=True)

    return df_sc


# %%% Delay Histograms

def plot_delay_hists(sc, region, data_pop='plasma', sample_interval='5min'):

    df = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_{sc}')

    plot_freq_hist(df['prop_time_s'], bin_width=60, data_name=f'Lag ({sc.upper()} to BS) [s]', brief_title=region.upper(), sub_directory='prop_hists', file_name=f'{region}_{sc}_{sample_interval}')

