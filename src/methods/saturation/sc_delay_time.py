# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 17:19:46 2025

@author: richarj2
"""
import numpy as np
import pandas as pd


from ...config import R_E, get_proc_directory
from ...coordinates.magnetic import convert_GSE_to_GSM_with_angles

from ...processing.writing import write_to_cdf
from ...processing.reading import import_processed_data
from ...processing.dataframes import resample_data_weighted
from ...plotting.distributions import plot_freq_hist


### So for this function, add the redundancy checks for missing data and simple flat approximation
### then re-run the merge_sc function


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

    delta_r = -(r_sc - r_bs)
    t = np.einsum('ij,ij->i', delta_r, v_sw) / np.linalg.norm(v_sw, axis=1)**2 # time = distance/speed

    df[f'prop_time_s_{sc_key}'] = t

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

# %% OMNI

def calc_omni_uncertainty(df_omni, column, dt_err_col='rms_timeshift'):
    """
    Estimate the uncertainty on a column in df_omni due to uncertainty in the applied time shift:
        sigma_X = |dX/dt| * sigma_dt
    """
    X = df_omni[column].values
    t_sec = df_omni.index.astype('int64')/1e9 # seconds since epoch

    dX_dt = np.gradient(X, t_sec)
    sigma_dt = df_omni[dt_err_col].values

    return np.abs(dX_dt) * sigma_dt
