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
from ...plotting.distributions import plot_freq_hist


def calc_bs_sc_delay(df, omni_key='sw', sc_key='sc', region='sw'):

    r_bs = df[[f'R_{comp}_BSN_{omni_key}' for comp in ('x','y','z')]].values * R_E
    r_sc = df[[f'r_{comp}_GSE_{sc_key}' for comp in ('x','y','z')]].values * R_E
    try:
        v_sw = df[[f'V_{comp}_GSE_{sc_key}' for comp in ('x','y','z')]].values
    except:
        rotated = convert_GSE_to_GSM_with_angles(df, [[f'V_{comp}_GSM_{sc_key}' for comp in ('x','y','z')]], coords_suffix='c1', inverse=True)
        v_sw = rotated[[f'V_{comp}_GSE_{sc_key}' for comp in ('x','y','z')]].values

    v_hat = v_sw / np.linalg.norm(v_sw, axis=1)[:, None]

    # t is the lag to add to the 17-minutes
    # so t>0 implies the plasma has reached the spacecraft before arriving at the bow shock nose
    # so the plasma has to travel this extra time, so is added onto 17

    delta_r = -(r_sc - r_bs)
    t = np.einsum('ij,ij->i', delta_r, v_hat) / np.linalg.norm(v_sw, axis=1)
    t[np.abs(t/60)>20] = 1200 # removes delays greater than 60-minutes
    if region=='msh':
        # t *= -1
        t[t>0] = 0 # delays should always be negative in MSH

    t[np.isnan(t)] = 0
    df[f'prop_time_s_{sc_key}'] = t



def add_dynamic_index_lag(df_sc, df_pc, indices=lagged_indices):

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

