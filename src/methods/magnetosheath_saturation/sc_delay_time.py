# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 17:19:46 2025

@author: richarj2
"""
import numpy as np
import pandas as pd

from ...config import R_E
from ...coordinates.magnetic import GSE_to_GSM_with_angles
from ...processing.mag.config import lagged_indices

def calc_bs_sc_delay(df, omni_key='sw', sc_key='sc', region='sw'):

    r_bs = df[[f'R_{comp}_BSN_{omni_key}' for comp in ('x','y','z')]].values * R_E
    r_sc = df[[f'r_{comp}_GSE_{sc_key}' for comp in ('x','y','z')]].values * R_E
    try:
        v_sw = df[[f'V_{comp}_GSE_{sc_key}' for comp in ('x','y','z')]].values
    except:
        rotated = GSE_to_GSM_with_angles(df, [[f'V_{comp}_GSM_{sc_key}' for comp in ('x','y','z')]], coords_suffix='c1', inverse=True)
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



def add_dynamic_index_lag(df, omni_key='sw', sc_key='sc', pc_key='pc', region='sw', indices=lagged_indices, plot_lags=True):

    calc_bs_sc_delay(df, omni_key, sc_key, region)

    dt_variable = pd.to_timedelta(df[f'prop_time_s_{sc_key}'], unit='s')

    for ind, lags in indices.items():
        ind_name = f'{ind}_{pc_key}'
        if ind_name not in df:
            print(ind_name,'not in dataframe.')
            continue

        print('lagging',ind_name)

        for lag in lags:
            dt_fixed = pd.to_timedelta(lag, unit='m')
            dt_lag = dt_fixed + dt_variable

            target_index = df.index + dt_lag

            # Drop duplicate timestamps and sort
            df = df[~df.index.duplicated(keep='first')]
            target_index = target_index.drop_duplicates()
            full_index = df.index.union(target_index).sort_values()

            temp = df[ind_name].reindex(full_index).interpolate(method='time')

            column_name = f'{ind}_{lag}m_{pc_key}'

            # Align target_index safely
            aligned_values = temp.reindex(target_index).reindex(df.index).values

            if column_name in df:
                df[column_name] = aligned_values
            else:
                df.insert(df.columns.get_loc(ind) + 1, column_name, aligned_values)
