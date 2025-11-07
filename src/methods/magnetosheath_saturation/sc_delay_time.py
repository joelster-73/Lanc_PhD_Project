# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 17:19:46 2025

@author: richarj2
"""
import numpy as np
import pandas as pd

from ...config import R_E
from ...processing.omni.config import lagged_indices
from ...coordinates.magnetic import GSE_to_GSM_with_angles

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

    delta_r = r_bs - r_sc
    t = np.einsum('ij,ij->i', delta_r, v_hat) / np.linalg.norm(v_sw, axis=1)
    t[np.abs(t/60)>60] = np.nan # removes delays greater than 60-minutes
    if region=='msh':
        # t *= -1
        t[t>0] = np.nan # delays should always be negative in MSH

    df[f'prop_time_s_{sc_key}'] = t



def add_dynamic_index_lag(df, omni_key='sw', sc_key='sc', indices=lagged_indices, plot_lags=True):

    calc_bs_sc_delay(df, omni_key, sc_key)

    dt_variable = pd.to_timedelta(df[f'prop_time_s_{sc_key}'], unit='s')

    for ind, lags in indices.items():
        if ind not in df:
            continue
        print(ind)

        ind_name = f'{ind}_{omni_key}'
        for lag in lags:
            column_name = f'{ind}_{lag}m_{omni_key}'
            dt_fixed = pd.to_timedelta(lag, unit='m')
            dt_lag = dt_fixed + dt_variable
            print(dt_lag)

            target_index = df.index + dt_lag

            print('Interpolating lag.')
            full_index = df.index.union(target_index)
            temp = df[ind_name].reindex(full_index).interpolate(method='time')

            if column_name in df:
                df[f'{ind}_{lag}m_{omni_key}'] = temp.loc[target_index].values
            else:
                df.insert(df.columns.get_loc(ind) + 1, f'{ind}_{lag}m_{omni_key}', temp.loc[target_index].values)