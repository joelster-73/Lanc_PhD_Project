# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 17:19:46 2025

@author: richarj2
"""
import numpy as np
import pandas as pd

from ...config import R_E, DEFAULT_VALUES
from ...coordinates.boundaries import bsn_jelinek2012
from ...coordinates.spatial import aGSE_to_car

P_SW  = DEFAULT_VALUES.get('sw',{}).get('p')
V_SW   = DEFAULT_VALUES.get('sw',{}).get('v')
V_MSH  = DEFAULT_VALUES.get('msh',{}).get('v')


def calc_flat_delay(df, region='sw', pos_col='r_x_GSE', pres_col='P_flow', vel_col='V_x_GSE', lag_col='prop_time_s', min_speeds={'sw': -200, 'msh': -50}):
    """
    Approximates the time it takes solar wind to reach the BSN using just the x components
    This is then added to the lag time from BSN to geomagnetic effect
    """

    pressure = P_SW # nPa
    speed    = {'sw': V_SW, 'msh': V_MSH} # km/s

    # Position

    valid_positions = ~df[pos_col].isna()
    positions = df.loc[valid_positions,pos_col].to_numpy()

    # Speed
    speeds = df.loc[valid_positions,vel_col].to_numpy()
    speeds[np.isnan(speeds)] = speed.get(region)
    speeds[speeds > min_speeds.get(region)] = np.nan # Flag unreliable speeds

    # Bowshock

    if region=='sw':
        # x2 is because I define pressure with 1/2 prefactor
        pressures = df.loc[valid_positions,pres_col].to_numpy() * 2

        pressures[np.isnan(pressures)] = pressure
        bowshocks = bsn_jelinek2012(pressures)

        solar_wind_speeds = speeds

    else:
        bowshocks = bsn_jelinek2012(pressure)

        solar_wind_speeds = V_SW

    # simple rotation

    bowshocks, _, _ = aGSE_to_car(bowshocks, np.zeros_like(bowshocks), np.zeros_like(bowshocks), simple=True, v_x=solar_wind_speeds)



    df.loc[valid_positions,lag_col] = -(positions - bowshocks) * R_E / speeds

def merge_with_lag(df1, df2, lag, resolution, lag_col='prop_time_s'):
    """
    Aligns df1 and df2 based on the lagged time of df1 and time of df2.
    Importantly, duplicate rows in df1 are kept and matched with df2.
    Replaces the older logic that relied on pre-shifted data:
        intersect = df_ind.index.intersection(df_dep.index)
        df_ind = df_ind.loc[intersect]
        df_dep = df_dep.loc[intersect]
    """
    total_lag = lag + np.rint(df1[lag_col].to_numpy())
    rounded_lag = pd.to_timedelta(total_lag, unit='min').round(resolution)

    if 'lagged_time' in df1:
        df1.drop('lagged_time', axis=1, inplace=True)

    df1.loc[:,'lagged_time'] = df1.index + rounded_lag

    merged = df1.merge(df2, left_on='lagged_time', right_index=True, how='inner', suffixes=('_1', '_2'))

    df1_cols = [c for c in df1.columns if c != 'lagged_time']
    df2_cols = df2.columns

    df1_out = merged[[c if c not in df2_cols else c + '_1' for c in df1_cols]]
    df1_out.columns = df1_cols
    df1_out = df1_out.reset_index(drop=True)
    df1_out.attrs = df1.attrs

    df2_out = merged[[c if c not in df1_cols else c + '_2' for c in df2_cols]]
    df2_out.columns = df2_cols
    df2_out = df2_out.reset_index(drop=True)
    df2_out.attrs = df2.attrs

    return df1_out, df2_out

