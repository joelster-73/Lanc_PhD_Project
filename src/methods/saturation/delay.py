# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 17:19:46 2025

@author: richarj2
"""
import numpy as np
import pandas as pd

from ...config import R_E
from ...coordinates.boundaries import bsn_jelinek2012, interpolate_missing
from ...coordinates.spatial import aGSE_to_car


RESPONSE_DELAYS = {'AE': 53, 'SME': 53, 'AL': 53, 'SML': 53, 'AU': 53, 'SMU': 53,
                   'THL': 17, 'RES': 17}

def get_response_delay(index):

    return RESPONSE_DELAYS.get(index,0)


def calc_flat_delay(df, df_omni, region='sw', pos_col='r_x_GSE', vel_col='V_x_GSE', omni_vel_col='V_x_GSE', omni_pres_col='P_flow', lag_col='prop_time_s', min_speeds={'sw': -200, 'msh': -50}):
    """
    Approximates the time it takes solar wind to reach the BSN using just the x components
    This is then added to the lag time from BSN to geomagnetic effect
    """

    # Position
    valid     = df[pos_col].notna().to_numpy()
    positions = df.loc[valid,pos_col].to_numpy()

    # Speed measured by spacecraft
    sc_speeds = df.loc[valid,vel_col].to_numpy()
    sc_speeds = np.minimum(sc_speeds, min_speeds[region]) # negative

    # Bowshock position using OMNI
    omni = df_omni.reindex(df.index) # reindexed so missing times don't return error

    omni_pres = interpolate_missing(omni[omni_pres_col]).to_numpy()
    omni_pres = np.maximum(omni_pres, 0.1) # prevent overflow from p^-1/6.55
    bowshocks = bsn_jelinek2012(omni_pres)

    omni_speed = interpolate_missing(omni[omni_vel_col]).to_numpy()

    # simple aberration
    bowshocks, _, _ = aGSE_to_car(bowshocks, np.zeros_like(bowshocks), np.zeros_like(bowshocks), simple=True, v_x=omni_speed)
    bowshocks = bowshocks[valid]

    distances = positions - bowshocks
    if region=='msh':
        distances = np.where(distances>0, 0, distances) # when sc is in the sheath, bowshock has to be upstream of it

    df.loc[valid,lag_col] = -distances * R_E / sc_speeds

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
    #df1_out = df1_out.reset_index(drop=True)
    df1_out.attrs = df1.attrs

    df2_out = merged[[c if c not in df1_cols else c + '_2' for c in df2_cols]]
    df2_out.columns = df2_cols
    #df2_out = df2_out.reset_index(drop=True)
    df2_out.attrs = df2.attrs

    return df1_out, df2_out

