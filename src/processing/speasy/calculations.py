# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 16:06:50 2025

@author: richarj2
"""

import numpy as np
import pandas as pd
from scipy.constants import mu_0

def gse_to_gsm_with_angle(df, ref='B', vec='V'):
    """
    Given we have ref_GSE and ref_GSM
    We can calculate the rotation matrix
    And apply this to vec_GSE to determine vec_GSM
    """

    theta = np.arctan2(df.loc[:,f'{ref}_z_GSM'], df.loc[:,f'{ref}_y_GSM']) - np.arctan2(df.loc[:,f'{ref}_z_GSE'], df.loc[:,f'{ref}_y_GSE'])
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    zeros = np.zeros_like(theta)
    ones = np.ones_like(theta)

    # Shape (N,3,3) for per-row rotation matrices
    R = np.stack([
        np.stack([ones, zeros, zeros], axis=-1),
        np.stack([zeros, cos_theta, -sin_theta], axis=-1),
        np.stack([zeros, sin_theta, cos_theta], axis=-1)
    ], axis=-2)  # shape (N,3,3)

    vectors = df.loc[:,[f'{vec}_x_GSE', f'{vec}_y_GSE', f'{vec}_z_GSE']].to_numpy()  # (N,3)

    vectors_rot = np.einsum('nij,ni->nj', R, vectors)

    df_rot = pd.DataFrame(
        vectors_rot,
        columns=[f'{vec}_x_GSM', f'{vec}_y_GSM', f'{vec}_z_GSM'],
        index=df.index
    )

    return df_rot



def cross_product(df1, df2=None, cross_name='CROSS', var1_name=None, var2_name=None):

    # Assume units are SI

    if df2 is None:
        if var1_name is None or var2_name is None:
            raise Exception('Need to pass two dataframes or the names of both products with one dataframe.')

        vec1_name, vec1_coords = var1_name.split('_')
        vec1_cols = [f'{vec1_name}_{comp}_{vec1_coords}' for comp in ('x','y','z')]
        if vec1_coords=='GSM' and f'{vec1_name}_x_GSM' not in df1:
            vec1_cols[0] = f'{vec1_name}_x_GSE'
        vec1 = df1.loc[:, vec1_cols].values

        vec2_name, vec2_coords = var2_name.split('_')
        vec2_cols = [f'{vec2_name}_{comp}_{vec2_coords}' for comp in ('x','y','z')]
        if vec2_coords=='GSM' and f'{vec2_name}_x_GSM' not in df1:
            vec2_cols[0] = f'{vec2_name}_x_GSE'
        vec2 = df1.loc[:, vec2_cols].values

        if vec1_coords != vec2_coords:
            raise Warning(f'{var1_name} and {var2_name} not in same coordinate system.')

        comp_columns = [f'{cross_name[0]}_{comp}_{vec1_coords}' for comp in ('x','y','z')]

    else:
        df1_cols = list(df1.columns)
        df2_cols = list(df2.columns)

        merged = df1.merge(df2, left_index=True, right_index=True, how='inner')

        vec1 = merged[df1_cols].values
        vec2 = merged[df2_cols].values

        comp_columns = [f'{cross_name}_x', f'{cross_name}_y', f'{cross_name}_z']

    if 'E_' in cross_name:
        # df1 is B and df2 is V
        vec1 *= 1e-9  # nT -> T
        vec2 *= 1e3   # km/s -> m/s

    elif 'S_' in cross_name:
        # df1 is E and df2 is B
        vec1 *= 1e-3  # mV/m -> V/m
        vec2 *= 1e-9  # nT -> T

    cross = np.cross(vec1, vec2)
    if df2 is None:
        df_cross = pd.DataFrame(cross, index=df1.index, columns=comp_columns)
    else:
        df_cross = pd.DataFrame(cross, index=merged.index, columns=comp_columns)

    unit = ''
    if 'E_' in cross_name:
        df_cross *= 1e3 # V/m -> mV/m
        unit = 'mV/m'
    elif 'S_' in cross_name:
        df_cross /= mu_0
        df_cross *= 1e6 # W/m2 -> uW/m2
        unit = 'uW/m2'

    df_cross[f'{cross_name[0]}_mag'] = df_cross[comp_columns].apply(lambda row: np.linalg.norm(row), axis=1)

    df_cross.attrs['units'] = {}
    for col in df_cross.columns:
        df_cross.attrs['units'][col] = unit

    return df_cross
