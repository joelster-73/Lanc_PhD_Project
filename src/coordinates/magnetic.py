# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:48:43 2025

@author: richarj2
"""

import warnings
import numpy as np
import pandas as pd

from uncertainties import unumpy as unp
from spacepy.time import Ticktock
from spacepy.ctrans import convert_multitime

from .spatial import cartesian_to_spherical
from ..processing.utils import add_unit

def convert_GSE_to_GSM(df, field, **kwargs):

    # Retrieve the field and optional parameters from kwargs
    time_col = kwargs.get('time_col',None)

    gse_label = field + '_x_GSE'
    if gse_label not in df.columns:
        raise ValueError(f'GSE coordinates for {field} not available in {df}.')

    # Extract GSE vectors and timings
    vectors = df[[f'{field}_{c}_GSE' for c in ('x','y','z')]].to_numpy(dtype=np.float64)
    if time_col == 'index' or time_col is None:
        timings = np.array(df.index.to_pydatetime())
    else:
        timings = np.array(df[time_col].dt.to_pydatetime())

    # Convert GSE coordinates to the desired system using the Coords class
    ticks = Ticktock(timings, 'UTC')
    new_coords = convert_multitime(vectors, ticks, 'GSE', 'GSM', itol=1.0)

    # Create the result DataFrame with the converted coordinates
    result = pd.DataFrame({
        field + '_x_GSM': new_coords[:, 0],
        field + '_y_GSM': new_coords[:, 1],
        field + '_z_GSM': new_coords[:, 2]
    }, index=df.index)

    return result


def calc_B_GSM_angles(df, **kwargs):

    time_col = kwargs.get('time_col', None)

    # Check if GSM components are missing, convert from GSE if necessary
    if 'B_y_GSM' not in df or 'B_z_GSM' not in df:
        gsm = convert_GSE_to_GSM(df, field='B', time_col=time_col)
        Bx, By, Bz = gsm[['B_x_GSM', 'B_y_GSM', 'B_z_GSM']].to_numpy().T
        gsm['B_mag'], gsm['B_pitch'], gsm['B_clock'] = cartesian_to_spherical(Bx, By, Bz)

        return gsm

    x_label = 'B_x_GSM'
    if 'B_x_GSM' not in df:
        x_label = 'B_x_GSE'
    # If GSM components are present, compute the spherical components
    Bx, By, Bz = df[[x_label, 'B_y_GSM', 'B_z_GSM']].to_numpy().T
    B_mag, B_pitch, B_clock = cartesian_to_spherical(Bx, By, Bz)

    # Return the new data as a DataFrame
    new_data = {'B_mag': B_mag, 'B_pitch': B_pitch, 'B_clock': B_clock}
    return pd.DataFrame(new_data, index=df.index)

def calc_B_angle_uncs(df, **kwargs):

    coords = kwargs.get('coords','GSM')

    x_label = f'B_x_{coords}'
    if 'B_x_{coords}' not in df:
        x_label = 'B_x_GSE'
    if x_label not in df:
        raise Exception('No x-component')
    y_label = f'B_y_{coords}'
    z_label = f'B_z_{coords}'

    x_unc = f'{x_label}_unc'
    y_unc = f'B_y_{coords}_unc'
    z_unc = f'B_z_{coords}_unc'
    if x_unc not in df:
        x_unc = 'B_vec_unc'
        y_unc = x_unc
        z_unc = x_unc

    if x_unc not in df:
        raise Exception('No B GSM uncertainties.')

    Bx = unp.uarray(df[x_label].to_numpy(), df[x_unc].to_numpy())
    By = unp.uarray(df[y_label].to_numpy(), df[y_unc].to_numpy())
    Bz = unp.uarray(df[z_label].to_numpy(), df[z_unc].to_numpy())
    B_mag, B_pitch, B_clock = cartesian_to_spherical(Bx, By, Bz)

    B_mag = unp.sqrt(Bx ** 2 + By** 2 + Bz ** 2)
    B_pitch = unp.arccos(Bx / B_mag)
    B_clock = unp.arctan2(By, Bz)

    # Return the new data as a DataFrame
    new_data = {'B_mag_unc': unp.std_devs(B_mag), 'B_pitch_unc': unp.std_devs(B_pitch), 'B_clock_unc': unp.std_devs(B_clock)}
    return pd.DataFrame(new_data, index=df.index)


def insert_clock_angle(df, coords='GSM'):

    By, Bz = df[['B_y_{coords}', 'B_z_{coords}']].to_numpy().T

    new_field = 'B_clock_{coords}' if coords != 'GSM' else 'B_clock'
    df[new_field] = np.arctan2(By, Bz)
    df.attrs['units'][new_field] = add_unit('theta')

    return df

def insert_field_mag(df, field='r', coords='GSE'):

    x_data = df[f'{field}_x_{coords}'].to_numpy()
    y_data = df[f'{field}_y_{coords}'].to_numpy()
    z_data = df[f'{field}_z_{coords}'].to_numpy()

    magnitude_from_comps = np.sqrt(x_data**2 + y_data**2 + z_data**2)

    df[f'{field}_mag'] = magnitude_from_comps

    df.attrs['units'][f'{field}_mag'] = df.attrs['units'][f'{field}_x_{coords}']

# %% Angles

def calc_GSE_to_GSM_angles(df_coords, ref='B', suffix=''):

    # Uses GSE and GSM B data to undo transformation
    uy = df_coords.loc[:, f'{ref}_y_GSM{suffix}']
    uz = df_coords.loc[:, f'{ref}_z_GSM{suffix}']
    vy = df_coords.loc[:, f'{ref}_y_GSE{suffix}']
    vz = df_coords.loc[:, f'{ref}_z_GSE{suffix}']

    dot   = uy * vy + uz * vz
    cross = uy * vz - uz * vy

    # signed rotation angle from v -> u
    return np.arctan2(cross, dot)

def convert_GSE_to_GSM_with_angles(df_transform, vectors, df_coords=None, ref='B', interp=False, coords_suffix='', inverse=False, include_unc=False):
    """
    df_transform : data to rotate from GSE to GSM
    vectors : column(s) to transform in df_transform
    df_coords : contains GSE and GSM vectors
    ref : column with GSE and GSM data in df_coords
    """

    print('Converting...')

    if df_coords is None:
        df_coords = df_transform

    if coords_suffix!='':
        coords_suffix = f'_{coords_suffix}'

    dfs_rotated = pd.DataFrame(index=df_transform.index)

    if f'gse_to_gsm_angle{coords_suffix}' not in df_coords:
        theta = calc_GSE_to_GSM_angles(df_coords, ref=ref, suffix=coords_suffix)

        if interp:
            theta = theta.reindex(df_transform.index, method=None).interpolate(method='time')

        dfs_rotated[f'gse_to_gsm_angle{coords_suffix}'] = theta

    else:
        theta = df_coords.loc[df_transform.index,f'gse_to_gsm_angle{coords_suffix}'].to_numpy()

    start, end = '_GSE','_GSM'
    if inverse:
        theta *= -1
        start, end = '_GSM','_GSE'

    # Builds rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    zeros = np.zeros_like(theta)
    ones = np.ones_like(theta)

    R = np.stack([
        np.stack([ones, zeros, zeros], axis=-1),
        np.stack([zeros, cos_theta, -sin_theta], axis=-1),
        np.stack([zeros, sin_theta, cos_theta], axis=-1)
    ], axis=-2)  # shape (N,3,3)

    dfs_to_concat = [dfs_rotated]
    for vec_cols in vectors: # vectors transforming

        vectors_to_rot = df_transform.loc[:,vec_cols].to_numpy()  # (N,3)
        vectors_rot = np.einsum('nij,ni->nj', R, vectors_to_rot)

        df_rot = pd.DataFrame(vectors_rot, columns=[col.replace(start,end) for col in vec_cols], index=dfs_rotated.index)

        if include_unc:
            # Covariance matrix
            unc_cols = [f'{col}_unc' for col in vec_cols]
            if any(col not in df_transform.columns for col in unc_cols):
                warnings.warn('Uncertainty columns not in dataframe to transform.')
            else:
                vec_unc_GSE = df_transform[unc_cols].to_numpy()  # shape (N,3)

                # Build Nx3x3 diagonal covariance matrices
                C_GSE = np.zeros((len(df_transform),3,3))
                C_GSE[:,0,0] = vec_unc_GSE[:,0]**2
                C_GSE[:,1,1] = vec_unc_GSE[:,1]**2
                C_GSE[:,2,2] = vec_unc_GSE[:,2]**2

                # Rotate covariances to GSM
                C_GSM = np.einsum('nij,njk,nlk->nil', R, C_GSE, R)
  # shape (N,3,3)

                # Extract GSM uncertainties from diagonal
                df_rot[[f'{col.replace("GSE","GSM")}_unc' for col in vec_cols]] = np.sqrt(np.diagonal(C_GSM, axis1=1, axis2=2))

        dfs_to_concat.append(df_rot)


    return pd.concat(dfs_to_concat,axis=1)


# %% GEO

def convert_GEO_to_GSE(df_field, param='H', inplace=True):

    # Extract B in global coordinates (i.e. GEO)
    B_geo = df_field[['B_n_GEO', 'B_e_GEO', 'B_z_GEO']].values

    # Uses UTC times in conversion
    ticks = Ticktock(df_field.index.to_pydatetime(), 'UTC')

    # Convert GEO to GSE
    B_gse = convert_multitime(B_geo, ticks, 'GEO', 'GSE', itol=1.0)

    columns = [f'{param}_x_GSE',f'{param}_y_GSE',f'{param}_z_GSE']

    if inplace:
        df_field.loc[:,columns] = pd.DataFrame(B_gse, index=df_field.index, columns=columns)
        for col in columns:
            df_field.attrs['units'][col] = 'nT'
        return None

    return pd.DataFrame(B_gse, index=df_field.index, columns=columns)

def convert_GSE_to_aGSE(df_field, df_sw, param='H', V_earth=29.78, alpha=None, inplace=True):

    overlap = df_field.index.intersection(df_sw.index)

    field_vals = df_field.loc[overlap, [f'{param}_x_GSE', f'{param}_y_GSE', f'{param}_z_GSE']].values
    V_vals     = df_sw.loc[overlap, ['V_x_GSE', 'V_y_GSE']].values

    if alpha is None:
        # Aberration including Earth orbital speed
        alpha      = -np.arctan((V_earth + V_vals[:,1])/np.abs(V_vals[:,0]))
    cosa, sina = np.cos(alpha), np.sin(alpha)

    # Rotate in-plane components
    B_x_agse =  field_vals[:,0]*cosa + field_vals[:,1]*sina
    B_y_agse = -field_vals[:,0]*sina + field_vals[:,1]*cosa
    B_z_agse =  field_vals[:,2]

    B_agse   = np.column_stack([B_x_agse, B_y_agse, B_z_agse])

    columns = [f'{param}_x_aGSE',f'{param}_y_aGSE',f'{param}_z_aGSE']

    if inplace:
        df_field.loc[overlap,columns] = pd.DataFrame(B_agse, index=overlap, columns=columns)
        for col in columns:
            df_field.attrs['units'][col] = 'nT'
        return None

    return pd.DataFrame(B_agse, index=overlap, columns=columns)