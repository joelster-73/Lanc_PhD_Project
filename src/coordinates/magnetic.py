# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:48:43 2025

@author: richarj2
"""

import numpy as np
import pandas as pd
from spacepy.coordinates import Coords
from spacepy.time import Ticktock

from .spatial import cartesian_to_spherical
from ..processing.utils import add_unit

def GSE_to_GSM(df, field, **kwargs):

    # Retrieve the field and optional parameters from kwargs
    time_col = kwargs.get('time_col',None)

    gse_label = field + '_x_GSE'
    if gse_label not in df.columns:
        raise ValueError(f'GSE coordinates for {field} not available in {df}.')

    # Extract GSE vectors and timings
    vectors = df[[field + '_x_GSE', field + '_y_GSE', field + '_z_GSE']].to_numpy(dtype=np.float64)
    if time_col == 'index' or time_col is None:
        timings = np.array(df.index.to_pydatetime())
    else:
        timings = np.array(df[time_col].dt.to_pydatetime())

    # Convert GSE coordinates to the desired system using the Coords class
    gses = Coords(vectors, 'GSE', 'car')
    gses.ticks = Ticktock(timings, 'UTC')
    new_coords = gses.convert('GSM', 'car').data

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
        gsm = GSE_to_GSM(df, field='B', time_col=time_col)
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

