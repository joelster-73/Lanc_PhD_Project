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

def GSE_to_GSM(df, **kwargs):
    """
    Converts coordinates of a specified field from GSE to another coordinate system.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the GSE coordinates for the specified field (e.g., magnetic field, position).

    **kwargs :
        - field : str, required
            The name of the field to be transformed (e.g., 'B_field').

        - coords : str, optional
            The target coordinate system to convert to. Defaults to 'GSM'. Other options can include 'car' for Cartesian coordinates.

        - geom : str, optional
            The type of geometry for the conversion. Defaults to 'car' (Cartesian). Other options may depend on the system.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the converted coordinates ('y' and 'z' components) in the specified coordinate system.
        The 'x' component is omitted as it may not be necessary for some applications.
    """
    # Retrieve the field and optional parameters from kwargs
    field = kwargs.get('field')
    time_col = kwargs.get('time_col',None)

    # Validate the presence of the required field
    if not field:
        raise ValueError('No field selected to transform.')

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
    },
    index=df.index
)

    return result


def calc_B_GSM_angles(df, **kwargs):
    """
    Calculates the clock angle of the magnetic field vector based on the B_y and B_z components in the GSM coordinate system.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing magnetic field components ('B_y_GSM' and 'B_z_GSM').

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the calculated 'B_clock' values, representing the clock angle in radians.
    """
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
    """
    Calculates the clock angle of the magnetic field vector in the GSE coordinate system and adds it as a new column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the magnetic field components ('B_y_GSE' and 'B_z_GSE').

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with an additional column 'B_clock_GSE' representing the clock angle in radians.
    """
    By, Bz = df[['B_y_{coords}', 'B_z_{coords}']].to_numpy().T

    new_field = 'B_clock_{coords}' if coords != 'GSM' else 'B_clock'
    df[new_field] = np.arctan2(By, Bz)
    df.attrs['units'][new_field] = add_unit('theta')

    return df

def insert_field_mag(df, field='r', coords='GSE'):
    """
    Converts the Cartesian coordinates of a field to spherical and cylindrical coordinates, then adds the results as new columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the Cartesian coordinates for the field (e.g., 'r_x_GSE', 'r_y_GSE', 'r_z_GSE').

    field : str, optional
        The name of the field to process (default is 'r').

    coords : str, optional
        The coordinate system used (default is 'GSE').

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with added columns for the magnitude, cylindrical, and spherical coordinates of the field.
    """
    x_data = df[f'{field}_x_{coords}'].to_numpy()
    y_data = df[f'{field}_y_{coords}'].to_numpy()
    z_data = df[f'{field}_z_{coords}'].to_numpy()

    magnitude_from_comps = np.sqrt(x_data**2 + y_data**2 + z_data**2)

    df[f'|{field}|'] = magnitude_from_comps

    df.attrs['units'][f'|{field}|'] = df.attrs['units'][f'{field}_x_{coords}']

