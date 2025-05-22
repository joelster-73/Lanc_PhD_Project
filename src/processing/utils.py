# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:32:02 2025

@author: richarj2
"""
import os

import numpy as np
import pandas as pd

from spacepy import pycdf


def add_unit(key):
    if any(keyword in key for keyword in ['theta','phi','alpha', 'pitch', 'clock', 'cone', 'angle']): # angles
        return 'rad'
    elif 'r_' in key:
        return 'Re'
    elif 'v_' in key:
        return 'km/s'
    elif 'B_' in key:
        return 'nT'
    elif 'epoch' in key:
        return 'cdf_epoch'
    elif 'p_' in key:
        return 'nPa'
    elif '_s' in key:
        return 's'
    elif any(keyword in key for keyword in ['ni','np','n_i','n_p']): # densities
        return 'n/cc'
    elif 'Nx' in key or 'Ny' in key or 'Nz' in key:
        return '1'
    elif 'time' in key:
        return 'datetime'
    else:
        return ''

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def datetime_to_cdf_epoch(datetimes):
    # Equivalent to Ticktock(datetimes, 'UTC').CDF
    if isinstance(datetimes, (pd.Series, pd.DatetimeIndex)):
        # If it's a pandas type, convert it directly to Python datetime objects
        datetimes = datetimes.to_pydatetime()
    elif isinstance(datetimes, np.ndarray):
        # For NumPy ndarray, convert datetime64 to datetime objects
        if np.issubdtype(datetimes.dtype, np.datetime64):
            # Convert to Python datetime objects
            datetimes = pd.to_datetime(datetimes).to_pydatetime()  # Direct conversion
        elif np.issubdtype(datetimes.dtype, np.object_):
            pass  # If it's already Python datetime objects, do nothing
        else:
            raise ValueError('Array must be of datetime64 type or an array of datetime objects.')
    else:
        raise ValueError('Input must be a pandas Series, DataFrame index, or a NumPy ndarray of datetime objects.')

    # Pass the converted datetime objects to SpacePy's CDF epoch converter
    return pycdf.lib.v_datetime_to_epoch(datetimes)

def cdf_epoch_to_datetime(cdf_epochs):
    # Equivalent to Ticktock(cdf_epochs, 'CDF').UTC

    if isinstance(cdf_epochs[0], (int, float)):
        return pycdf.lib.v_epoch_to_datetime(cdf_epochs)
    return cdf_epochs