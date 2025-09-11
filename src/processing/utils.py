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
    elif any(keyword in key for keyword in ['v_','V_']):
        return 'km/s'
    elif any(keyword in key for keyword in ['Nx', 'Ny', 'Nz', 'M_', 'ratio', 'Beta_', 'coeff', 'beta', 'MA']):
        return '1'
    elif any(keyword in key for keyword in ['mode','quality']):
        return 'NUM'
    elif any(keyword in key for keyword in ['N_tot','ni','np','n_i','n_p']): # densities
        return 'n/cc'
    elif any(keyword in key for keyword in ['unc_s','_s','_delay']):
        return 's'
    elif '_time' in key:
        return 'datetime'
    elif 'r_' in key:
        return 'Re'
    elif any(keyword in key for keyword in ['B_','AE','AU','AL']):
        return 'nT'
    elif 'E_' in key:
        return 'mV/m'
    elif 'T_' in key:
        return 'K'
    elif 'epoch' in key:
        return 'cdf_epoch'
    elif 'P_' in key:
        return 'nPa'
    elif 'S_' in key:
        return 'uW/m2' # Poynting Flux
    else:
        print(f'No unit for "{key}"')
        return ''

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



CDF_EPOCH_FILLVAL = -9223372036854775808

def datetime_to_cdf_epoch(datetimes):
    # Convert input to an array of Python datetime objects
    if isinstance(datetimes, (pd.Series, pd.DatetimeIndex)):
        datetimes = datetimes.to_pydatetime()
    elif isinstance(datetimes, np.ndarray):
        if np.issubdtype(datetimes.dtype, np.datetime64):
            datetimes = pd.to_datetime(datetimes).to_pydatetime()
        elif np.issubdtype(datetimes.dtype, np.object_):
            # Assume array of datetime objects, no change needed
            pass
        else:
            raise ValueError('Array must be datetime64 or array of datetime objects.')
    else:
        raise ValueError('Input must be pandas Series, DatetimeIndex, or numpy array of datetime objects.')

    # Create a mask for valid (non-NaT) entries
    valid_mask = pd.notna(datetimes)

    # Prepare output array with fill values (int64)
    result = np.full(len(datetimes), CDF_EPOCH_FILLVAL, dtype=np.int64)

    # Convert valid datetimes using pycdf.lib.v_datetime_to_epoch
    if valid_mask.any():
        valid_datetimes = np.array(datetimes)[valid_mask]
        # Returns float64 microseconds
        valid_epoch_floats = pycdf.lib.v_datetime_to_epoch(valid_datetimes)
        # Convert to int64
        valid_epoch_ints = valid_epoch_floats.astype(np.int64)
        result[valid_mask] = valid_epoch_ints

    return result


def cdf_epoch_to_datetime(cdf_epochs):
    # Equivalent to Ticktock(cdf_epochs, 'CDF').UTC

    if isinstance(cdf_epochs[0], (int, float)):
        return pycdf.lib.v_epoch_to_datetime(cdf_epochs)
    return cdf_epochs