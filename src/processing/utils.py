# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:32:02 2025

@author: richarj2
"""
import os
import re

import numpy as np
import pandas as pd

from spacepy import pycdf


UNIT_MAP = {
    r'^(B_[XYZ]_(GSE|GSM))': 'nT',
    r'^(V_[XYZ]_(GSE|GSM))': 'km/s',
    r'^(E_[XYZ]_(GSE|GSM))': 'mV/m',
    r'^(S_[XYZ]_(GSE|GSM))': 'uW/m2',
    r'(theta|phi|alpha|pitch|clock|cone|angle)': 'rad',
    r'^(Nx|Ny|Nz|M_|Beta_|beta)': '1',
    r'(ratio|coeff|beta,r_F)': '1',
    r'(mode|quality|_num)': 'NUM',
    r'_time$': 'datetime',
    r'^(N_|ni|np|n_i|n_p)': 'n/cc',
    r'(unc_s|_delay|_s|_duration)$': 's',
    r'^(r_|R_)': 'Re',
    r'^(B_|AE|AU|AL|SM|AA)': 'nT',
    r'^(V_|v_)': 'km/s',
    r'^(E_|PC)': 'mV/m',
    r'^T_': 'K',
    r'^P_': 'nPa',
    r'^S_': 'uW/m2',
    r'epoch': 'cdf_epoch',
}

def add_unit(key):
    for pattern, unit in UNIT_MAP.items():
        if re.search(pattern, key):
            return unit
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