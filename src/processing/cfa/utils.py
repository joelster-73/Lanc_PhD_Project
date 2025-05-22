# -*- coding: utf-8 -*-
"""
Created on Mon May 12 11:39:53 2025

@author: richarj2
"""

import numpy as np

from datetime import datetime, timedelta
from uncertainties import ufloat



def calculate_shock_datetime(year, fractional_day, seconds_of_day):
    # Floor the fractional day to the nearest whole day
    whole_day = np.floor(float(fractional_day))
    # Start of the year + whole_day as timedelta
    start_of_year = datetime(int(year), 1, 1)
    day_datetime = start_of_year + timedelta(days=whole_day - 1)  # Subtract 1 for day-of-year
    # Add seconds of the day as timedelta
    secs, unc = split_uncertainty(seconds_of_day)
    time_delta = timedelta(seconds=secs)
    return day_datetime + time_delta, unc

def split_uncertainty(the_string, return_type='tuple'):
    value, uncertainty = the_string.split('±')

    # Strip any extra whitespace or return NaN
    try:
        value = float(value.strip())
    except:
        value = np.nan

    try:
        uncertainty = float(uncertainty.strip())
    except:
        uncertainty = np.nan

    if return_type == 'tuple':
        return value, uncertainty
    elif return_type == 'ufloat':
        ufloat(value,uncertainty)

