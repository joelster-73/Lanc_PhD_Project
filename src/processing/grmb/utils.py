# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:22:42 2025

@author: richarj2
"""
import numpy as np

def calculate_region_duration(df, time_col='epoch', duration_label='region_duration'):
    """
    Calculate the time spent between consecutive rows and add it as a new column 'region_duration'.

    The time spent for the first row will be the difference between the first and second row,
    and the last row will have 'NaN' for time spent.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the time column (`time_col`).
    time_col : str, optional
        The name of the column containing the timestamps. Default is 'epoch'.

    Returns:
    --------
    df : pandas.DataFrame
        The original DataFrame with the 'region_duration' column added.
    """
    df[duration_label] = df[time_col].diff().dt.total_seconds()
    df[duration_label] = df[duration_label].shift(-1)
    df.loc[df.index[-1], duration_label] = np.nan

    return df



def time_each_region(df, region_label='loc_num', duration_label='region_duration'):

    df = df.dropna(subset=[duration_label])

    time_per_label = df.groupby(region_label)[duration_label].sum()
    time_per_label_years = time_per_label / (60 * 60 * 24 * 365.25)

    time_dict = time_per_label_years.to_dict()

    return time_dict
