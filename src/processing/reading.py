# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:08:23 2025

@author: richarj2
"""

import pandas as pd
from spacepy import pycdf

from .filtering import exclude_days, filter_data
from .handling import get_processed_files
from .dataframes import set_df_indices



def import_processed_data(directory, year=None, date_range=None, bad_data=None):
    """
    Imports and processes CDF files from a specified directory, filtering by keyword and date range.

    Parameters
    ----------
    directory : str
        The path to the directory containing the CDF files.

    file_keyword : str
        A keyword to filter the CDF files by. Only files whose names contain this keyword will be processed.

    start : datetime
        The start of the date range for filtering the CDF files. Only data from files within this range will be included.

    end : datetime
        The end of the date range for filtering the CDF files. Only data from files within this range will be included.

    time_col : str, optional
        The name of the column containing time data in CDF epoch format. Defaults to 'epoch'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the merged and filtered data from the CDF files. The DataFrame will have a 'time' column
        (converted from CDF epoch to datetime) as the index, and it will be filtered to the specified date range.
    """
    # Find all .cdf files containing the specified keyword in their names and within the date range
    cdf_files = get_processed_files(directory, year)

    merged_data = pycdf.concatCDF([pycdf.CDF(f) for f in cdf_files])
    df = read_spacepy_object(merged_data)
    for column in df.columns:
        if df.attrs['units'].get(column, '') == 'STRING':
            df[column] = df[column].str.strip()

    with pycdf.CDF(cdf_files[0]) as cdf:
        global_attrs = {}
        crossings_attrs = {}
        for key, value in cdf.attrs.items():
            if 'crossings' in key:
                number = int(key.replace('crossings_', ''))
                crossings_attrs[number] = str(value).strip()
            else:
                global_attrs[key] = str(value[0]) if isinstance(value, list) and len(value) == 1 else str(value)

        df.attrs['global'] = global_attrs
        if crossings_attrs:
            df.attrs['crossings'] = crossings_attrs

    # Removes any placeholder dates
    time_col = df.attrs['global'].get('time_col','epoch')
    placeholder_dates = [pd.Timestamp('9999-12-13 23:59:59.999'),pd.Timestamp('9999-12-13 23:59:59.998')]
    if time_col!='none':
        placeholder_date = pd.Timestamp('9999-12-13 23:59:59.999')
        set_df_indices(df, time_col)  # Sets the index as datetime

    for placeholder_date in placeholder_dates:
        df = df.mask(df == placeholder_date, pd.NaT)


    if bad_data is not None:
        exclude_days(df, bad_data)
    if date_range is not None:
        filter_data(df, 'index', date_range[0], date_range[1])  # Slices data to be in desired time range

    return df


def read_spacepy_object(cdf_data):
    """
    Reads data from a merged CDF object and stores it in a pandas DataFrame.

    Parameters
    ----------
    cdf_data : spacepy.pycdf.CDF
        The merged CDF object containing the data to be read.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all data from the merged CDF, with each variable as a column.
        The units for each variable are stored as attributes in the DataFrame.
    """
    data_dict = {}
    units_dict = {}

    # Extract data and units from each variable in the CDF
    for col in cdf_data:
        data_dict[col] = cdf_data[col][...]
        units_dict[col] = cdf_data[col].attrs.get('units', None)  # Extract units if available

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame(data_dict)
    df.attrs['units'] = units_dict  # Stores the units

    return df