# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:08:23 2025

@author: richarj2
"""

import pandas as pd
from spacepy import pycdf
from contextlib import ExitStack

from .handling import get_processed_files, get_cdf_file
from .dataframes import set_df_indices
from ..config import get_proc_directory


def import_processed_data(source, sample=' ', dtype=' ', resolution=' ', file_name=None, year=None, axis=0):

    directory = get_proc_directory(source, sample, dtype, resolution)

    # axis = 0 combines files that have the same columns at different times
    # axis = 1 combines files that have different columns

    if file_name:
        cdf_file = get_cdf_file(directory, filename=file_name)
        df = read_spacepy_object(cdf_file)

    else: # Find all .cdf files containing the specified keyword in their names and within the date range
        cdf_files = get_processed_files(directory, year)
        cdf_file = cdf_files[0]

        if len(cdf_files)==1:
            df = read_spacepy_object(cdf_files)

        else:
            try:
                df = read_spacepy_object(cdf_files)
            except Exception: # CDF files for different parameters
                print('Different file structures')
                df = pd.DataFrame()
                for f in cdf_files:
                    df_param = read_spacepy_object(f)
                    df = pd.concat([df, df_param], axis=axis)
                    for k, v in df_param.attrs.items():
                        if k not in df.attrs:
                            df.attrs[k] = v

    for column in df.columns:
        if df.attrs.get('units',{}).get(column, '') == 'STRING':
            df[column] = df[column].str.strip()
        elif df.attrs.get('units',{}).get(column,'') == 'LIST':
            str_series = df[column].str.strip()
            df[column] = [s.split(',') for s in str_series]

    with pycdf.CDF(cdf_file) as cdf:
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
    placeholder_dates = [pd.Timestamp('9999-12-31 23:59:59.999'),pd.Timestamp('9999-12-31 23:59:59.998')]
    if time_col!='none':
        placeholder_date = pd.Timestamp('9999-12-31 23:59:59.999')
        set_df_indices(df, time_col)  # Sets the index as datetime

    ########
    # weighted average overlapping year boundaries
    # for time being just drop earlier one
    #######
    df = df.loc[~df.index.duplicated(keep='last')]

    for placeholder_date in placeholder_dates:
        df = df.mask(df == placeholder_date, pd.NaT)

    return df


def read_spacepy_object(file_paths):

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    data_dict = {}
    units_dict = {}

    with ExitStack() as stack:
        cdfs = [stack.enter_context(pycdf.CDF(f)) for f in file_paths]

        # If multiple files: concatCDF
        if len(cdfs) > 1:
            cdf_data = pycdf.concatCDF(cdfs)
        else:
            cdf_data = cdfs[0]

        # Extract data + units
        for col in cdf_data:
            data_dict[col] = cdf_data[col][...]
            units_dict[col] = cdf_data[col].attrs.get('units', None)

    # Build DataFrame outside the with-block (CDFs closed now)
    df = pd.DataFrame(data_dict)
    df.attrs['units'] = units_dict

    return df