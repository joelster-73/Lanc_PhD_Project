# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:08:23 2025

@author: richarj2
"""
import warnings
from spacepy.pycdf import CDFWarning
from collections import Counter, defaultdict

import pandas as pd
from spacepy import pycdf
from contextlib import ExitStack

from .handling import get_processed_files, get_cdf_file
from .dataframes import set_df_indices, merge_dataframes

from .omni.update import update_omni
from .mag.config import PC_STATIONS, resolutions, native_resolutions

from ..config import get_proc_directory, get_data_populations


warnings.filterwarnings('ignore', category=CDFWarning)

def import_processed_spacecraft(spacecraft, populations=None, resolution='1min', year=None, region='sw'):

    if populations is None:
        populations = get_data_populations(spacecraft, 'plasma', region)

    if len(populations)==1:
        return import_processed_data(spacecraft, dtype=populations[0], resolution=resolution, year=year)

    all_data = []
    for data in populations:
        df_sc = import_processed_data(spacecraft, dtype=data, resolution=resolution, year=year)
        all_data.append(df_sc)

    return merge_dataframes(*all_data, warn=False)

def import_processed_omni(resolution='1min', file_name=None, year=None):
    """
    A wrapper to import omni data with the update procedure called to match the columns with the spacecraft data.
    """
    omni = import_processed_data('omni', resolution=resolution, file_name=file_name, year=year)

    update_omni(omni)

    return omni

def import_processed_index(index, resolution='1min', return_series=True, return_full=False):
    """
    Wrapper function to import a processed index as the structure depends on the index
    """
    native_res  = native_resolutions.get(index,'1min')
    if resolutions.index(resolution) < resolutions.index(native_res):
        warnings.warn(f'Resolution {resolution} is not available as the native resolution is {native_res}.')
        resolution = native_res

    index_parents = {'SYM_D': 'SYM', 'SYM_H': 'SYM', 'ASY_D': 'ASY', 'ASY_H': 'ASY', 'PCC': 'PC', 'PCN': 'PC', 'AL': 'AE', 'AU': 'AE', 'SMU': 'SME', 'SML': 'SME'}
    index_parents |= {f'{station}_mag': station for station in PC_STATIONS} | {f'{station}_y_GSE': station for station in PC_STATIONS} | {f'{station}_y_GSM': station for station in PC_STATIONS}

    df = import_processed_data('indices', resolution=resolution, file_name=index_parents.get(index,index))
    df = df[[col for col in df.columns if col.startswith(index)]] # retains counts and errs, discards others

    if return_full:
        return df

    elif return_series:
        return df.loc[:,index]

    return df.loc[:,[index]]

# %% reader

def import_processed_data(source, dtype=' ', resolution=' ', file_name=None, year=None, axis=0):

    directory = get_proc_directory(source, dtype, resolution)
    if directory=='EMPTY':
        return pd.DataFrame()

    # axis = 0 combines files that have (mostly) the same columns at different times
    # axis = 1 combines files that have different columns

    if file_name:
        cdf_file = get_cdf_file(directory, filename=file_name)
        df = read_spacepy_object(cdf_file)

    else: # Find all .cdf files containing the specified keyword in their names and within the date range
        cdf_files = get_processed_files(directory, year)
        cdf_file  = cdf_files[0]

        if axis==1: # merging files with different columns
            df = read_different_structures(cdf_files, axis)

        else:

            # Groups cdf files by their column structure
            groups = defaultdict(list)

            for f in cdf_files:
                groups[get_cdf_structure(f)].append(f)

            if len(groups) == 1:
                df = read_spacepy_object(cdf_files)

            else:

                if not ((source, dtype) == ('thb', 'sw') or (source, resolution) == ('omni', '15min')):

                    # (1) All of E and S were NaN when resampling for 2008
                    # (2) 2020 AE/AL/AU values NaN in 1-min/5-min, which are dropped when averaged to 15 min

                    diagnose_different_file(cdf_files, read_spacepy_object)
                    raise Exception(f'Different file structures: {source} {dtype} {resolution}.')

                df = read_different_structures(groups.values(), axis)

    # Tidying up
    split_columns(df)
    extract_crossing_attrs(cdf_file, df)

    # Removes any placeholder dates
    time_col = df.attrs.get('time_col','epoch')
    placeholder_dates = [pd.Timestamp('9999-12-31 23:59:59.999'),pd.Timestamp('9999-12-31 23:59:59.998')]
    if time_col!='none':
        placeholder_date = pd.Timestamp('9999-12-31 23:59:59.999')
        set_df_indices(df, time_col)  # Sets the index as datetime

    df = df.loc[~df.index.duplicated(keep='last')]

    for placeholder_date in placeholder_dates:
        df = df.mask(df == placeholder_date, pd.NaT)

    return df

# %% wrappers

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

def read_different_structures(cdf_files, axis):
    """
    When CDF files are expected to have different structures, this reads each file as a separate object and concats.
    """

    dfs = [read_spacepy_object(f) for f in cdf_files]

    df = pd.concat(dfs, axis=axis, ignore_index=(axis==0))

    attrs = {}
    for df_file in dfs:
        for key, value in df_file.attrs.items():
            if key == 'units':
                attrs.setdefault('units', {}).update(value)
            else:
                attrs[key] = value

    df.attrs = attrs

    return df

# %% misc

def get_cdf_structure(file):
    with pycdf.CDF(file) as cdf:
        return frozenset(cdf.keys())

def diagnose_different_file(files, read_func):
    """
    Print files with columns that are different to the rest.
    """
    dfs = [(f, read_func(f)) for f in files]
    structures = Counter(frozenset(df.columns) for _, df in dfs)
    expected = structures.most_common(1)[0][0]

    for f, df in dfs:
        cols = set(df.columns)
        if cols != expected:
            print(f'\nDifferent structure in file: {f}')
            print(f'Missing columns: {expected - cols}')
            print(f'Extra columns:   {cols - expected}')


def split_columns(df):

    for column in df.columns:

        if df.attrs.get('units',{}).get(column, '') == 'STRING':
            df[column] = df[column].str.strip()
        elif df.attrs.get('units',{}).get(column,'') == 'LIST':
            str_series = df[column].str.strip()
            df[column] = [s.split(',') for s in str_series]


def extract_crossing_attrs(cdf_file, df):

    with pycdf.CDF(cdf_file) as cdf:
        crossings_attrs = {}
        for key, value in cdf.attrs.items():
            if 'crossings' in key:
                number = int(key.replace('crossings_', ''))
                crossings_attrs[number] = str(value).strip()
            else:
                df.attrs[key] = str(value[0]) if isinstance(value, list) and len(value) == 1 else str(value)

        if crossings_attrs:
            df.attrs['crossings'] = crossings_attrs