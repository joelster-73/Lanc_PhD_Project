# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 20:11:20 2025

@author: richarj2
"""

import os

import numpy as np
import pandas as pd

from .handling import create_log_file
from .dataframes import resample_data
from .writing import write_to_cdf
from .utils import create_directory

from ..config import R_E, get_proc_directory

from .writing import resample_cdf_files
from .handling import get_file_keys, refactor_keys


def resample_files(spacecraft, data, raw_res='spin', new_grouping='yearly', **kwargs):
    """
    Resample monthly files (as well as yearly files) into yearly files at a lower resolution, e.g. 1min, 5min.
    """

    files_by_keys = get_file_keys(spacecraft, data, raw_res)
    files_by_year = refactor_keys(files_by_keys, new_grouping)

    kwargs['files_by_keys'] = files_by_year

    resample_cdf_files(spacecraft, data, raw_res=raw_res, **kwargs)


# %% Process

def process_overlapping_files(spacecraft, data, process_func, variables_dict, files_dict, sample_intervals, qual_func=None, **kwargs):

    overwrite   = kwargs.get('overwrite',True)
    time_col    = kwargs.get('time_col','epoch')
    resolutions = kwargs.get('resolutions',{})

    if not variables_dict:
        raise ValueError(f'No valid variables dict for -{data}-.')

    ###----------SET UP----------###

    print(f'Processing {spacecraft.upper()}.')

    directory_name = data.upper()

    save_directory = get_proc_directory(spacecraft, dtype=data, create=True)
    log_file_path = os.path.join(save_directory, f'{directory_name}_files_not_added.txt')  # Stores not loaded files
    create_log_file(log_file_path)

    save_directories = {}

    for sample_interval in sample_intervals:

        save_directory = get_proc_directory(spacecraft, dtype=data, resolution=sample_interval, create=True)
        create_directory(save_directory)
        save_directories[sample_interval] = save_directory

    ###----------PROCESS----------###
    attributes = {'time_col': time_col, 'R_E': R_E}
    next_key_df = pd.DataFrame()
    for k_i, (key, files) in enumerate(files_dict.items()):

        print(f'Processing {key} data.')
        kwargs_key = kwargs.copy()
        kwargs_key['qual_files'] = kwargs.get('qual_files',{}).get(key,[])

        key_df = process_func(variables_dict, files, directory_name, log_file_path, time_col=time_col, **kwargs_key)
        if key_df.empty:
            continue
        df_attrs = key_df.attrs

        # Files overlap into next day
        if not next_key_df.empty:
            key_df = pd.concat([key_df,next_key_df])
            key_df.sort_index(inplace=True)

            next_key_df = pd.DataFrame()

        if k_i != len(files_dict)-1:

            if isinstance(key, int): # key is year
                keep = (key_df[time_col].dt.year==key)
            elif isinstance(key,str): # key is year-month
                the_year, the_month = key.split('-')
                keep = (key_df[time_col].dt.year==int(the_year)) & (key_df[time_col].dt.month==int(the_month))

            next_key_df = key_df.loc[~keep] # store for next key
            key_df      = key_df.loc[keep]

        key_df.attrs = df_attrs

        print(f'{key} processed.')
        # resample and write to file
        for sample_interval, samp_dir in save_directories.items():

            sampled_df = key_df

            if sample_interval in ('raw','spin','fast'):
                res = resolutions.get(sample_interval,sample_interval)

            else:
                res = sample_interval
                if qual_func:
                    sampled_df = qual_func(sampled_df)
                else:
                    print('No quality filter function.')

                # resample and write to file
                sampled_df = resample_data(sampled_df, time_col, sample_interval, drop_nans=False)

            attributes['sample_interval'] = res
            write_to_cdf(sampled_df, directory=samp_dir, file_name=f'{directory_name}_{key}', attributes=attributes, overwrite=overwrite)


def format_extracted_vector(dictionary, data, var_name, coords='GSE', suffix='', fourD=False):

    if suffix != '': # ion or unc
        suffix = f'_{suffix}'

    if f'{var_name}_mag' not in dictionary:
        dictionary[f'{var_name}_mag']   = np.sqrt(data[:,0]**2+data[:,1]**2+data[:,2]**2)

    if fourD:
        dictionary[f'{var_name}_avg'] = data[:, 3]

    if coords=='GSE' or f'{var_name}_x_GSE{suffix}' not in dictionary: # doesn't add X_GSM
        dictionary[f'{var_name}_x_{coords}{suffix}'] = data[:, 0]

    dictionary[f'{var_name}_y_{coords}{suffix}'] = data[:, 1]
    dictionary[f'{var_name}_z_{coords}{suffix}'] = data[:, 2]


