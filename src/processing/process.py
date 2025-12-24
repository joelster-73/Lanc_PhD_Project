# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 20:11:20 2025

@author: richarj2
"""

import os

import pandas as pd

from .handling import create_log_file
from .dataframes import resample_data
from .writing import write_to_cdf
from .utils import create_directory

from ..config import R_E, get_proc_directory


def process_overlapping_files(spacecraft, data, process_func, variables_dict, files_dict, sample_intervals, filter_func=None, **kwargs):

    overwrite   = kwargs.get('overwrite',True)
    time_col    = kwargs.get('time_col','epoch')
    resolutions = kwargs.get('resolutions',{})

    if not variables_dict:
        raise ValueError(f'No valid variables dict for {data}')

    ###----------SET UP----------###

    print(f'Processing {spacecraft.upper()}')

    directory_name = data.upper()

    save_directory = get_proc_directory(spacecraft, dtype=data, create=True)
    log_file_path = os.path.join(save_directory, f'{directory_name}_files_not_added.txt')  # Stores not loaded files
    create_log_file(log_file_path)

    save_directories = {}

    for sample_interval in sample_intervals:

        save_directory = get_proc_directory(spacecraft, dtype=data, resolution=sample_interval, create=True)
        create_directory(save_directory)
        save_directories[sample_interval] = save_directory

    if not variables_dict:
        raise ValueError(f'No valid variables dict for {data}')

    ###----------PROCESS----------###
    attributes = {'time_col': time_col, 'R_E': R_E}
    next_key_df = pd.DataFrame()
    for k_i, (key, files) in enumerate(files_dict.items()):

        print(f'Processing {key} data.')
        key_df = process_func(variables_dict, files, directory_name, log_file_path, time_col=time_col, **kwargs)
        if key_df.empty:
            continue

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

        print(f'{key} processed.')
        # resample and write to file
        for sample_interval, samp_dir in save_directories.items():

            if sample_interval in ('raw','spin','fast'):
                sampled_df = key_df

                attributes['sample_interval'] = resolutions.get(sample_interval,sample_interval)
            else:
                if filter_func:
                    sampled_df = filter_func(sampled_df)
                # resample and write to file
                print('Resampling...')
                sampled_df = resample_data(key_df, time_col, sample_interval)
                print(f'{sample_interval} resampled.')
                attributes['sample_interval'] = sample_interval

            write_to_cdf(sampled_df, directory=samp_dir, file_name=f'{directory_name}_{key}', attributes=attributes, overwrite=overwrite)