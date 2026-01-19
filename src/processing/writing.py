# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:08:31 2025

@author: richarj2
"""
import os
import numpy as np
import pandas as pd

from spacepy import pycdf

from .utils import datetime_to_cdf_epoch, add_unit, create_directory
from .handling import get_processed_files
from .reading import import_processed_data
from .dataframes import resample_data, add_df_units
from ..config import R_E, get_proc_directory


def write_to_cdf(df, output_file=None, directory=None, file_name=None, attributes=None, overwrite=True, append_rows=False, update_column=False, reset_index=False):
    """
    Writes a pandas dataframe to a CDF file.
    If the dataframe is indexed by time, must set 'reset_index' to True.
    """

    df_attrs = df.attrs.copy()
    df = df.copy()
    df.attrs = df_attrs

    if attributes is not None:
        for key, val in attributes.items():
            df.attrs[key] = val
    attributes = df.attrs

    # Check if all columns have the same length
    column_lengths = df.apply(len)
    if column_lengths.nunique() > 1:
        raise ValueError(f'Columns have different lengths: {column_lengths.to_dict()}')

    # If the time is the index, moves into column
    if reset_index:
        df.reset_index(inplace=True)
    time_col = attributes.get('time_col','epoch')

    if not any((output_file,directory,file_name)):
        raise ValueError('No suitable directory/file name passed in.')
    elif output_file is None:
        output_file = os.path.join(directory, file_name)

    # Adds ".cdf" if not there
    root, ext = os.path.splitext(output_file)
    if not ext:
       ext = '.cdf'
    output_file = root + ext

    # Creates directory folder and checks if file already exists
    create_directory(os.path.dirname(output_file))
    if overwrite and os.path.exists(output_file):
        os.remove(output_file)

    with pycdf.CDF(output_file, create=(not os.path.exists(output_file))) as cdf:
        cdf.readonly(False)

        if update_column and (len(df) != len(cdf['epoch'][...])):
            raise Exception(f'New data ({len(df)}) and cdf file ({len(cdf["epoch"][...])}) not same length')

        # Write each column in the DataFrame to the CDF file
        for column in df.columns:
            new_data = df[column].to_numpy()
            if column not in cdf or update_column:
                unit = df.attrs.get('units',{}).get(column,None)
                if unit is None:
                    unit = add_unit(column)

                try:
                    if column == time_col or unit == 'datetime':
                        new_data = datetime_to_cdf_epoch(new_data)
                        cdf.new(column, data=new_data, type=pycdf.const.CDF_EPOCH)

                    elif isinstance(new_data[0], str) or unit == 'STRING':  # Check if data is a string (unicode or bytes)
                        max_len = max(len(s) for s in new_data if isinstance(s,str))
                        padded_data = np.array([s.ljust(max_len) for s in new_data], dtype=f'U{max_len}')
                        # Store the data as CDF_CHAR with variable length
                        cdf.new(column, data=padded_data, type=pycdf.const.CDF_CHAR)
                        df.attrs['units'][column] = 'STRING'

                    elif isinstance(new_data[0], list) or unit == 'LIST':  # Check if data is a list (unicode or bytes)
                        new_data_str = np.array([','.join(lst) for lst in new_data])
                        max_len = max(len(s) for s in new_data_str if isinstance(s,str))
                        padded_data = np.array([s.ljust(max_len) for s in new_data_str], dtype=f'U{max_len}')
                        # Store the data as CDF_CHAR with variable length
                        cdf.new(column, data=padded_data, type=pycdf.const.CDF_CHAR)
                        df.attrs['units'][column] = 'LIST'

                    else:
                        cdf.new(column, data=new_data)
                except:
                    print(f'Cannot add "{column}" to cdf file.')

                else:
                    cdf[column].attrs['units'] = str(unit)

            elif append_rows:
                # If the column already exists, extend the existing data
                cdf[column] = np.concatenate((cdf[column][...], new_data))

        # Adds attributes
        for key, value in attributes.items():
            if key=='units':
                continue
            if isinstance(value, dict):
                cdf.attrs[key] = {}
                for sub_key, sub_value in value.items():
                    cdf.attrs[key][f'{sub_key}'] = sub_value
            else:
                # If it's not a dictionary, just assign the value
                cdf.attrs[key] = value

    print('Data written to file.\n')


def add_columns_to_cdf(new_df, update_file, update=False):

    write_to_cdf(new_df, update_file, overwrite=False, update_column=update)


    print('File has been updated.\n')


def update_cdf_attributes(directory, new_values, add=True):

    # Iterate through all files in the directory
    cdf_files = get_processed_files(directory)

    # Use glob to find files matching the pattern
    for cdf_file in cdf_files:
            try:
                # Open the CDF file
                with pycdf.CDF(cdf_file, create=False) as cdf:
                    cdf.readonly(False)
                    # Check if the attribute exists
                    for key, value in new_values.items():
                        if key in cdf.attrs or add:
                            cdf.attrs[key] = value

            except Exception as e:
                print(f'Error processing {cdf_file}: {e}')

    print('Attribute update complete.')

def resample_cdf_files(spacecraft, data, raw_res='spin', sample_intervals=('1min',), time_col='epoch', overwrite=True, qual_func=None, files_by_keys={}):
    """
    Resample monthly files (as well as yearly files) into yearly files at a lower resolution, e.g. 1min, 5min.
    """
    ###----------SET UP----------###
    print('Resampling.')

    save_directories = {}

    for sample_interval in sample_intervals:
        save_directory = get_proc_directory(spacecraft, dtype=data, resolution=sample_interval, create=True)
        create_directory(save_directory)
        save_directories[sample_interval] = save_directory

    ###----------PROCESS----------###
    for key, files in files_by_keys.items():

        ############ TEMPORARY ##########
        year, month = key.split('-')
        if int(year)<2018:
            print(f'Skipping {key} - REMOVE TEMP')
            continue
        elif int(year)==2018 and int(month)<7:
            print(f'Skipping {key} - REMOVE TEMP')
            continue

        print(f'Updating {key}.')
        yearly_list = []

        for file in files:
            df = import_processed_data(spacecraft, dtype=data, resolution=raw_res, file_name=file)
            yearly_list.append(df)

        dir_name = '_'.join(files[0].split('_')[:-1])

        yearly_df = pd.concat(yearly_list) # don't set ignore_index to True
        yearly_df.drop(columns=[c for c in yearly_df.columns if c.endswith('_unc')],inplace=True) # measurement error << statistical uncertainty
        add_df_units(yearly_df)

        if qual_func: # filter quality etc
            yearly_df = qual_func(yearly_df)

        for sample_interval, samp_dir in save_directories.items():

            sampled_df = resample_data(yearly_df, time_col='index', sample_interval=sample_interval, drop_nans=False)

            attributes = {'sample_interval': sample_interval, 'time_col': time_col, 'R_E': R_E}
            write_to_cdf(sampled_df, directory=samp_dir, file_name=f'{dir_name}_{key}', attributes=attributes, overwrite=overwrite, reset_index=True)

