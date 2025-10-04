# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:42:01 2025

@author: richarj2
"""
import os
import glob
import numpy as np
import pandas as pd

from spacepy import pycdf
import spacepy.pycdf.istp

from ..dataframes import add_df_units, resample_data
from ..handling import create_log_file, log_missing_file
from ..writing import write_to_cdf
from ..utils import create_directory

from ...coordinates.magnetic import calc_B_GSM_angles

from ...config import R_E

def process_mms_files(directory, data_directory, variables, sample_intervals='1min', time_col='epoch', year=None, overwrite=True):

    print('Processing MMS.')

    ###----------DIRECTORIES----------###

    directory_name = os.path.basename(os.path.normpath(directory))
    if 'FGM' in directory_name:
        save_directory = os.path.join(data_directory, 'field')
    else:
        save_directory = os.path.join(data_directory, 'plasma')

    log_file_path = os.path.join(save_directory, f'{directory_name}_files_not_added.txt')  # Stores not loaded files
    create_log_file(log_file_path)

    #raw_dir = os.path.join(save_directory, 'raw')
    #create_directory(raw_dir)

    if isinstance(sample_intervals,str):
        sample_intervals = (sample_intervals,)

    for sample_interval in sample_intervals:

        samp_dir = os.path.join(save_directory, sample_interval)
        create_directory(samp_dir)

    ###----------FILES----------###

    files_by_year = get_mms_files(directory, year)

    for the_year, files in files_by_year.items():
        print(f'Processing {the_year} data.')
        state_list = []
        field_list = []

        # Loop through each daily file in the year
        for cdf_file in files:
            file_date = os.path.basename(cdf_file).split('_')[4]
            try:  # Bad data check
                state_dict, field_dict = extract_mms_data(cdf_file, variables)
                if not state_dict or not field_dict:
                    log_missing_file(log_file_path, file_date, 'Empty file.')
                    continue

                state_list.append(pd.DataFrame(state_dict))
                field_list.append(pd.DataFrame(field_dict))

            except Exception as e:
                log_missing_file(log_file_path, file_date, e)

            print(f'{file_date} read.')

        ###---------------COMBINING------------###

        if not state_list or not field_list:
            print(f'No data for {the_year}')
            continue

        yearly_state_df = pd.concat(state_list, ignore_index=True)
        add_df_units(yearly_state_df)
        del state_list

        yearly_field_df = pd.concat(field_list, ignore_index=True)
        yearly_field_df = yearly_field_df.loc[yearly_field_df['B_flag']==0] # Quality 0 = no problems
        yearly_field_df.drop(columns=['B_flag'],inplace=True)

        gsm = calc_B_GSM_angles(yearly_field_df, time_col=time_col)
        yearly_field_df = pd.concat([yearly_field_df, gsm], axis=1)
        add_df_units(yearly_field_df)
        del field_list

        # resample and write to file
        print('Resampling.')

        for sample_interval in sample_intervals:
            samp_dir = os.path.join(save_directory, sample_interval)

            merged_chunks = []

            for month_num in range(1, 13):
                print(f'Month {month_num}')

                state_mask = yearly_state_df[time_col+'_pos'].dt.month == month_num
                field_mask = yearly_field_df[time_col].dt.month == month_num
                if np.sum(state_mask)==0 or np.sum(field_mask)==0:
                    continue

                state_month = resample_data(yearly_state_df.loc[state_mask], time_col+'_pos', sample_interval)
                field_month = resample_data(yearly_field_df.loc[field_mask], time_col, sample_interval)

                merged_chunk = pd.merge(state_month, field_month, how='outer', on=time_col)
                merged_chunks.append(merged_chunk)

            merged_df = pd.concat(merged_chunks, ignore_index=True)
            add_df_units(merged_df)
            del merged_chunks

            print(f'{sample_interval} reprocessed.')

            output_file = os.path.join(samp_dir, f'{directory_name}_{the_year}.cdf')
            attributes = {'sample_interval': sample_interval, 'time_col': time_col, 'R_E': R_E}
            write_to_cdf(merged_df, output_file, attributes, overwrite)

        print(f'{the_year} processed.')


def get_mms_files(directory=None, year=None):

    files_by_year = {}

    if directory is None:
        directory = os.getcwd()

    for year_folder in sorted(os.listdir(directory)):
        if year and year_folder != str(year):
            continue
        elif not os.path.isdir(os.path.join(directory, year_folder)):
            continue

        for month_folder in sorted(os.listdir(os.path.join(directory, year_folder))):

            pattern = os.path.join(directory, year_folder, month_folder, '*.cdf')
            for cdf_file in sorted(glob.glob(pattern)):
                files_by_year.setdefault(year_folder, []).append(cdf_file)


    if year and str(year) not in files_by_year:
        raise ValueError(f'No files found for {year}.')
    if not files_by_year:
        raise ValueError('No files found.')

    return files_by_year

def extract_mms_data(cdf_file, variables):

    state_dict = {}
    field_dict = {}

    with pycdf.CDF(cdf_file) as cdf:

        for var_name, var_code in variables.items():
            if var_name in ('epoch_pos','r_gse'):
                data_dict = state_dict
            else:
                data_dict = field_dict
            try:
                if var_name not in ('epoch','epoch_pos'):
                    spacepy.pycdf.istp.nanfill(cdf[var_code])
                data = cdf[var_code][...]
            except: # Variable not in file
                continue

            if isinstance(data,float) or isinstance(data,int):
                # Empty dataset
                return {}, {}

            ###-----------CHECKING LENGTHS----------###

            if data.ndim == 2 and data.shape[1] == 4:  # Assuming a 2D array for vector mag and components

                suffix = 'avg' # For consistency with other spacecraft
                if var_name == 'r_gse':
                    data /= R_E  # Scales distances to multiples of Earth radii
                    suffix = 'mag'

                if '_gse' in var_name:
                    coords = 'GSE'
                elif '_gsm' in var_name:
                    coords = 'GSM'
                else:
                    raise Exception(f'Coord system of variable not implemented: {var_name}.')
                field = var_name.split('_')[0]

                if f'{field}_{suffix}' not in data_dict: # stops redundant write
                    data_dict[f'{field}_{suffix}']   = data[:, 3]

                if coords=='GSE' or f'{field}_x_GSE' not in data_dict: # doesn't add X_GSM
                    data_dict[f'{field}_x_{coords}'] = data[:, 0]

                data_dict[f'{field}_y_{coords}'] = data[:, 1]
                data_dict[f'{field}_z_{coords}'] = data[:, 2]

            else:
                data_dict[var_name] = data  # pycdf extracts as datetime, no conversion from epoch needed

    return state_dict, field_dict