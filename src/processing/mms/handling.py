# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:42:01 2025

@author: richarj2
"""
import os
import glob

import numpy as np
import pandas as pd
from scipy.constants import k as kB, e
from scipy.constants import physical_constants
m_a = physical_constants['alpha particle mass'][0]

from spacepy import pycdf
import spacepy.pycdf.istp

from .config import ION_MASS_DICT, ION_SPECIES, VARIABLES_DICT

from ..handling import log_missing_file
from ..dataframes import add_df_units
from ..process import process_overlapping_files

from ...coordinates.magnetic import calc_B_GSM_angles
from ...config import R_E, get_luna_directory

def process_mms_files(spacecraft, data, sample_intervals=('raw',), year=None, **kwargs):

    directory = get_luna_directory(spacecraft, data)

    # Process function
    if data=='fgm':
        files_dict = get_mms_files_month(directory, year)
        process = process_mms_fgm
        def filtering(df):
            return filter_quality(df, column='B_flag')
    elif data=='state':
        files_dict = get_mms_files_year(directory, year)
        process = process_mms_state
        filtering = None
    elif data=='hpca':
        files_dict = get_mms_files_year(directory, year)
        process = process_mms_hpca
        if 'raw' in sample_intervals:
            kwargs['keep_ions'] = True
            sample_intervals = ('raw',) # prevents resampling
        filtering = None
    elif data=='fpi':
        files_dict = get_mms_files_month(directory, year)
        process = process_mms_fpi
        def filtering(df):
            return filter_quality(df, column='flag')
    else:
        raise ValueError(f'"{data}" not valid data to sample')

    variables = VARIABLES_DICT.get(data,{}).get(spacecraft,{})

    # Sample intervals
    samples = []
    for sample_interval in sample_intervals:

        if sample_interval=='none':
            sample_interval='raw'

        samples.append(sample_interval)

    kwargs['resolutions'] = {'spin' : '3s', '5vps': '0.2s'}

    process_overlapping_files(spacecraft, data, process, variables, files_dict, samples, filt_func=filtering, **kwargs)

def get_mms_files_year(directory=None, year=None):
    """
    Obtains a list of all files in the directory
    """
    files_by_year = {}

    if directory is None:
        directory = os.getcwd()

    for year_folder in sorted(os.listdir(directory)):
        if year and year_folder != str(year):
            continue
        elif not os.path.isdir(os.path.join(directory, year_folder)):
            continue

        files = []

        for month_folder in sorted(os.listdir(os.path.join(directory, year_folder))):

            pattern = os.path.join(directory, year_folder, month_folder, '*.cdf')
            for cdf_file in sorted(glob.glob(pattern)):
                files.append(cdf_file)
        files_by_year[int(year_folder)] = files

    if year and str(year) not in files_by_year:
        raise ValueError(f'No files found for {year}.')
    if not files_by_year:
        raise ValueError('No files found.')

    return files_by_year

def get_mms_files_month(directory=None, year=None):
    """
    Obtains a list of all files in the directory, organised by year-month
    """
    files_by_year_month = {}

    if directory is None:
        directory = os.getcwd()

    for year_folder in sorted(os.listdir(directory)):
        if year and year_folder != str(year):
            continue

        year_path = os.path.join(directory, year_folder)
        if not os.path.isdir(year_path):
            continue

        for month_folder in sorted(os.listdir(year_path)):
            month_path = os.path.join(year_path, month_folder)
            if not os.path.isdir(month_path):
                continue

            year_month = f'{year_folder}-{month_folder}'
            pattern = os.path.join(month_path, '*.cdf')

            for cdf_file in sorted(glob.glob(pattern)):
                files_by_year_month.setdefault(year_month, []).append(cdf_file)

    if year and not any(k.startswith(f'{year}-') for k in files_by_year_month):
        raise ValueError(f'No files found for {year}.')
    if not files_by_year_month:
        raise ValueError('No files found.')

    return files_by_year_month

def extract_mms_data(cdf_file, variables):
    """
    Extracts data from the cdf and converts to df
    """

    data_dict = {}

    with pycdf.CDF(cdf_file) as cdf:

        for var_name, var_code in variables.items():
            if var_code not in cdf:
                continue

            data = cdf[var_code].copy()
            if var_name not in ('epoch','epoch_pos'):
                spacepy.pycdf.istp.nanfill(data)
            data = data[...]

            if isinstance(data,float) or isinstance(data,int):
                # Empty dataset
                return {}

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

                data_dict[f'{field}_x_{coords}'] = data[:, 0]
                data_dict[f'{field}_y_{coords}'] = data[:, 1]
                data_dict[f'{field}_z_{coords}'] = data[:, 2]

            elif data.ndim == 2 and data.shape[1] == 3:  # Assuming a 2D array for vector components

                if '_gse' in var_name:
                    coords = 'GSE'
                elif '_gsm' in var_name:
                    coords = 'GSM'
                else:
                    raise Exception(f'Coord system of variable not implemented: {var_name}.')

                if 'hpca' in var_code:
                    field = var_name.split('_')[0]
                    ion   = var_name.split('_')[-1]

                    if coords=='GSE' or f'{field}_x_GSE_{ion}' not in data_dict: # doesn't add X_GSM
                        data_dict[f'{field}_x_{coords}_{ion}'] = data[:, 0]

                    data_dict[f'{field}_y_{coords}_{ion}'] = data[:, 1]
                    data_dict[f'{field}_z_{coords}_{ion}'] = data[:, 2]

                    if f'{field}_mag_{ion}' not in data_dict:
                        data_dict[f'{field}_mag_{ion}'] = np.linalg.norm(data,axis=1)

                elif 'dis' in var_code:
                    field = var_name.split('_')[0]
                    rest_name = '_'.join(var_name.split('_')[2:]) # The [1] component is coordinates
                    if rest_name=='':
                        suffix = ''
                    else:
                        suffix = f'_{rest_name}'

                    if coords=='GSE' or f'{field}_x_GSE{suffix}' not in data_dict: # doesn't add X_GSM
                        data_dict[f'{field}_x_{coords}{suffix}'] = data[:, 0]

                    data_dict[f'{field}_y_{coords}{suffix}'] = data[:, 1]
                    data_dict[f'{field}_z_{coords}{suffix}'] = data[:, 2]

                    if f'{field}_mag{suffix}' not in data_dict:
                        data_dict[f'{field}_mag{suffix}'] = np.linalg.norm(data,axis=1)

            elif data.ndim == 3 and data.shape[1:] == (3, 3):  # Assuming a 3D array for tensor components
                # Compute 1/3 * trace for each time step
                scalar_val = np.trace(data, axis1=1, axis2=2) / 3.0

                data_dict[var_name] = scalar_val

            else:

                data_dict[var_name] = data  # pycdf extracts as datetime, no conversion from epoch needed

            # Temp is scalar or a tensor
            if var_name.startswith('T_'):
                # data is in eV
                data_dict[var_name] *= (e / kB)  # (eV → J) / (J/K) = K


    return data_dict

# %% Process_raw_files

def process_mms_fgm(variables, files, directory_name, log_file_path, time_col='epoch', **kwargs):
    """
    Extracts the state and field data into one dataframe
    """

    ###----------FILES----------###

    field_list = []

    # Loop through each daily file in the year
    for cdf_file in files:
        file_date = os.path.basename(cdf_file).split('_')[4]
        try:  # Bad data check
            field_dict = extract_mms_data(cdf_file, variables)
            if not field_dict:
                log_missing_file(log_file_path, file_date, 'Empty file.')
                continue

            field_list.append(pd.DataFrame(field_dict))

        except Exception as e:
            log_missing_file(log_file_path, file_date, e)

        print(f'{file_date} read.')

    ###---------------COMBINING------------###

    if not field_list:
        print('No data.')
        return pd.DataFrame()

    field_df = pd.concat(field_list, ignore_index=True)
    gsm      = calc_B_GSM_angles(field_df, time_col=time_col)
    field_df = pd.concat([field_df, gsm], axis=1)
    add_df_units(field_df)

    return field_df

def process_mms_state(variables, files, directory_name, log_file_path, time_col='epoch', **kwargs):
    """
    Extracts the state and field data into one dataframe
    """

    ###----------FILES----------###

    state_list = []

    # Loop through each daily file in the year
    for cdf_file in files:
        file_date = os.path.basename(cdf_file).split('_')[4]
        try:  # Bad data check
            state_dict = extract_mms_data(cdf_file, variables)
            if not state_dict:
                log_missing_file(log_file_path, file_date, 'Empty file.')
                continue

            state_list.append(pd.DataFrame(state_dict))

        except Exception as e:
            log_missing_file(log_file_path, file_date, e)

        print(f'{file_date} read.')

    ###---------------COMBINING------------###

    if not state_list:
        print('No data.')
        return pd.DataFrame()

    state_df = pd.concat(state_list, ignore_index=True)
    state_df.rename(columns={f'{time_col}_pos': time_col}, inplace=True)
    add_df_units(state_df)
    del state_list

    return state_df

def process_mms_hpca(variables, files, directory_name, log_file_path, time_col='epoch', **kwargs):
    """
    Extracts the hpca data into a dataframe
    """
    ###----------FILES----------###

    plasma_list = []

    # Loop through each daily file in the year
    for cdf_file in files:
        file_date = os.path.basename(cdf_file).split('_')[5][:8]

        try:  # Bad data check
            plasma_dict = extract_mms_data(cdf_file, variables)
            if not plasma_dict:
                log_missing_file(log_file_path, file_date, 'Empty file.')
                continue

            plasma_list.append(pd.DataFrame(plasma_dict))

        except Exception as e:
            log_missing_file(log_file_path, file_date, e)

    print('Files read.')

    ###---------------COMBINING------------###

    if not plasma_list:
        print('No data.')
        return pd.DataFrame()

    plasma_df = pd.concat(plasma_list, ignore_index=True)
    add_df_units(plasma_df)
    del plasma_list

    if kwargs.get('keep_ions',False):
        return plasma_df

    plasma_df = process_hpca_data(plasma_df, time_col)
    add_df_units(plasma_df)

    return plasma_df

def process_mms_fpi(variables, files, directory_name, log_file_path, time_col='epoch', **kwargs):
    """
    Extracts the fpi data into a dataframe
    Does not process the data or resample it; need to combine with field data first
    """
    ###----------FILES----------###

    plasma_list = []

    # Loop through each daily file in the year
    for cdf_file in files:
        file_date = os.path.basename(cdf_file).split('_')[5][:8]
        file_hour = os.path.basename(cdf_file).split('_')[5][8:10]

        try:  # Bad data check
            plasma_dict = extract_mms_data(cdf_file, variables)
            if not plasma_dict:
                log_missing_file(log_file_path, f'{file_date}-{file_hour}', 'Empty file.')
                continue

            plasma_list.append(pd.DataFrame(plasma_dict))

        except Exception as e:
            log_missing_file(log_file_path, f'{file_date}-{file_hour}', e)

    print('Files read.')

    ###---------------COMBINING------------###

    if not plasma_list:
        print('No data.')
        return pd.DataFrame()

    plasma_df = pd.concat(plasma_list, ignore_index=True)
    add_df_units(plasma_df)
    del plasma_list

    return plasma_df

# %% Plasma

def process_hpca_data(field_df, plasma_df):
    """
    Once the plasma data has been extracted from the raw moments files
    This function will combine into total flow pressure and velocity etc
    """

    ###----------CLEAN UP RAW DATA----------###

    combined_df = pd.concat([field_df, plasma_df], axis=1)


    columns = ['rho_tot','P_flow','P_th','N_tot','T_tot','V_flow','V_x_GSE','V_y_GSE','V_z_GSE','V_y_GSM','V_z_GSM','beta']
    for col in columns:
        combined_df[col] = np.zeros(len(plasma_df))

    # All add linearly
    # P_i  = 0.5 * rho_i * V_i^2    additive across species
    # P_th = n_i * kB * T_i         sums linearly, additive across species
    # N_tot = sum n_i               sums linearly

    # Temperature is a density-weighted average
    # sum_i (n_i*T_i) / sum_i n_i
    numerator_T = np.zeros(len(plasma_df))

    for ion in ION_SPECIES:
        combined_df['rho_tot'] += ION_MASS_DICT.get(ion) * plasma_df[f'N_{ion}'] # for Alfvén speed
        combined_df['P_flow']  += ION_MASS_DICT.get(ion) * plasma_df[f'N_{ion}'] * plasma_df[f'V_mag_{ion}']**2
        combined_df['N_tot']   += plasma_df.loc[~plasma_df[f'N_{ion}'].isna(),f'N_{ion}']
        combined_df['P_th']    += plasma_df[f'P_{ion}']

        numerator_T   += plasma_df[f'N_{ion}'] * plasma_df[f'T_{ion}']

    combined_df['P_flow'] *= 5e20 # N *= 1e6, V *= 1e6, P *= 1e9, so P_flow *= 1e21, x0.5 for factor infront
    combined_df['T_tot'] = numerator_T / combined_df['N_tot']

    # Bulk flow velocity is a mass-density-weighted average
    # V_d = sum_i (rho_i * V_d,i) / sum_i rho_i

    for coords in ('GSE','GSM'):
        for comp in ('x','y','z'):

            if (comp,coords)==('x','GSM') and 'V_x_GSE' in combined_df:
                continue

            numerator_V   = np.zeros(len(plasma_df))
            denominator_V = np.zeros(len(plasma_df))

            for ion in ION_SPECIES:
                col_label = f'V_{comp}_{coords}_{ion}'
                if col_label not in plasma_df:
                    col_label = f'V_{comp}_GSE_{ion}'

                rho_ion = ION_MASS_DICT[ion] * plasma_df[f'N_{ion}']
                numerator_V   += rho_ion * plasma_df[col_label]
                denominator_V += rho_ion

            combined_df[f'V_{comp}_{coords}'] = numerator_V / denominator_V

    return combined_df

def update_fpi_data(field_df, plasma_df):
    """
    Once the plasma data has been extracted from the raw moments files
    This function convert the coordinates and calculate other parameters
    Will work on monthly files, and then resample afterwards
    """

    ###----------CLEAN UP RAW DATA----------###

    # Align field on plasma timestamps (as plasma contains mode, quality etc. numbers)
    field_df = field_df.reindex(plasma_df.index, method=None).interpolate(method='time')
    filter_quality(field_df, column='B_flag')

    plasma_df['N_tot']      -= plasma_df['N_tot_bg'] # removes background counts
    plasma_df['P_th_tens']  -= plasma_df['P_th_bg']

    plasma_df.rename(columns={'P_th_tens': 'P_th', 'T_tens': 'T_tot', 'V_mag': 'V_flow', 'V_mag_unc': 'V_flow_unc'}, inplace=True)
    plasma_df.drop(columns=['N_tot_bg', 'P_th_bg'], inplace=True)
    filter_quality(plasma_df, column='flag')

    merged_df = pd.concat([field_df, plasma_df], axis=1)

    return merged_df


def filter_quality(df, column='flag'):

    good_quality = 0 # good data

    if column not in df:
        print(f'No "{column}" column.')
        return df

    # solar wind flag
    mask = (df[column].fillna(-2) == good_quality)

    filtered_df = df.loc[mask]
    filtered_df.drop(columns=[column],inplace=True)
    filtered_df.attrs = df.attrs

    return filtered_df