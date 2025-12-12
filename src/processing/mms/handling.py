# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:42:01 2025

@author: richarj2
"""
import os
import glob

import numpy as np
import pandas as pd
from scipy.constants import k as kB, e, mu_0

from spacepy import pycdf
import spacepy.pycdf.istp

from .config import ION_MASS_DICT, ION_SPECIES, VARIABLES_DICT

from ..dataframes import add_df_units, resample_data
from ..handling import create_log_file, log_missing_file
from ..writing import write_to_cdf
from ..reading import import_processed_data
from ..utils import create_directory

from ...coordinates.magnetic import calc_B_GSM_angles
from ...config import R_E, get_luna_directory, get_proc_directory

def process_mms_files(spacecraft, data, sample_intervals='raw', time_col='epoch', year=None, overwrite=True):

    print('Processing MMS.')


    directory      = get_luna_directory(spacecraft, data)
    directory_name = os.path.basename(os.path.normpath(directory))

    save_directory = get_proc_directory(spacecraft, dtype=data, create=True)
    log_file_path = os.path.join(save_directory, f'{directory_name}_files_not_added.txt')  # Stores not loaded files
    create_log_file(log_file_path)

    if isinstance(sample_intervals,str):
        sample_intervals = (sample_intervals,)

    save_directories = {}

    for sample_interval in sample_intervals:

        if sample_interval=='none':
            sample_interval='raw'

        save_directory = get_proc_directory(spacecraft, dtype=data, resolution=sample_interval, create=True)
        create_directory(save_directory)
        save_directories[sample_interval] = save_directory

    variables = VARIABLES_DICT.get(data,{}).get(spacecraft,{})
    if not variables:
        raise ValueError(f'No valid variables dict for {data}')
    files_by_year = get_mms_files(directory, year)

    if 'FGM' in directory_name:
        process_mms_field(variables, files_by_year, directory_name, log_file_path, save_directories, time_col=time_col, overwrite=overwrite)
    elif 'HPCA' in directory_name:
        process_mms_hpca(variables, files_by_year, directory_name, log_file_path, save_directories, time_col=time_col, overwrite=overwrite)
    elif 'FPI' in directory_name:
        process_mms_fpi(variables, files_by_year, directory_name, log_file_path, save_directories, time_col=time_col, overwrite=overwrite)

def get_mms_files(directory=None, year=None):
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
    """
    Extracts data from the cdf and converts to df
    """

    state_dict = {}
    field_dict = {}

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
                return {}, {}

            if var_name in ('epoch_pos','r_gse'):
                data_dict = state_dict
            else:
                data_dict = field_dict

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

                    if coords=='GSE' or f'{field}_x_GSE_{rest_name}' not in data_dict: # doesn't add X_GSM
                        data_dict[f'{field}_x_{coords}_{rest_name}'] = data[:, 0]

                    data_dict[f'{field}_y_{coords}_{rest_name}'] = data[:, 1]
                    data_dict[f'{field}_z_{coords}_{rest_name}'] = data[:, 2]

                    if f'{field}_mag_{rest_name}' not in data_dict:
                        data_dict[f'{field}_mag_{rest_name}'] = np.linalg.norm(data,axis=1)

            elif data.ndim == 3 and data.shape[1:] == (3, 3):  # Assuming a 3D array for tensor components
                # Compute 1/3 * trace for each time step
                scalar_val = np.trace(data, axis1=1, axis2=2) / 3.0

                data_dict[var_name] = scalar_val

            else:
                if var_name.startswith('T_'):
                    # data is in eV
                    data *= (e / kB)  # (eV → J) / (J/K) = K

                data_dict[var_name] = data  # pycdf extracts as datetime, no conversion from epoch needed


    return state_dict, field_dict

def process_mms_field(variables, files_by_year, directory_name, log_file_path, save_directories, time_col='epoch', overwrite=True):
    """
    Extracts the state and field data into one dataframe
    Resamples into different resolutions
    """

    ###----------FILES----------###

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

        for sample_interval, samp_dir in save_directories.items():

            merged_chunks = []

            for month_num in range(1, 13):
                print(f'Month {month_num}')

                state_mask = yearly_state_df[time_col+'_pos'].dt.month == month_num
                field_mask = yearly_field_df[time_col].dt.month == month_num
                if np.sum(state_mask)==0 or np.sum(field_mask)==0:
                    continue

                if sample_interval=='raw':
                    state_month = yearly_state_df.loc[state_mask]
                    state_month.rename(columns={time_col+'_pos': time_col}, inplace=True)
                    field_month = yearly_field_df.loc[field_mask]

                else:
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

def process_mms_hpca(variables, files_by_year, directory_name, log_file_path, save_directories, time_col='epoch', overwrite=True):
    """
    Extracts the hpca data into a dataframe
    Resamples into different resolutions
    """
    ###----------FILES----------###

    for the_year, files in files_by_year.items():
        print(f'Processing {the_year} data.')
        plasma_list = []

        # Loop through each daily file in the year
        for cdf_file in files:
            file_date = os.path.basename(cdf_file).split('_')[5][:8]

            try:  # Bad data check
                _, plasma_dict = extract_mms_data(cdf_file, variables)
                if not plasma_dict:
                    log_missing_file(log_file_path, file_date, 'Empty file.')
                    continue

                plasma_list.append(pd.DataFrame(plasma_dict))

            except Exception as e:
                log_missing_file(log_file_path, file_date, e)

        print('Files read.')

        ###---------------COMBINING------------###

        if not plasma_list:
            print(f'No data for {the_year}')
            continue

        yearly_plasma_df = pd.concat(plasma_list, ignore_index=True)
        add_df_units(yearly_plasma_df)
        del plasma_list

        processed_plasma_df = process_hpca_data(yearly_plasma_df, time_col)
        add_df_units(processed_plasma_df)

        print('Data processed.')

        # resample and write to file
        for sample_interval, samp_dir in save_directories.items():

            if sample_interval =='raw':
                sampled_df = yearly_plasma_df
            elif sample_interval =='spin':
                sampled_df = processed_plasma_df
            else:
                # resample and write to file
                print('Resampling...')
                sampled_df = resample_data(processed_plasma_df, time_col, sample_interval)
                print(f'{sample_interval} resampled.')

            output_file = os.path.join(samp_dir, f'{directory_name}_{the_year}.cdf')
            attributes = {'sample_interval': sample_interval, 'time_col': time_col, 'R_E': R_E}
            write_to_cdf(sampled_df, output_file, attributes, overwrite)

        print(f'{the_year} processed.')

def process_mms_fpi(variables, files_by_year, directory_name, log_file_path, save_directories, time_col='epoch', overwrite=True):
    """
    Extracts the fpi data into a dataframe
    Does not process the data or resample it; need to combine with field data first
    """
    ###----------FILES----------###

    for the_year, files in files_by_year.items():
        print(f'Processing {the_year} data.')
        plasma_list = []

        # Loop through each daily file in the year
        for cdf_file in files:
            file_date = os.path.basename(cdf_file).split('_')[5][:8]
            file_hour = os.path.basename(cdf_file).split('_')[5][8:10]

            try:  # Bad data check
                _, plasma_dict = extract_mms_data(cdf_file, variables)
                if not plasma_dict:
                    log_missing_file(log_file_path, f'{file_date}-{file_hour}', 'Empty file.')
                    continue

                plasma_list.append(pd.DataFrame(plasma_dict))

            except Exception as e:
                log_missing_file(log_file_path, f'{file_date}-{file_hour}', e)

        print('Files read.')

        ###---------------COMBINING------------###

        if not plasma_list:
            print(f'No data for {the_year}')
            continue

        yearly_plasma_df = pd.concat(plasma_list, ignore_index=True)
        add_df_units(yearly_plasma_df)
        del plasma_list

        # resample and write to file
        for sample_interval, samp_dir in save_directories.items():

            if sample_interval == 'raw':
                sampled_df = yearly_plasma_df
                sample_interval = 'spin'
            else:
                # resample and write to file
                print('Resampling...')
                sampled_df = resample_data(yearly_plasma_df, time_col, sample_interval)
                print(f'{sample_interval} resampled.')

            attributes = {'sample_interval': sample_interval, 'time_col': time_col, 'R_E': R_E}
            write_to_cdf(sampled_df, directory=samp_dir, file_name=f'{directory_name}_{the_year}.cdf', attributes=attributes, overwrite=overwrite)

        print(f'{the_year} processed.')

# %%
def process_hpca_data(plasma_df, time_col='epoch', ion_species=ION_SPECIES):
    """
    Once the plasma data has been extracted from the raw moments files
    This function will combine into total flow pressure and velocity etc
    As well as re-sample to 1-min and 5-min
    """

    combined_df           = pd.DataFrame(index=plasma_df.index)
    combined_df[time_col] = plasma_df[time_col]

    columns = ['rho_tot','P_flow','P_th','N_tot','T_tot','V_flow','V_x_GSE','V_y_GSE','V_z_GSE','V_y_GSM','V_z_GSM','beta']
    for col in columns:
        combined_df[col] = np.zeros(len(plasma_df))

    ###----------MOMENTS----------###

    # All add linearly
    # P_i  = 0.5 * rho_i * V_i^2    additive across species
    # P_th = n_i * kB * T_i         sums linearly, additive across species
    # N_tot = sum n_i               sums linearly

    # Temperature is a density-weighted average
    # sum_i (n_i*T_i) / sum_i n_i
    numerator_T = np.zeros(len(plasma_df))

    for ion in ion_species:
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

            for ion in ion_species:
                col_label = f'V_{comp}_{coords}_{ion}'
                if col_label not in plasma_df:
                    col_label = f'V_{comp}_GSE_{ion}'

                rho_ion = ION_MASS_DICT[ion] * plasma_df[f'N_{ion}']
                numerator_V   += rho_ion * plasma_df[col_label]
                denominator_V += rho_ion

            combined_df[f'V_{comp}_{coords}'] = numerator_V / denominator_V

    combined_df['V_flow'] = np.linalg.norm(combined_df[[f'V_{comp}_GSE' for comp in ('x','y','z')]].to_numpy(),axis=1)

    # Plasma beta
    # Beta = p_th / p_mag, p_mag = B^2/2mu_0

    combined_df['beta'] = combined_df['P_th'] / (plasma_df['B_avg']**2) * (2*mu_0) * 1e9
    # p_dyn *= 1e-9, B_avg *= 1e18, so beta *= 1e9


    # Alfvén Speed
    # vA = B / sqrt(mu_0 * rho)

    combined_df['V_A'] = plasma_df['B_avg'] / np.sqrt(mu_0 * combined_df['rho_tot']) * 1e-15
    # B_avg *= 1e-9, 1/sqrt(rho) *= 1e-3, vA *= 1e-3, so speed *= 1e-15

    combined_df.drop(columns=['rho_tot'], inplace=True)

    ###----------CROSS PRODUCTS----------###

    vec_coords = 'GSM'
    B_cols = ['B_x_GSE',f'B_y_{vec_coords}',f'B_z_{vec_coords}']
    V_cols = ['V_x_GSE',f'V_y_{vec_coords}',f'V_z_{vec_coords}']

    # E = -V x B = B x V
    combined_df[[f'E_{comp}_{vec_coords}' for comp in ('x','y','z')]] = np.cross(plasma_df[B_cols], combined_df[V_cols]) * 1e-3
    # V *= 1e3, B *= 1e-9, and E *= 1e3 so e_gse *= 1e-3
    combined_df['E_mag'] =  np.linalg.norm(combined_df[[f'E_{comp}_{vec_coords}' for comp in ('x','y','z')]],axis=1)

    # S = E x H = E x B / mu_0
    combined_df[[f'S_{comp}_{vec_coords}' for comp in ('x','y','z')]] = np.cross(combined_df[[f'E_{comp}_{vec_coords}' for comp in ('x','y','z')]], plasma_df[B_cols]) * 1e-6 / mu_0
    # E combined_df 1e-3, B *= 1e-9, and S *= 1e6 so s_gse *= 1e-6
    combined_df['S_mag'] =  np.linalg.norm(combined_df[[f'S_{comp}_{vec_coords}' for comp in ('x','y','z')]],axis=1)

    ###----------KAN AND LEE----------###

    # Clock Angle: theta = atan2(By,Bz)
    B_clock = np.arctan2(plasma_df[f'B_y_{vec_coords}'], plasma_df[f'B_z_{vec_coords}'])

    # Kan and Lee Electric Field: E_R = |V| * B_T * sin^2 (clock/2)
    combined_df['E_R'] = (combined_df['V_flow'] * np.sqrt(plasma_df[f'B_y_{vec_coords}']**2+plasma_df[f'B_z_{vec_coords}']**2) * (np.sin(B_clock/2))**2) * 1e-3
    # V *= 1e3, B *= 1e-9, E *= 1e3, so E_R *= 1e-3

    return combined_df

def combine_mms_fpi_field():
    """
    Wrapper function to combine fpi and field data, process, and then resample
    """

    ###----------FILES----------###

    for the_year, files in files_by_year.items():
        print(f'Processing {the_year} data.')

        plasma_df = import_processed_data(spacecraft, dtype='fpi')

        processed_plasma_df = process_fpi_data(yearly_plasma_df, time_col)
        add_df_units(processed_plasma_df)

        print('Data processed.')

        # resample and write to file
        for sample_interval, samp_dir in save_directories.items():

            if sample_interval =='raw':
                sampled_df = yearly_plasma_df
            elif sample_interval =='spin':
                sampled_df = processed_plasma_df
            else:
                # resample and write to file
                print('Resampling...')
                sampled_df = resample_data(processed_plasma_df, time_col, sample_interval)
                print(f'{sample_interval} resampled.')

            output_file = os.path.join(samp_dir, f'{directory_name}_{the_year}.cdf')
            attributes = {'sample_interval': sample_interval, 'time_col': time_col, 'R_E': R_E}
            write_to_cdf(sampled_df, output_file, attributes, overwrite)

        print(f'{the_year} processed.')

    return files_by_year

def process_fpi_data(plasma_df, time_col='epoch', ion_species=ION_SPECIES):
    """
    Once the plasma data has been extracted from the raw moments files
    This function will combine into total flow pressure and velocity etc
    As well as re-sample to 1-min and 5-min
    """

    combined_df           = pd.DataFrame(index=plasma_df.index)
    combined_df[time_col] = plasma_df[time_col]

    columns = ['rho_tot','P_flow','P_th','N_tot','T_tot','V_flow','V_x_GSE','V_y_GSE','V_z_GSE','V_y_GSM','V_z_GSM','beta']
    for col in columns:
        combined_df[col] = np.zeros(len(plasma_df))

    ###----------MOMENTS----------###

    # All add linearly
    # P_i  = 0.5 * rho_i * V_i^2    additive across species
    # P_th = n_i * kB * T_i         sums linearly, additive across species
    # N_tot = sum n_i               sums linearly

    # Temperature is a density-weighted average
    # sum_i (n_i*T_i) / sum_i n_i
    numerator_T = np.zeros(len(plasma_df))

    for ion in ion_species:
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

            for ion in ion_species:
                col_label = f'V_{comp}_{coords}_{ion}'
                if col_label not in plasma_df:
                    col_label = f'V_{comp}_GSE_{ion}'

                rho_ion = ION_MASS_DICT[ion] * plasma_df[f'N_{ion}']
                numerator_V   += rho_ion * plasma_df[col_label]
                denominator_V += rho_ion

            combined_df[f'V_{comp}_{coords}'] = numerator_V / denominator_V

    combined_df['V_flow'] = np.linalg.norm(combined_df[[f'V_{comp}_GSE' for comp in ('x','y','z')]].to_numpy(),axis=1)

    # Plasma beta
    # Beta = p_th / p_mag, p_mag = B^2/2mu_0

    combined_df['beta'] = combined_df['P_th'] / (plasma_df['B_avg']**2) * (2*mu_0) * 1e9
    # p_dyn *= 1e-9, B_avg *= 1e18, so beta *= 1e9


    # Alfvén Speed
    # vA = B / sqrt(mu_0 * rho)

    combined_df['V_A'] = plasma_df['B_avg'] / np.sqrt(mu_0 * combined_df['rho_tot']) * 1e-15
    # B_avg *= 1e-9, 1/sqrt(rho) *= 1e-3, vA *= 1e-3, so speed *= 1e-15

    combined_df.drop(columns=['rho_tot'], inplace=True)

    ###----------CROSS PRODUCTS----------###

    vec_coords = 'GSM'
    B_cols = ['B_x_GSE',f'B_y_{vec_coords}',f'B_z_{vec_coords}']
    V_cols = ['V_x_GSE',f'V_y_{vec_coords}',f'V_z_{vec_coords}']

    # E = -V x B = B x V
    combined_df[[f'E_{comp}_{vec_coords}' for comp in ('x','y','z')]] = np.cross(plasma_df[B_cols], combined_df[V_cols]) * 1e-3
    # V *= 1e3, B *= 1e-9, and E *= 1e3 so e_gse *= 1e-3
    combined_df['E_mag'] =  np.linalg.norm(combined_df[[f'E_{comp}_{vec_coords}' for comp in ('x','y','z')]],axis=1)

    # S = E x H = E x B / mu_0
    combined_df[[f'S_{comp}_{vec_coords}' for comp in ('x','y','z')]] = np.cross(combined_df[[f'E_{comp}_{vec_coords}' for comp in ('x','y','z')]], plasma_df[B_cols]) * 1e-6 / mu_0
    # E combined_df 1e-3, B *= 1e-9, and S *= 1e6 so s_gse *= 1e-6
    combined_df['S_mag'] =  np.linalg.norm(combined_df[[f'S_{comp}_{vec_coords}' for comp in ('x','y','z')]],axis=1)

    ###----------KAN AND LEE----------###

    # Clock Angle: theta = atan2(By,Bz)
    B_clock = np.arctan2(plasma_df[f'B_y_{vec_coords}'], plasma_df[f'B_z_{vec_coords}'])

    # Kan and Lee Electric Field: E_R = |V| * B_T * sin^2 (clock/2)
    combined_df['E_R'] = (combined_df['V_flow'] * np.sqrt(plasma_df[f'B_y_{vec_coords}']**2+plasma_df[f'B_z_{vec_coords}']**2) * (np.sin(B_clock/2))**2) * 1e-3
    # V *= 1e3, B *= 1e-9, E *= 1e3, so E_R *= 1e-3

    return combined_df


    #### CLUSTER METHOD #####


    field_df = import_processed_data(field_dir, year=year)


    print(f'Processing {year} data.')

    ###----------GSE to GSM----------###

    # Align field on plasma timestamps (as plasma contains mode, quality etc. numbers)
    field_df = field_df.reindex(plasma_df.index, method=None).interpolate(method='time')

    merged_df = pd.concat([field_df, plasma_df], axis=1)
    vec_coords = 'GSE'

    if fvps_directory is not None:

        convert_cols = ('V',)

        fvps_df = import_processed_data(fvps_directory, year=year)
        gsm_vectors = GSE_to_GSM_with_angles(merged_df, [[f'{vec}_{comp}_GSE' for comp in ('x','y','z')] for vec in convert_cols], df_coords=fvps_df, ref='B', interp=True)

        merged_df = pd.concat([merged_df,gsm_vectors], axis=1)
        vec_coords = 'GSM'

        # Freeing up memory
        del fvps_df

        print('GSE to GSM.')

    # For now treating all species together
    merged_df.rename(columns={'P_ion': 'P_th', 'T_ion': 'T_tot', 'N_ion': 'N_tot'}, inplace=True)


    print('Calculations...')
    ###----------CONTAMINATION----------###

    merged_df.loc[merged_df['B_avg']>200,'B_avg'] = np.nan
    for comp in ('x','y','z'):
        merged_df.loc[merged_df[f'B_{comp}_GSM'].abs()>150,f'B_{comp}_GSM'] = np.nan

    merged_df.loc[merged_df['V_mag']>2e3,'V_mag'] = np.nan
    for comp, limit in zip(('x','y','z'),(2000,400,400)):
        merged_df.loc[np.abs(merged_df[f'V_{comp}_GSM'])>limit,f'V_{comp}_GSM'] = np.nan

    merged_df.loc[merged_df['N_tot']>500,'N_tot'] = np.nan
    merged_df.loc[merged_df['P_th']>100,'P_th'] = np.nan

    ###----------CALCULATIONS----------###

    # Clock Angle: theta = atan2(By,Bz)
    merged_df['B_clock'] = np.arctan2(merged_df[f'B_y_{vec_coords}'], merged_df[f'B_z_{vec_coords}'])

    # Kan and Lee Electric Field: E_R = |V| * B_T * sin^2 (clock/2)
    merged_df['E_R'] = (merged_df['V_mag'] * np.sqrt(merged_df[f'B_y_{vec_coords}']**2+merged_df[f'B_z_{vec_coords}']**2) * (np.sin(merged_df['B_clock']/2))**2) * 1e-3
    # V *= 1e3, B *= 1e-9, E *= 1e3, so E_R *= 1e-3

    ###----------PRESSURES----------###

    try: # find ratio of alpha and proton densities
        omni_df = import_processed_data(omni_dir,year=year)
        merged_df = pd.merge_asof(
            merged_df.sort_index(),
            omni_df[['na_np_ratio']].sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'   # take the closest value on or before the timestamp
        )

        del omni_df
    except:
        merged_df['na_np_ratio'] = 0.05

    # Dynamic pressure
    # P = 0.5 * rho * V^2

    m_avg = (m_p + merged_df['na_np_ratio'] * m_a) / (merged_df['na_np_ratio'] + 1) # kg
    merged_df['P_flow'] = 0.5 * m_avg * merged_df['N_tot']  * merged_df['V_mag']**2 * 1e21
    # N *= 1e6, V *= 1e6, P *= 1e9, so P_flow *= 1e21

    # Beta = p_th / p_mag, p_mag = B^2/2mu_0

    merged_df['beta'] = merged_df['P_th'] / (merged_df['B_avg']**2) * (2*mu_0) * 1e9
    # p_dyn *= 1e-9, 1/B_avg^2 *= 1e18, so beta *= 1e9

    # Alfven Speed
    # vA = B / sqrt(mu_0 * rho)

    merged_df['V_A'] = merged_df['B_avg'] / np.sqrt(mu_0 * m_avg * merged_df['N_tot']) * 1e-15
    # B_avg *= 1e-9, 1/sqrt(rho) *= 1e-3, vA *= 1e-3, so speed *= 1e-15

    ###----------CROSS PRODUCTS----------###

    # E = -V x B = B x V
    merged_df[[f'E_{comp}_{vec_coords}' for comp in ('x','y','z')]] = np.cross(merged_df[[f'B_{comp}_{vec_coords}' for comp in ('x','y','z')]],merged_df[[f'V_{comp}_{vec_coords}' for comp in ('x','y','z')]]) * 1e-3
    # V *= 1e3, B *= 1e-9, and E *= 1e3 so e_gse *= 1e-3
    merged_df['E_mag'] =  np.linalg.norm(merged_df[[f'E_{comp}_{vec_coords}' for comp in ('x','y','z')]],axis=1)

    # S = E x H = E x B / mu_0
    merged_df[[f'S_{comp}_{vec_coords}' for comp in ('x','y','z')]] = np.cross(merged_df[[f'E_{comp}_{vec_coords}' for comp in ('x','y','z')]],merged_df[[f'B_{comp}_{vec_coords}' for comp in ('x','y','z')]]) * 1e-6 / mu_0
    # E *= 1e-3, B *= 1e-9, and S *= 1e6 so s_gse *= 1e-6
    merged_df['S_mag'] =  np.linalg.norm(merged_df[[f'S_{comp}_{vec_coords}' for comp in ('x','y','z')]],axis=1)

    ###----------WRITE TO FILE----------###
    if not os.path.exists(combined_dir):
        os.makedirs(combined_dir)

    output_file = os.path.join(combined_dir, f'C1_SPIN_{year}.cdf')

    print(f'{year} processed.')
    write_to_cdf(merged_df, output_file, {'R_E': R_E}, overwrite=True, reset_index=True)



