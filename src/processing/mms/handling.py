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

from .config import ION_MASS_DICT, ION_SPECIES

from ..dataframes import add_df_units, resample_data
from ..handling import create_log_file, log_missing_file
from ..writing import write_to_cdf
from ..utils import create_directory

from ...coordinates.magnetic import calc_B_GSM_angles
from ...config import R_E

def process_mms_files(directory, data_directory, variables, sample_intervals='raw', time_col='epoch', year=None, overwrite=True):

    print('Processing MMS.')

    ###----------DIRECTORIES----------###

    directory_name = os.path.basename(os.path.normpath(directory))
    if 'FGM' in directory_name:
        save_directory = os.path.join(data_directory, 'field')
    else:
        save_directory = os.path.join(data_directory, 'plasma')

    log_file_path = os.path.join(save_directory, f'{directory_name}_files_not_added.txt')  # Stores not loaded files
    create_log_file(log_file_path)

    if isinstance(sample_intervals,str):
        sample_intervals = (sample_intervals,)

    save_directories = {}

    for sample_interval in sample_intervals:

        if sample_interval=='none':
            sample_interval=='raw'

        samp_dir = os.path.join(save_directory, sample_interval)
        create_directory(samp_dir)
        save_directories[sample_interval] = samp_dir

    ###----------FILES----------###

    files_by_year = get_mms_files(directory, year)

    if 'FGM' in directory_name:
        process_mms_field(variables, files_by_year, directory_name, log_file_path, save_directories, time_col=time_col, overwrite=overwrite)
    else:
        process_mms_plasma(variables, files_by_year, directory_name, log_file_path, save_directories, time_col=time_col, overwrite=overwrite)


def process_mms_field(variables, files_by_year, directory_name, log_file_path, save_directories, time_col='epoch', overwrite=True):


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


def process_mms_plasma(variables, files_by_year, directory_name, log_file_path, save_directories, time_col='epoch', overwrite=True):

    ###----------FILES----------###

    for the_year, files in files_by_year.items():
        print(f'Processing {the_year} data.')
        plasma_list = []

        # Loop through each daily file in the year
        for cdf_file in files:
            file_date = os.path.basename(cdf_file).split('_')[5][:8]
            _, plasma_dict = extract_mms_data(cdf_file, variables)
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

        processed_plasma_df = process_plasma_data(yearly_plasma_df, time_col)
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
                field = var_name.split('_')[0]
                ion   = var_name.split('_')[-1]

                if coords=='GSE' or f'{field}_x_GSE_{ion}' not in data_dict: # doesn't add X_GSM
                    data_dict[f'{field}_x_{coords}_{ion}'] = data[:, 0]

                data_dict[f'{field}_y_{coords}_{ion}'] = data[:, 1]
                data_dict[f'{field}_z_{coords}_{ion}'] = data[:, 2]

                if f'{field}_mag_{ion}' not in data_dict:
                    data_dict[f'{field}_mag_{ion}'] = np.sqrt(data[:,0]**2+data[:,1]**2+data[:,2]**2)

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

# %%
def process_plasma_data(plasma_df, time_col='epoch', ion_species=ION_SPECIES):
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
        combined_df['rho_tot'] += ION_MASS_DICT.get(ion) * plasma_df[f'N_{ion}'] # for Alfven speed
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


    # Alfven Speed
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

