# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:42:01 2025

@author: richarj2
"""
import os
import glob

import numpy as np
import pandas as pd
from uncertainties import unumpy as unp
from scipy.constants import k as kB, e, mu_0, m_p
from scipy.constants import physical_constants
m_a = physical_constants['alpha particle mass'][0]

from spacepy import pycdf
import spacepy.pycdf.istp

from .config import ION_MASS_DICT, ION_SPECIES, VARIABLES_DICT

from ..dataframes import add_df_units, resample_data
from ..handling import create_log_file, log_missing_file
from ..writing import write_to_cdf
from ..reading import import_processed_data
from ..utils import create_directory

from ...coordinates.magnetic import calc_B_GSM_angles, convert_GSE_to_GSM_with_angles
from ...config import R_E, get_luna_directory, get_proc_directory

def process_mms_files(spacecraft, data, sample_intervals=('raw',), time_col='epoch', year=None, overwrite=True):

    ###----------SET UP----------###

    print('Processing MMS.')

    directory      = get_luna_directory(spacecraft, data)
    directory_name = os.path.basename(os.path.normpath(directory))
    if data=='state':
        directory_name = 'STATE'

    save_directory = get_proc_directory(spacecraft, dtype=data, create=True)
    log_file_path = os.path.join(save_directory, f'{directory_name}_files_not_added.txt')  # Stores not loaded files
    create_log_file(log_file_path)

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

    ###----------PROCESS----------###

    kwargs = {}

    if data=='fgm':
        files_dict = get_mms_files_month(directory, year)
        process = process_mms_fgm
    elif data=='state':
        files_dict = get_mms_files_year(directory, year)
        process = process_mms_state
    elif data=='hpca':
        files_dict = get_mms_files_year(directory, year)
        process = process_mms_hpca
        if 'raw' in sample_intervals:
            kwargs['keep_ions'] = True
            sample_intervals = ('raw',) # prevents resampling
    elif data=='fpi':
        files_dict = get_mms_files_month(directory, year)
        process = process_mms_fpi
    else:
        raise ValueError(f'"{data}" not valid data to sample')

    next_key_df = pd.DataFrame()
    for k_i, (key, files) in enumerate(files_dict.items()):

        print(f'Processing {key} data.')
        key_df = process(variables, files, directory_name, log_file_path, time_col=time_col, **kwargs)
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
            else:
                # resample and write to file
                print('Resampling...')
                sampled_df = resample_data(key_df, time_col, sample_interval)
                print(f'{sample_interval} resampled.')

            attributes = {'sample_interval': sample_interval, 'time_col': time_col, 'R_E': R_E}
            write_to_cdf(sampled_df, directory=samp_dir, file_name=f'{directory_name}_{key}', attributes=attributes, overwrite=overwrite)

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
                if var_name.startswith('T_'):
                    # data is in eV
                    data *= (e / kB)  # (eV → J) / (J/K) = K

                data_dict[var_name] = data  # pycdf extracts as datetime, no conversion from epoch needed


    return data_dict

def resample_monthly_mms(spacecraft, data, raw_res='spin', sample_intervals=('1min',), time_col='epoch', overwrite=True):
    """
    Resample monthly files into yearly files.
    """
    ###----------SET UP----------###
    print('Resampling.')

    save_directories = {}

    for sample_interval in sample_intervals:

        save_directory = get_proc_directory(spacecraft, dtype=data, resolution=sample_interval, create=True)
        create_directory(save_directory)
        save_directories[sample_interval] = save_directory

    raw_dir = get_proc_directory(spacecraft, dtype=data, resolution=raw_res)
    pattern = os.path.join(raw_dir, '*.cdf')
    files_by_year = {}

    for cdf_file in sorted(glob.glob(pattern)):
        file_name = os.path.basename(cdf_file)
        dir_name = '_'.join(file_name.split('_')[:-1])
        year = file_name.split('_')[-1][:4]
        files_by_year.setdefault(year, []).append(file_name)

    ###----------PROCESS----------###
    for year, files in files_by_year.items():
        yearly_list = []

        for file in files:
            df = import_processed_data(spacecraft, dtype=data, resolution=raw_res, file_name=file)
            yearly_list.append(df)

        yearly_df = pd.concat(yearly_list) # don't set ignore_index to True
        yearly_df.drop(columns=[c for c in yearly_df.columns if c.endswith('_unc')],inplace=True) # measurement error << statistical uncertainty
        add_df_units(yearly_df)

        for sample_interval, samp_dir in save_directories.items():

            sampled_df = resample_data(yearly_df, time_col='index', sample_interval=sample_interval)

            attributes = {'sample_interval': sample_interval, 'time_col': time_col, 'R_E': R_E}
            write_to_cdf(sampled_df, directory=samp_dir, file_name=f'{dir_name}_{year}', attributes=attributes, overwrite=overwrite, time_col=time_col, reset_index=True)

            print(f'{sample_interval} reprocessed.')

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
    field_df = field_df.loc[field_df['B_flag']==0] # Quality 0 = no problems
    field_df.drop(columns=['B_flag'],inplace=True)

    gsm = calc_B_GSM_angles(field_df, time_col=time_col)
    field_df = pd.concat([field_df, gsm], axis=1)
    add_df_units(field_df)
    del field_list

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
    plasma_df = plasma_df.loc[plasma_df['flag']==0] # Quality 0 = no problems
    plasma_df.drop(columns=['flag'],inplace=True)

    add_df_units(plasma_df)
    del plasma_list

    return plasma_df


# %% Process_extracted_data

def insert_magnitude(df, field, vec_coords, suffix='mag'):
    # insert before x column
    x_column = df.columns.get_loc(f'{field}_x_{vec_coords}')
    df.insert(x_column, f'{field}_{suffix}', (df[[f'{field}_{c}_{vec_coords}' for c in ('x', 'y', 'z')]].pow(2).sum(axis=1))**0.5)

def process_hpca_data(plasma_df, time_col='epoch', ion_species=ION_SPECIES):
    """
    Once the plasma data has been extracted from the raw moments files
    This function will combine into total flow pressure and velocity etc
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

    insert_magnitude(combined_df, 'V', 'GSE', suffix='flow')

    # Plasma beta = p_th / p_mag, p_mag = B^2/2mu_0
    # p_dyn *= 1e-9, B_avg *= 1e18, so beta *= 1e9
    combined_df['beta'] = combined_df['P_th'] / (plasma_df['B_avg']**2) * (2*mu_0) * 1e9

    # Alfvén Speed vA = B / sqrt(mu_0 * rho)
    # B_avg *= 1e-9, 1/sqrt(rho) *= 1e-3, vA *= 1e-3, so speed *= 1e-15
    combined_df['V_A'] = plasma_df['B_avg'] / np.sqrt(mu_0 * combined_df['rho_tot']) * 1e-15

    combined_df.drop(columns=['rho_tot'], inplace=True)

    ###----------KAN AND LEE----------###

    vec_coords = 'GSM'
    B_cols = ['B_x_GSE',f'B_y_{vec_coords}',f'B_z_{vec_coords}']
    V_cols = ['V_x_GSE',f'V_y_{vec_coords}',f'V_z_{vec_coords}']
    E_cols = [f'E_x_{vec_coords}',f'E_y_{vec_coords}',f'E_z_{vec_coords}']

    # Clock Angle: theta = atan2(By,Bz)
    B_clock = np.arctan2(plasma_df[f'B_y_{vec_coords}'], plasma_df[f'B_z_{vec_coords}'])

    # Kan and Lee Electric Field: E_R = |V| * B_T * sin^2 (clock/2)
    # V *= 1e3, B *= 1e-9, E *= 1e3, so E_R *= 1e-3
    combined_df['E_R'] = (combined_df['V_flow'] * (plasma_df[[f'B_y_{vec_coords}', f'B_z_{vec_coords}']].pow(2).sum(axis=1)** 0.5) * (np.sin(B_clock/2))**2) * 1e-3

    ###----------CROSS PRODUCTS----------###

    # E = -V x B = B x V
    # V *= 1e3, B *= 1e-9, and E *= 1e3 so e_gse *= 1e-3
    combined_df[[f'E_{comp}_{vec_coords}' for comp in ('x','y','z')]] = np.cross(plasma_df[B_cols], combined_df[V_cols]) * 1e-3
    insert_magnitude(combined_df, 'E', vec_coords)

    # S = E x H = E x B / mu_0
    # E *= 1e-3, B *= 1e-9, and S *= 1e6 so s_gse *= 1e-6
    combined_df[[f'S_{comp}_{vec_coords}' for comp in ('x','y','z')]] = np.cross(combined_df[E_cols], plasma_df[B_cols]) * 1e-6 / mu_0
    insert_magnitude(combined_df, 'S', vec_coords)


    return combined_df

def update_fpi_data(spacecraft, raw_res='raw', save_res='spin', ion_source='hpca', ion_species=ION_SPECIES):
    """
    Takes in the monthly fpi files and uses monthly fgm files (and OMNI)
    Then will save files in a 'spin' directory to be re-sampled after
    """
    print('Updating MMS FPI.')

    def get_files(data):
        files = {}
        data_dir = get_proc_directory(spacecraft, dtype=data, resolution=raw_res)
        pattern = os.path.join(data_dir, '*.cdf')
        for cdf_file in sorted(glob.glob(pattern)):
            file_name = os.path.basename(cdf_file)
            key = file_name.split('_')[-1][:7]
            files[key] = file_name
        return files

    fgm_files = get_files('fgm')
    fpi_files = get_files('fpi')

    save_directory = get_proc_directory(spacecraft, dtype='fpi', resolution=save_res, create=True)

    if ion_source == 'omni':
        heavy_ions = import_processed_data('omni', resolution='5min')
    elif ion_source == 'hpca':
        heavy_ions = import_processed_data('mms1', dtype='hpca', resolution='raw')
        heavy_ions = heavy_ions[[f'N_{ion}' for ion in ion_species]] # Just need ion density columns

    ###----------PROCESS----------###
    for key in fgm_files:
        if key not in fpi_files:
            continue

        print(f'Updating {key}...')
        year, month = key.split('-')

        ion_df = heavy_ions.loc[(heavy_ions.index.year==int(year))&(heavy_ions.index.month==int(month))]
        fgm_df = import_processed_data(spacecraft, dtype='fgm', resolution=raw_res, file_name=fgm_files[key])
        fpi_df = import_processed_data(spacecraft, dtype='fpi', resolution=raw_res, file_name=fpi_files[key])

        updated_fpi = process_fpi_data(fpi_df, fgm_df, ion_df, ion_source=ion_source)

        print('Writing...')
        write_to_cdf(updated_fpi, directory=save_directory, file_name=f'{spacecraft}_FPI_{save_res}_{key}', attributes={'R_E': R_E}, overwrite=True, time_col='epoch', reset_index=True)


def assign_values(df, column, uarr, col_before=None):

    if col_before:
        idx = df.columns.get_loc(col_before)
        df.insert(idx, f'{column}_unc', unp.std_devs(uarr))
        df.insert(idx, column, unp.nominal_values(uarr))
    else:
        df[column]          = unp.nominal_values(uarr)
        df[f'{column}_unc'] = unp.std_devs(uarr)

def build_uarr(df, columns):
    if isinstance(columns, str):
        # single column
        values = np.nan_to_num(df[columns].to_numpy(), nan=0.0)
        uncs = np.nan_to_num(df[f'{columns}_unc'].to_numpy(), nan=0.0)
    else:
        # list of columns
        values = np.nan_to_num(df[columns].to_numpy(), nan=0.0)
        uncs = np.nan_to_num(df[[f'{c}_unc' for c in columns]].to_numpy(), nan=0.0)
    return unp.uarray(values, uncs)

def process_fpi_data(plasma_df, field_df, ion_df, ion_source='hpca'):
    """
    Once the plasma data has been extracted from the raw moments files
    This function convert the coordinates and calculate other parameters
    Will work on monthly files, and then resample afterwards
    """

    ###----------CLEAN UP RAW DATA----------###

    plasma_df['N_tot']      -= plasma_df['N_tot_bg'] # removes background counts
    plasma_df['P_th_tens']  -= plasma_df['P_th_bg']

    plasma_df.rename(columns={'P_th_tens': 'P_th', 'T_tens': 'T_tot', 'V_mag': 'V_flow', 'V_mag_unc': 'V_flow_unc'}, inplace=True)
    plasma_df.drop(columns=['N_tot_bg', 'P_th_bg'], inplace=True)

    # Align field on plasma timestamps (as plasma contains mode, quality etc. numbers)
    field_df = field_df.reindex(plasma_df.index, method=None).interpolate(method='time')
    field_df_cols = list(field_df.columns)

    merged_df = pd.concat([field_df, plasma_df], axis=1)

    print('Calculations...')

    ###----------CALCULATIONS----------###

    merged_df['V_flow_unc'] = (merged_df['V_x_GSE']**2 * merged_df['V_x_GSE_unc']**2 + merged_df['V_y_GSE']**2 * merged_df['V_y_GSE_unc']**2 + merged_df['V_z_GSE']**2 * merged_df['V_z_GSE_unc']**2) ** 0.5 / merged_df['V_flow']

    v_flow = build_uarr(merged_df, 'V_flow')

    # Dynamic pressure
    _  = calc_avg_ion_mass(merged_df, ion_df, ion_source, m_p, m_a, ION_MASS_DICT, default_ratio=0.05)

    n_tot = build_uarr(merged_df, 'N_tot')
    rho_tot = (merged_df['m_avg_ratio']*m_p) * n_tot # kg/cc

    mask = np.array(unp.nominal_values(rho_tot)) <= 0
    rho_tot[mask] = unp.uarray(np.nan, 0)

    # P = 0.5 * rho * V^2
    # N *= 1e6, V *= 1e6, P *= 1e9, so P_flow *= 1e21
    p_flow = 0.5 * rho_tot * v_flow**2 * 1e21
    assign_values(merged_df, 'P_flow', p_flow)

    # Beta = p_th / p_mag, p_mag = B^2/2mu_0
    # p_dyn *= 1e-9, 1/B_avg^2 *= 1e18, so beta *= 1e9
    P_th = build_uarr(merged_df, 'P_th')
    beta = P_th / (merged_df['B_avg']**2) * (2*mu_0) * 1e9
    assign_values(merged_df, 'beta', beta)

    # Alfven Speed = B / sqrt(mu_0 * rho)
    # B_avg *= 1e-9, 1/sqrt(rho) *= 1e-3, vA *= 1e-3, so speed *= 1e-15
    V_A = merged_df['B_avg'] / unp.sqrt(mu_0 * rho_tot) * 1e-15
    assign_values(merged_df, 'V_A', V_A)

    ###----------GSE to GSM----------###

    print('Rotating...')

    gsm_vectors = convert_GSE_to_GSM_with_angles(merged_df, [[f'V_{comp}_GSE' for comp in ('x','y','z')]], ref='B', interp=True, include_unc=True)

    merged_df = pd.concat([merged_df,gsm_vectors], axis=1)
    vec_coords = 'GSM'

    # Kan and Lee Electric Field: E_R = |V| * B_T * sin^2 (clock/2)
    # V *= 1e3, B *= 1e-9, E *= 1e3, so E_R *= 1e-3
    E_R = (v_flow * np.sqrt(merged_df[f'B_y_{vec_coords}']**2+merged_df[f'B_z_{vec_coords}']**2) * (np.sin(merged_df['B_clock']/2))**2) * 1e-3
    assign_values(merged_df, 'E_R', E_R)

    ###----------CROSS PRODUCTS----------###

    def vec_cols(field, vec_coords=vec_coords):
        return [f'{field}_x_{vec_coords}',f'{field}_y_{vec_coords}',f'{field}_z_{vec_coords}']

    def cross_u(a, b):
        # Handles uncertainties
        return np.stack([
            a[:,1]*b[:,2] - a[:,2]*b[:,1],
            a[:,2]*b[:,0] - a[:,0]*b[:,2],
            a[:,0]*b[:,1] - a[:,1]*b[:,0]
        ], axis=1)

    # Build uarray for V
    V_u = build_uarr(merged_df, vec_cols('V'))
    B = merged_df[vec_cols('B')].to_numpy()

    # E = -V x B = B x V
    # V *= 1e3, B *= 1e-9, and E *= 1e3 so E_gse *= 1e-3
    E_u = cross_u(B, V_u) * 1e-3
    merged_df.loc[:, vec_cols('E')] = unp.nominal_values(E_u)
    merged_df.loc[:, [f'{c}_unc' for c in vec_cols('E')]] = unp.std_devs(E_u)

    E_mag_u = unp.sqrt(np.sum(E_u**2, axis=1))
    assign_values(merged_df, 'E_mag', E_mag_u, col_before=f'E_x_{vec_coords}')

    # S = E x H = E x B / mu_0
    # E *= 1e-3, B *= 1e-9, and S *= 1e6 so S_gse *= 1e-6
    S_u = cross_u(E_u, B) * 1e-6 / mu_0
    merged_df.loc[:, vec_cols('S')] = unp.nominal_values(S_u)
    merged_df.loc[:, [f'{c}_unc' for c in vec_cols('S')]] = unp.std_devs(S_u)

    S_mag_u = unp.sqrt(np.sum(S_u**2, axis=1))
    assign_values(merged_df, 'S_mag', S_mag_u, col_before=f'S_x_{vec_coords}')

    merged_df.drop(columns=field_df_cols,inplace=True)

    return merged_df


def calc_avg_ion_mass(merged_df, ion_df, ion_source, m_p, m_a, ION_MASS_DICT, default_ratio=0.05):
    """
    This method assumes the relative amount of ions measured by the HPCA instrument (so the ratios) are correct.
    This ratio is then scaled by the total denisty measured by FPI.
    """

    try: # find ratio of alpha and proton densities
        if ion_source=='omni':
            print('Using OMNI alpha ratio.')
            merged_df = pd.merge_asof(merged_df.sort_index(), ion_df[['na_np_ratio']].sort_index(), left_index=True, right_index=True, direction='backward')   # take the closest value on or before the timestamp
            m_avg   = (m_p + merged_df['na_np_ratio'] * m_a) / (merged_df['na_np_ratio'] + 1) # kg
            idx = merged_df.columns.get_loc('N_tot')
            merged_df.insert(idx+2, 'm_avg_ratio', m_avg/m_p)

            return merged_df

        elif ion_source=='hpca':
            print('Using HPCA data.')
            # avoid extrapolation
            ion_interp = ion_df.reindex(merged_df.index).interpolate(method='time')
            ion_interp = ion_interp.loc[(ion_interp.index >= ion_df.index.min()) & (ion_interp.index <= ion_df.index.max())]

            # hplus density saturates in HPCA
            ion_ratio_map = {'heplus': 'nhe_np_ratio', 'oplus': 'no_np_ratio', 'heplusplus': 'na_np_ratio'}
            default_ratios = {'heplus': 0.0005, 'oplus': 0.01, 'heplusplus': 0.05}  # default values

            num = m_p
            den = 1.0

            idx = merged_df.columns.get_loc('N_tot')
            for ion, ratio_col in ion_ratio_map.items():

                r = ion_interp[f'N_{ion}'] / ion_interp['N_hplus']
                r = r.fillna(default_ratios[ion]).clip(lower=0)

                try:
                    merged_df.insert(idx+2, ratio_col, r)
                except:
                    merged_df[ratio_col] = r

                num += r * ION_MASS_DICT[ion]
                den += r

            m_avg = num / den # kg
            merged_df.insert(idx+2, 'm_avg_ratio', m_avg/m_p)

    except:
        print('Using default alpha ratio')

        idx = merged_df.columns.get_loc('N_tot')
        if 'na_np_ratio' in merged_df:
            merged_df['na_np_ratio'] = default_ratio
        else:
            merged_df.insert(idx+2, 'na_np_ratio', default_ratio)
        m_avg   = (m_p + merged_df['na_np_ratio'] * m_a) / (merged_df['na_np_ratio'] + 1) # kg
        merged_df.insert(idx+2, 'm_avg_ratio', m_avg/m_p)

    return None
