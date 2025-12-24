import os
import glob
import re

import numpy as np
import pandas as pd
import itertools as it
from scipy.constants import k as kB, e

from spacepy import pycdf
import spacepy.pycdf.istp

from .config import VARIABLES_DICT

from ..dataframes import add_df_units
from ..handling import log_missing_file
from ..updating import process_overlapping_files

from ...coordinates.magnetic import calc_B_GSM_angles
from ...config import R_E, get_luna_directory

def process_themis_files(spacecraft, data, sample_intervals=('raw',), time_col='epoch', year=None, overwrite=True, **kwargs):

    directory = get_luna_directory(spacecraft, instrument=data)

    # Process function
    if data=='state':
        process = process_themis_state
    elif data=='fgm':
        process = process_themis_fgm
    elif data=='mom':
        process = process_themis_mom
    else:
        raise ValueError(f'"{data}" not valid data to sample.')

    files_dict = get_themis_files(directory, year)
    variables  = VARIABLES_DICT.get(data,{}).get(spacecraft,{})

    # Sample intervals
    samples = []
    for sample_interval in sample_intervals:

        if sample_interval=='none':
            sample_interval = '1min' if data == 'state' else 'raw'

        samples.append(sample_interval)

    kwargs['resolutions'] = {'spin' : '3s'}

    process_overlapping_files(spacecraft, data, process, variables, files_dict, samples, filt_func=filter_quality, **kwargs)

def get_themis_files(directory=None, year=None):

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

def extract_themis_data(cdf_file, variables):

    info_dict = {}
    flag_dict = {}

    with pycdf.CDF(cdf_file) as cdf:

        for var_name, var_code in variables.items():

            if var_code not in cdf:
                continue

            data = cdf[var_code].copy()
            if len(data)==0:
                continue
            spacepy.pycdf.istp.nanfill(data) # Replaces fill values with NANs
            data = data[...]

            if 'flag' in var_name:
                data_dict = flag_dict
            else:
                data_dict = info_dict

            if var_name == 'T_vec': # special case

                # Field aligned temperature; scalar is mean of components
                temp = (data[:,0] + data[:,1] + data[:,2]) / 3
                temp *= (e / kB)  # (eV → J) / (J/K) = K
                data_dict['T_ion'] = temp

            elif data.ndim == 2 and data.shape[1] == 3:  # Assuming a 2D array for vector components

                system = 'GSE'
                if var_name == 'r':
                    data /= R_E  # Scales distances to multiples of Earth radii

                elif '_GSE' in var_name or '_GSM' in var_name:
                    if '_GSM' in var_name:
                        system = 'GSM'

                    var_name = var_name.split('_')[0]

                data_dict[f'{var_name}_x_{system}'] = data[:, 0]
                data_dict[f'{var_name}_y_{system}'] = data[:, 1]
                data_dict[f'{var_name}_z_{system}'] = data[:, 2]

            else:
                if 'time' in var_name:
                    data = pd.to_datetime(data, unit='s', origin='unix')
                    var_name = var_name.replace('time','epoch') # consistency with Cluster

                elif var_name.startswith('P_'): # eV/cc
                    data *= (e * 1e6 * 1e9) # eV -> J -> /m3 -> nPa

                data_dict[var_name] = data  # pycdf extracts as datetime, no conversion from epoch needed

    return info_dict, flag_dict

def select_latest_versions(files):
    file_map = {}

    for file in files:
        # Extract the date and version from the filename (e.g., "tha_l1_state_20160506_v01.cdf")
        filename = os.path.basename(file)
        match = re.search(r'_(\d{8})_v(\d{2})\.cdf$', filename)
        if not match:
            continue  # Skip files that don't match the expected pattern

        date = match.group(1)
        version = int(match.group(2))

        # Keep only the latest version for each date
        if date not in file_map or version > file_map[date][1]:
            file_map[date] = (file, version)

    # Extract only the file paths
    return [file_version[0] for file_version in file_map.values()]

# %% Process

def process_themis_fgm(variables, files, directory_name, log_file_path, time_col='epoch', **kwargs):
    """
    FGS data is the priority (spin ~3s)
    """
    priority_suffices = kwargs.get('priority_suffices',('fgs','fgl','fgh','fge'))

    fgm_list = []

    for fgm_file in files:

        success = False
        for suffix in priority_suffices:
            fgm_variables = variables.get(suffix,{})
            if not fgm_variables:
                print(f'No variables for {suffix}')
                continue

            try:
                fgm_dict, _ = extract_themis_data(fgm_file, fgm_variables)
                if not fgm_dict:  # Check if the dictionary is empty
                    raise ValueError(f'{fgm_file} empty.')

                fgm_df = pd.DataFrame(fgm_dict)
                valid_b = ~fgm_df['B_avg'].isna()
                if np.sum(valid_b)==0:  # Check if valid B data
                    raise ValueError(f'{fgm_file} empty.')

                fgm_list.append(fgm_df)
                success = True
                break  # exit suffix loop if successful

            except (ValueError, FileNotFoundError, pd.errors.EmptyDataError):
                # Continue trying next suffix if specific expected errors
                continue

            except Exception as e:
                log_missing_file(log_file_path, fgm_file, e)
                success = True  # treat this as handled and exit suffix loop
                break

        if not success:
            # If none of the suffixes worked, log that the file was skipped
            log_missing_file(log_file_path, fgm_file, 'No valid suffix data found')

    ###---------------COMBINING------------###
    if not fgm_list:
        print('No data.')
        return pd.DataFrame()

    # Field data
    fgm_df = pd.concat(fgm_list, ignore_index=True)
    add_df_units(fgm_df)

    # GSM angles
    gsm = calc_B_GSM_angles(fgm_df, time_col=time_col)
    fgm_df = pd.concat([fgm_df, gsm], axis=1)
    add_df_units(fgm_df)

    return fgm_df

def process_themis_state(variables, files, directory_name, log_file_path, time_col='epoch', **kwargs):

    pos_list = []

    # Loop through each daily file in the year
    for pos_file in files:

        try:
            # position data at every minute
            pos_dict, _ = extract_themis_data(pos_file, variables)
            pos_df = pd.DataFrame(pos_dict)
            pos_list.append(pos_df)
        except:
            #print(f'{pos_file} empty.')
            log_missing_file(log_file_path, pos_file)
            continue

    ###---------------COMBINING------------###
    if not pos_list:
        print('No data.')
        return pd.Dataframe()

    pos_df = pd.concat(pos_list, ignore_index=True)
    add_df_units(pos_df)

    return pos_df

def process_themis_mom(variables, files, directory_name, log_file_path, time_col='epoch', **kwargs):

    plas_list = []
    flag_list = [] # solar wind/magnetosphere flag

    for file in files:

        try:
            # flag data is not exact same resolution - need to double check
            plasma_dict, flag_dict = extract_themis_data(file, variables)
            plas_list.append(pd.DataFrame(plasma_dict))
            flag_list.append(pd.DataFrame(flag_dict))

        except ValueError as e:
            raise Exception(e)
        except:
            log_missing_file(log_file_path, file)
            continue

    ###---------------COMBINING------------###
    if not (plas_list and flag_list):
        print('No data.')
        return pd.DataFrame()

    plas_df = pd.concat(plas_list, ignore_index=True).sort_values(by=['epoch'])
    flag_df   = pd.concat(flag_list, ignore_index=True).sort_values(by=['epoch_flag'])

    full_df = pd.merge_asof(plas_df, flag_df, left_on='epoch', right_on='epoch_flag', direction='backward')
    full_df.drop(columns=['epoch_flag'],inplace=True)
    add_df_units(full_df)

    return full_df

# %% Plasma

def update_esa_data(field_df, plasma_df):

    ###----------CLEAN UP RAW DATA----------###

    field_df = field_df[~field_df.index.duplicated(keep='first')]
    field_df.sort_index(inplace=True)

    field_df  = filter_quality(field_df, 'fgm')
    plasma_df = filter_quality(plasma_df, 'esa')

    merged_df = pd.merge_asof(plasma_df.reset_index(), field_df.reset_index(), on='epoch', direction='nearest', tolerance=pd.Timedelta('3s'))
    merged_df = merged_df.set_index('epoch')

    # For now treating all species together
    plasma_df.rename(columns={'P_ion': 'P_th', 'T_ion': 'T_tot', 'N_ion': 'N_tot'}, inplace=True)

    # Mistake made in original processing
    plasma_df.rename(columns={'V_GSE_x_GSE': 'V_x_GSE', 'V_GSE_y_GSE': 'V_y_GSE', 'V_GSE_z_GSE': 'V_z_GSE',
                      'V_GSM_x_GSE': 'V_x_GSM', 'V_GSM_y_GSE': 'V_y_GSM', 'V_GSM_z_GSE': 'V_z_GSM'}, inplace=True)

    return merged_df

def filter_quality(df, instrument='esa', column='quality'):

    if instrument=='esa':
        # Quality flags for ESA instrument
        # Could possibly remove 1 or 4
        # Quality = 0 indicates good data (-2 placeholder for no quality provided)
        qualities = (1, 4, 8, 16, 32, 64)

        bad_qualities = []
        for i in range(1, len(qualities)+1):
            for combo in it.combinations(qualities, i):
                bad_qualities.append(sum(combo))
        bad_qualities = sorted(bad_qualities)

    elif instrument=='fgm':
        bad_qualities = (1, 2, 3, 4) # 0 = good quality

    if column not in df:
        print(f'No "{column}" column.')
        return df

    mask = ~df[column].isin(bad_qualities)

    filtered_df = df.loc[mask]
    filtered_df.drop(columns=[column],inplace=True)
    filtered_df.attrs = df.attrs

    return filtered_df

def filter_esa_data(df, region='sw'):

    # Solar wind flag =1 in solar wind, =0 if not
    # So want to remove those with the wrong flag
    wrong_flag = 1
    if region=='sw':
        wrong_flag = 0

    if 'flag' not in df:
        print('"flag" not in dataframe.')
        return df

    # solar wind flag
    mask = (df['flag'].fillna(-2) != wrong_flag)

    filtered_df = df.loc[mask]
    filtered_df.drop(columns=['flag'],inplace=True)
    filtered_df.attrs = df.attrs

    return filtered_df