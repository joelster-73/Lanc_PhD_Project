import os
import glob

import numpy as np
import pandas as pd
from spacepy import pycdf
import spacepy.pycdf.istp

from scipy.constants import physical_constants
m_a = physical_constants['alpha particle mass'][0]

from .config import VARIABLES_DICT

from ..handling import log_missing_file, get_processed_files
from ..writing import write_to_cdf
from ..dataframes import add_df_units
from ..reading import import_processed_data
from ..process import process_overlapping_files

from ...coordinates.magnetic import calc_B_GSM_angles
from ...config import R_E, get_proc_directory, get_luna_directory

def process_cluster_files(spacecraft, data, data_info='SPIN', sample_intervals=('none',), time_col='epoch', year=None, overwrite=True, **kwargs):

    directory = get_luna_directory(spacecraft, instrument=data, info=data_info)

    # Process function
    if data in ('state','fgm'):
        if data=='state':
            process = process_cluster_state
        elif data=='fgm':
            process = process_cluster_fgm

        sub_folder = True if data_info=='5VPS' else False
        files_dict = get_cluster_files(directory, year, sub_folders=sub_folder)
        variables  = VARIABLES_DICT.get(data,{}).get(spacecraft.upper(),{}).get(data_info,{})

    elif data=='hia':
        process    = process_cluster_hia
        files_dict = get_cluster_files(directory, year, sub_folders=True)
        variables  = VARIABLES_DICT.get(data,{}).get(spacecraft.upper(),{})

        kwargs['qual_files'] = get_cluster_files(directory, year, sub_folders=True)
        kwargs['qual_vars']  = VARIABLES_DICT.get('quality',{}).get(spacecraft.upper(),{})

    else:
        raise ValueError(f'"{data}" not valid data to sample.')

    # Sample intervals
    samples = []
    for sample_interval in sample_intervals:

        if sample_interval=='none':
            if data in ('fgm',):
                sample_interval = 'raw' if data_info.lower()=='spin' else data_info.lower()
            if data in ('state',):
                sample_interval = data_info.lower()
            elif data in ('hia',):
                sample_interval = 'spin'

        samples.append(sample_interval)

    kwargs['resolutions'] = {'spin' : '4s', '5vps': '0.2s'}

    process_overlapping_files(spacecraft, data, process, variables, files_dict, samples, filt_func=filter_quality, **kwargs)

def get_cluster_files(directory=None, year=None, sub_folders=False):
    """
    Obtains a list of all files in the luna directory, organised by year-month
    """
    files_by_month = {}

    if directory is None:
        directory = os.getcwd()

    def add_file(key, path):
        files_by_month.setdefault(key, []).append(path)

    if sub_folders:
        for sub_folder in sorted(os.listdir(directory)):
            if year and sub_folder != str(year):
                continue

            sub_folder_path = os.path.join(directory, sub_folder)
            pattern = os.path.join(sub_folder_path, '*.cdf')

            for cdf_file in sorted(glob.glob(pattern)):
                if 'CAVEATS' in cdf_file:
                    continue

                # Expect __YYYYMMDD in filename
                yyyymm = os.path.basename(cdf_file).split('__')[1][:6]
                key = f'{yyyymm[:4]}-{yyyymm[4:6]}'
                add_file(key, cdf_file)
    else:
        if year:
            pattern = os.path.join(directory, f'*__{year}*.cdf')
        else:
            pattern = os.path.join(directory, '*.cdf')

        for cdf_file in sorted(glob.glob(pattern)):
            if 'CAVEATS' in cdf_file:
                continue

            yyyymm = os.path.basename(cdf_file).split('__')[1][:6]
            key = f'{yyyymm[:4]}-{yyyymm[4:6]}'
            add_file(key, cdf_file)

    if year and not any(k.startswith(str(year)) for k in files_by_month):
        raise ValueError(f'No files found for {year}.')
    if not files_by_month:
        raise ValueError('No files found.')

    return files_by_month

def extract_cluster_data(cdf_file, variables):

    data_dict = {}
    expected_len = None

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
            current_len = len(data) if data.ndim > 0 else 1
            if expected_len is None:
                expected_len = current_len
            elif current_len != expected_len:
                raise ValueError('Data arrays not same length.')


            if data.ndim == 2 and data.shape[1] == 3:  # Assuming a 2D array for vector components
                if var_name == 'r':
                    data /= R_E  # Scales distances to multiples of Earth radii

                data_dict[f'{var_name}_mag']   = np.sqrt(data[:,0]**2+data[:,1]**2+data[:,2]**2)
                data_dict[f'{var_name}_x_GSE'] = data[:, 0]
                data_dict[f'{var_name}_y_GSE'] = data[:, 1]
                data_dict[f'{var_name}_z_GSE'] = data[:, 2]

            else:
                if var_name == 'T_ion':
                    # data is in MK
                    data *= 1e6
                data_dict[var_name] = data  # pycdf extracts as datetime, no conversion from epoch needed

    return data_dict

# %% Process

def process_cluster_fgm(variables, files, directory_name, log_file_path, time_col='epoch', **kwargs):

    fgm_list = []

    # Loop through each daily file in the year
    for cdf_file in files:
        file_date = os.path.basename(cdf_file).split('__')[1].split('_')[0]
        try:  # Bad data check
            new_df = pd.DataFrame(extract_cluster_data(cdf_file, variables))
            if new_df.empty:
                log_missing_file(log_file_path, file_date, 'Empty file.')
                continue

            fgm_list.append(new_df)
            if False:
                print(f'{file_date} file appended.')

        except Exception as e:
            log_missing_file(log_file_path, file_date, e)

    if not fgm_list:
        return pd.DataFrame()

    fgm_df = pd.concat(fgm_list, ignore_index=True)
    add_df_units(fgm_df)

    return fgm_df

def process_cluster_state(variables, files, directory_name, log_file_path, time_col='epoch', **kwargs):

    state_list = []

    # Loop through each daily file in the year
    for cdf_file in files:
        file_date = os.path.basename(cdf_file).split('__')[1].split('_')[0]
        try:  # Bad data check
            new_df = pd.DataFrame(extract_cluster_data(cdf_file, variables))
            if new_df.empty:
                log_missing_file(log_file_path, file_date, 'Empty file.')
                continue

            state_list.append(new_df)
            if False:
                print(f'{file_date} file appended.')

        except Exception as e:
            log_missing_file(log_file_path, file_date, e)

    if not state_list:
        return pd.DataFrame()

    state_df = pd.concat(state_list, ignore_index=True)
    add_df_units(state_df)

    return state_df

def process_cluster_hia(variables, files, directory_name, log_file_path, time_col='epoch', **kwargs):
    """
    Also processes relevant quality files

    """
    hia_list = []

    # Loop through each daily file in the year
    for cdf_file in files:
        file_date = os.path.basename(cdf_file).split('__')[1].split('_')[0]
        try:  # Bad data check
            new_df = pd.DataFrame(extract_cluster_data(cdf_file, variables))
            if new_df.empty:
                log_missing_file(log_file_path, file_date, 'Empty file.')
                continue

            hia_list.append(new_df)
            if False:
                print(f'{file_date} file appended.')

        except Exception as e:
            log_missing_file(log_file_path, file_date, e)

    if not hia_list:
        return pd.DataFrame()

    hia_df = pd.concat(hia_list, ignore_index=True)
    add_df_units(hia_df)

    qual_files = kwargs.get('qual_files',[])
    qual_vars  = kwargs.get('qual_vars',{})
    if qual_files and qual_vars:
        process_cluster_hia_quality(hia_df, qual_vars, qual_vars, time_col)

    return hia_df

def process_cluster_hia_quality(df, variables, files, time_col='epoch'):

    print('Adding quality...')
    quality_list = []

    for qual_file in files:
        qual_df = pd.DataFrame(extract_cluster_data(qual_file, variables))
        if qual_df.empty:
            continue

        quality_list.append(qual_df)

    if len(quality_list)>0:
        quality_df = pd.concat(quality_list, ignore_index=True)
        quality_df[time_col] = pd.to_datetime(quality_df[time_col], errors='coerce')
        quality_df = quality_df.dropna(subset=[time_col])

        merged = pd.merge_asof(df, quality_df, left_on=time_col, right_on=time_col, direction='backward')

        df['quality'] = merged['quality'].values
        print('Quality added.')
    else:
        df['quality'] = -2 # indicate no quality data
        print('No quality data.')



# %% Update

def update_fgm_data(spacecraft, sample='raw'):
    """
    Updates fgm files with GSM data.

    """
    print('Processing CLUSTER.')

    directory      = get_proc_directory(spacecraft, dtype='fgm', resolution=sample)
    out_directory  = get_proc_directory(spacecraft, dtype='fgm', resolution='spin', create=True)

    files = get_processed_files(directory)

    for cdf_file in files:
        file_name = os.path.basename(cdf_file)
        print(f'Converting {file_name}.')

        key_df = import_processed_data(spacecraft, dtype='fgm', resolution=sample, file_name=file_name)
        attributes = key_df.attrs

        month = []

        for day, daily_df in key_df.groupby(pd.Grouper(freq='D')):
            print(day)
            if daily_df.empty:
                continue

            gsm = (calc_B_GSM_angles(daily_df, time_col='index').drop(columns=['B_mag']))

            daily_df = pd.concat([daily_df, gsm], axis=1)
            month.append(daily_df)

        key_df = pd.concat(month)
        key_df.attrs = attributes
        add_df_units(key_df)

        write_to_cdf(key_df, directory=out_directory, file_name=file_name, overwrite=True, reset_index=True)


# %% Plasma
def update_hia_data(field_df, plasma_df):

    ###----------CLEAN UP RAW DATA----------###

    # Align field on plasma timestamps (as plasma contains mode, quality etc. numbers)
    field_df = field_df.reindex(plasma_df.index, method=None).interpolate(method='time')

    # For now treating all species together
    plasma_df.rename(columns={'P_ion': 'P_th', 'T_ion': 'T_tot', 'N_ion': 'N_tot'}, inplace=True)
    plasma_df = filter_quality(plasma_df)

    field_df.loc[field_df['B_avg']>200,'B_avg'] = np.nan
    for comp in ('x','y','z'):
        field_df.loc[field_df[f'B_{comp}_GSE'].abs()>150,f'B_{comp}_GSE'] = np.nan

    plasma_df.loc[plasma_df['V_mag']>2e3,'V_mag'] = np.nan
    for comp, limit in zip(('x','y','z'),(2000,400,400)):
        plasma_df.loc[np.abs(plasma_df[f'V_{comp}_GSE'])>limit,f'V_{comp}_GSE'] = np.nan

    plasma_df.loc[plasma_df['N_tot']>500,'N_tot'] = np.nan
    plasma_df.loc[plasma_df['P_th']>100,'P_th'] = np.nan

    merged_df = pd.concat([field_df, plasma_df], axis=1)

    return merged_df

def filter_quality(df, column='quality'):

    bad_qualities = (0, 1, 2)
    # 0 : science mode
    # 1 : major
    # 2 : minor
    # 3 : good
    # 4 : excellent

    if column not in df:
        print(f'No {column} column.')
        return df

    mask = ~df[column].isin(bad_qualities)

    filtered_df = df.loc[mask]
    filtered_df.drop(columns=[column], inplace=True)
    filtered_df.attrs = df.attrs

    return filtered_df

def filter_hia_data(df, region='sw'):

    # Modes for CIS instrument
    region_modes = {'msh': [8,9,10,11,12,13,14],
                    'sw':  [0,1,2,3,4,5]}

    if 'mode' not in df:
        print('"mode" column not in dataframe.')
        return df

    mask = df['mode'].isin(region_modes.get(region))

    filtered_df = df.loc[mask]
    filtered_df.drop(columns=['mode'], inplace=True)
    filtered_df.attrs = df.attrs

    return filtered_df
