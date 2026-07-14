import os
import glob

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from .config import imf_bad_cols, plasma_bad_cols, column_units, omni_columns, omni_columns_5min

from ..handling import create_log_file, log_missing_file
from ..writing import write_to_cdf
from ..dataframes import add_df_units
from ..utils import create_directory
from ..speasy.calculations import cross_product, gse_to_gsm_with_angle

from ...coordinates.magnetic import calc_B_GSM_angles
from ...analysing.coupling import kan_lee_field
from ...config import get_luna_directory, get_proc_directory

from ...z_archive.processing_omni_handling import extract_omni_data_old

def process_omni_files(resolution='1min', year=None, overwrite=True, ext='asc'):

    directory = get_luna_directory('omni',resolution=resolution)
    data_directory = get_proc_directory('omni',resolution=resolution)
    if resolution=='1min':
        variables = omni_columns
    elif resolution=='5min':
        variables = omni_columns_5min

    print('Processing OMNI.\n')
    # Gather all OMNI files for the specified year (or all files if no year is specified)
    files_to_process = get_omni_files(directory, year=year, ext=ext)

    directory_name = os.path.basename(os.path.normpath(directory))

    if '5min' in directory_name:
        raw_dir = os.path.join(data_directory, '5min')
    elif 'min' in directory_name:
        raw_dir = os.path.join(data_directory, 'min')
    else:
        raw_dir = os.path.join(data_directory, 'raw')
    create_directory(raw_dir)

    log_file_path = os.path.join(data_directory, f'{directory_name}_not_added_files.txt')  # Log for unprocessed files
    create_log_file(log_file_path)

    # Process each file
    for i, omni_file in enumerate(files_to_process):
        file_name = os.path.basename(omni_file)
        print(f'Processing {file_name}.')

        try:
            if ext=='asc': # old version of OMNI
                data_dict = extract_omni_data_old(omni_file, variables)
                df = pd.DataFrame(data_dict)
                add_df_units(df)
            elif ext=='lst': # definitive version of OMNI
                df = extract_omni_data(omni_file, variables)
                df.attrs['units'] = column_units
            else:
                raise Exception(f'Unknown file type "{ext}".')

            print(f'Data from {file_name} extracted.')

        except Exception as e:
            log_missing_file(log_file_path, file_name, e)
            continue

        if ext=='asc':
            # Extract the year from the filename assuming the format omni_minYYYY.asc
            file_year = file_name.split('_')[1][3:7]
        elif ext=='lst':
            # Extract the year from the filename assuming the format omni_min_def_YYYY.lst
            file_year = file_name.split('_')[3][0:4]
        else:
            file_year = i+1

        output_file = os.path.join(raw_dir, f'{directory_name}_{file_year}.cdf')
        attributes = {'time_col': 'epoch'}
        print(f'Writing {file_name} to file...')
        write_to_cdf(df, output_file, attributes, overwrite)



def get_omni_files(directory, year=None, ext='asc'):

    # Build the search pattern based on the presence of a specific year
    if year:
        pattern = os.path.join(directory, f'*_{year}*.{ext}')  # Match files with the specified year
    else:
        pattern = os.path.join(directory, f'*.{ext}')  # Match all .asc files if no year is specified

    # Use glob to find files matching the pattern
    files_to_process = sorted(glob.glob(pattern))

    if not files_to_process:
        raise ValueError(f'No .{ext} files found in the directory: {directory}')

    return files_to_process

def extract_omni_data(lst_file, omni_columns):

    df = pd.read_csv(lst_file, sep=r'\s+', names=omni_columns)

    df.loc[df['imf_sc'] == 99, imf_bad_cols] = np.nan
    df.loc[df['plasma_sc'] == 99, plasma_bad_cols] = np.nan

    # Times
    years = df.loc[:,'Year'].astype(int)
    days = df.loc[:,'Day']
    hours = df.loc[:,'Hour']
    minutes = df.loc[:,'Minute']

    dates = [
        datetime(year, 1, 1) + timedelta(days=day - 1, hours=hour, minutes=minute)
        for year, day, hour, minute in zip(years, days, hours, minutes)
    ]

    df.insert(0, 'epoch', dates)
    df.drop(columns=['Year','Day','Hour','Minute'],inplace=True)

    # B_uncertainties
    B_avg_unc = df.loc[:, 'B_avg_rms'] / np.sqrt(df.loc[:, 'imf_counts'])
    B_vec_unc = df.loc[:, 'B_vec_rms'] / np.sqrt(df.loc[:, 'imf_counts'])

    df.insert(df.columns.get_loc('B_avg')+1, 'B_avg_unc', B_avg_unc)
    df.insert(df.columns.get_loc('B_avg')+2, 'B_vec_unc', B_vec_unc)
    df.drop(columns=['B_avg_rms'],inplace=True)
    df.drop(columns=['B_vec_rms'],inplace=True)

    # B_angles
    b_gsm = calc_B_GSM_angles(df, time_col='epoch')
    b_gsm.drop(columns=['B_mag'],inplace=True) # drops magntiude of average component
    df = pd.concat([df, b_gsm], axis=1)

    # b_gsm_unc = calc_B_angle_uncs(df, coords='GSM')
    # b_gsm_unc.drop(columns=['B_mag_unc'],inplace=True)
    # df = pd.concat([df, b_gsm_unc], axis=1)

    # Angle between IMF and BS normal at nose
    b = df[[f'B_{comp}_GSE' for comp in ('x','y','z')]].values
    r = df[[f'R_{comp}_BSN' for comp in ('x','y','z')]].values
    n = r / np.linalg.norm(r, axis=1)[:, None]
    cos_theta = np.einsum('ij,ij->i',b,n) / np.linalg.norm(b, axis=1)

    df['theta_Bn'] = np.clip(cos_theta, -1, 1)

    # E_GSM
    v_gsm = gse_to_gsm_with_angle(df, ref='B', vec='V')
    v_gsm.drop(columns=['V_x_GSM'],inplace=True) # Already have GSE component
    df = pd.concat([df, v_gsm], axis=1)

    # E = -V x B = B x V
    e_gsm = cross_product(df, cross_name='E_GSM', var1_name='B_GSM', var2_name='V_GSM')
    df = pd.concat([df, e_gsm], axis=1)

    # S = E x H = E x B / mu0
    s_gsm = cross_product(df, cross_name='S_GSM', var1_name='E_GSM', var2_name='B_GSM')
    df = pd.concat([df, s_gsm], axis=1)

    kan_lee_field(df)

    # Removing erroneous values
    df.loc[df['AE']>2000,'AE'] = np.nan
    df.loc[df['E_y']>20,'E_y'] = np.nan
    df.loc[df['S_mag']>150,'S_mag'] = np.nan
    df.loc[df['na_np_ratio']>1,'na_np_ratio'] = np.nan

    # Only in 5-minute data
    for flux_param in ('PSI_P_10','PSI_P_10','PSI_P_10'):
        if flux_param in df:
            df.loc[df[flux_param]==99999.99,flux_param] = np.nan

    return df




# %% Resample

def resample_omni_files(spacecraft, data, raw_res='spin', new_grouping='yearly', **kwargs):
    """
    Resample definitive omni files to a lower resolution, e.g. 15-min
    """

    def filter_mms_fgm(df):
        return filter_quality(df, column='B_flag')
    def filter_mms_fpi(df):
        return filter_quality(df, column='flag')

    QUAL_FUNCTIONS = {'fgm': filter_mms_fgm, 'fpi': filter_mms_fpi}
    kwargs['qual_func'] = QUAL_FUNCTIONS.get(data,None)

    resample_files(spacecraft, data, raw_res=raw_res, new_grouping=new_grouping, **kwargs)