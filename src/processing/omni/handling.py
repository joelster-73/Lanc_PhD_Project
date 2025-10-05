import os
import glob

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from .config import imf_bad_cols, plasma_bad_cols, column_units

from ..handling import create_log_file, log_missing_file
from ..writing import write_to_cdf
from ..dataframes import add_df_units
from ..utils import create_directory
from ..speasy.calculations import cross_product, gse_to_gsm_with_angle

from ...coordinates.magnetic import calc_B_GSM_angles
from ...analysing.coupling import kan_lee_field


def process_omni_files(directory, data_directory, variables, year=None, overwrite=True, ext='asc'):

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
            if ext=='asc':
                data_dict = extract_omni_data_old(omni_file, variables)
                df = pd.DataFrame(data_dict)
                add_df_units(df)
            elif ext=='lst':
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


def extract_omni_data_old(asc_file, variables):
    """
    For the old version of OMNI that is in ascii files
    The newer version is in .list files
    And uses definitive WIND data
    The columns are also in a different order
    """

    # Load data from the ASCII file
    data_set = np.array(np.loadtxt(asc_file))  # each element is a row; one row per minute

    # Initialize DataFrame to store extracted data
    df = pd.DataFrame()

    # Add spacecraft IDs for filtering
    df['imf_sc']    = data_set[:, 4]
    df['plasma_sc'] = data_set[:, 5]

    # Iterate over requested variables and extract data
    for var in variables:
        if var == 'time':
            years = data_set[:, 0].astype(int)
            days = data_set[:, 1]
            hours = data_set[:, 2]
            minutes = data_set[:, 3]

            dates = [
                datetime(year, 1, 1) + timedelta(days=day - 1, hours=hour, minutes=minute)
                for year, day, hour, minute in zip(years, days, hours, minutes)
            ]

            # Convert datetime to CDF epoch format
            df['epoch'] = dates

        elif var == 'B_field':
            # Magnetic field data (nT)
            df['B_avg']   = data_set[:, 13]  # nT
            df['B_x_GSE'] = data_set[:, 14]
            df['B_y_GSE'] = data_set[:, 15]
            df['B_z_GSE'] = data_set[:, 16]
            df['B_y_GSM'] = data_set[:, 17]
            df['B_z_GSM'] = data_set[:, 18]

            # Set to NaN for bad data (sc_id 99)
            df.loc[df['imf_sc'] == 99, ['B_avg', 'B_x_GSE', 'B_y_GSE', 'B_z_GSE', 'B_y_GSM', 'B_z_GSM']] = np.nan

            gsm = calc_B_GSM_angles(df, time_col='epoch')
            df = pd.concat([df, gsm], axis=1)

        elif var == 'pressure':
            # Flow pressure calculation: (2*10**-6)*Np*Vp**2 nPa
            df['p_flow'] = data_set[:, 27]  # nPa

            # Set to NaN for bad data (sc_id 99)
            df.loc[df['plasma_sc'] == 99, 'p_flow'] = np.nan

        elif var == 'density':
            df['n_p'] = data_set[:, 25]  # n/cc

            # Set to NaN for bad data (sc_id 99)
            df.loc[df['plasma_sc'] == 99, 'n_p'] = np.nan

        elif var == 'velocity':
            df['v_flow']  = data_set[:, 21]  # km/s - something odd
            df['v_x_GSE'] = data_set[:, 22]  # km/s
            df['v_y_GSE'] = data_set[:, 23]  # km/s
            df['v_z_GSE'] = data_set[:, 24]  # km/s

            # Set to NaN for bad data (sc_id 99)
            df.loc[df['plasma_sc'] == 99, ['v_flow', 'v_x_GSE', 'v_y_GSE', 'v_z_GSE']] = np.nan

        elif var == 'satellite':
            # Satellite ID and position data
            df['sc_id']   = data_set[:, 5]
            df['r_x_GSE'] = data_set[:, 31]
            df['r_y_GSE'] = data_set[:, 32]
            df['r_z_GSE'] = data_set[:, 33]

            # Set to NaN for bad data (sc_id 99)
            df.loc[df['plasma_sc'] == 99, ['sc_id', 'r_x_GSE', 'r_y_GSE', 'r_z_GSE']] = np.nan

        elif var == 'bow_shock_nose':
            df['r_x_BSN'] = data_set[:, 34]
            df['r_y_BSN'] = data_set[:, 35]
            df['r_z_BSN'] = data_set[:, 36]

            # Set to NaN for bad data (sc_id 99)
            df.loc[df['imf_sc'] == 99, ['r_x_BSN', 'r_y_BSN', 'r_z_BSN']] = np.nan

        elif var == 'propagation':
            # Satellite ID and position data
            df['prop_time_s'] = data_set[:, 9]

            # Set to NaN for bad data (sc_id 99)
            df.loc[df['imf_sc'] == 99, ['prop_time_s']] = np.nan

        else:
            raise ValueError(f'Unknown variable {var}')

    # Drop spacecraft IDs as they are no longer needed
    df = df.drop(columns=['imf_sc', 'plasma_sc'])
    return df