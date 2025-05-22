import os
import glob

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from ..handling import create_log_file, log_missing_file
from ..writing import write_to_cdf
from ..dataframes import add_df_units
from ...coordinates.magnetic import calc_B_GSM_angles

### Main Code
"""
# Process all the data from the CDF files and save to a new CDF file
from handling_omni import process_omni_files
process_omni_files(luna_omni_dir, proc_omni_dir, omni_variables)
"""

def process_omni_files(directory, data_directory, variables, year=None, overwrite=True):
    """
    Processes a list of OMNI CDF files, extracts the specified variables, and saves the data to a CDF file.

    Parameters
    ----------
    directory : str
        The path to the directory containing the OMNI ASCII files to process.

    data_directory : str
        The directory where the processed CDF files will be saved.

    year : str, optional
        The year to filter OMNI files by. If None, all files in the directory will be processed.

    variables : dict
        Dictionary where the keys are the variable names to be used in the resulting DataFrame,
        and the values are the variable codes used to extract data from the CDF files.

    overwrite : bool, optional
        Whether to overwrite existing CDF files. Defaults to True.

    Returns
    -------
    None
        This function processes the data and saves it to a CDF file in the `data_directory`.
    """
    print('Processing OMNI.\n')
    # Gather all OMNI files for the specified year (or all files if no year is specified)
    files_to_process = get_omni_files(directory, year=year)

    directory_name = os.path.basename(os.path.normpath(directory))
    log_file_path = os.path.join(data_directory, f'{directory_name}_not_added_files.txt')  # Log for unprocessed files
    create_log_file(log_file_path)

    # Process each file
    for asc_file in files_to_process:
        file_name= os.path.basename(asc_file)
        print(f'Processing {file_name}.')
        try:
            data_dict = extract_omni_data(asc_file, variables)
            print(f'Data from {file_name} extracted.')
            df = pd.DataFrame(data_dict)
            add_df_units(df)
            # Extract the year from the filename assuming the format omni_minYYYY.asc
            file_year = file_name.split('_')[1][3:7]

            output_file = os.path.join(data_directory, f'{directory_name}_{file_year}.cdf')
            attributes = {'time_col': 'epoch'}
            write_to_cdf(df, output_file, attributes, overwrite)
            print(f'Written {file_name} to file.\n')

        except (AttributeError, ValueError, RuntimeError) as e:
            print('Known error.')
            log_missing_file(log_file_path, asc_file, e)
        except Exception as e:
            print('Unknown error.')
            log_missing_file(log_file_path, asc_file, e)



def get_omni_files(directory, year=None):
    """
    Retrieves .asc files from a specified directory, optionally filtered by a specific year.

    Parameters
    ----------
    directory : str
        The path to the directory containing the .asc files to search.

    year : str, optional
        The year to filter the .asc files by. If not specified, all .asc files in the directory are retrieved.

    Returns
    -------
    list
        A sorted list of file paths for .asc files in the specified directory that match the search pattern.
    """
    # Build the search pattern based on the presence of a specific year
    if year:
        pattern = os.path.join(directory, f'*_{year}*.asc')  # Match files with the specified year
    else:
        pattern = os.path.join(directory, '*.asc')  # Match all .asc files if no year is specified

    # Use glob to find files matching the pattern
    files_to_process = sorted(glob.glob(pattern))

    if not files_to_process:
        raise ValueError(f'No .asc files found in the directory: {directory}')

    return files_to_process


def extract_omni_data(asc_file, variables):
    """
    Loads specified variables from an ASCII file and returns a DataFrame with extracted data.

    Parameters
    ----------
    asc_file : str
        The path to the ASCII file.

    variables : dict
        Dictionary where keys are the variable names to be used in the DataFrame,
        and values are the variable codes used to extract data from the file.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns for each requested variable.

    Raises
    ------
    ValueError
        If an unknown variable is requested.
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
            df['v_flow']  = data_set[:, 21]  # km/s
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