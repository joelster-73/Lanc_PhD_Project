import os
import glob

import pandas as pd
from spacepy import pycdf
import spacepy.pycdf.istp

from handling_files import create_log_file, log_missing_file, write_to_cdf, create_directory
from data_processing import replace_inf, add_df_units
from data_analysis import circular_mean
from coordinates import calc_B_GSM_angles

R_E = 6378 # Wind takes 1 earth radius to be 6378 km

### Main Code
"""
#Process all the data from the CDF files and save to a new CDF file
from handling_wind import process_wind_files
process_wind_files(luna_wind_dir, proc_wind_dir_1, wind_variables_1_hour, frequency='1hour')
process_wind_files(luna_wind_dir, proc_wind_dir, wind_variables, frequency='1min')
process_wind_files(luna_wind_dir, proc_wind_dir_3, wind_variables_3_sec, frequency='3sec')
"""

def process_wind_files(directory, data_directory, variables, frequency, year=None, time_col='epoch', overwrite=True):

    files_by_year = get_wind_files(directory, year=year)

    create_directory(data_directory)
    directory_name = os.path.basename(os.path.normpath(directory))
    log_file_path = os.path.join(data_directory, f'{directory_name}_files_not_added.txt')  # Stores not loaded files
    create_log_file(log_file_path)

    # Process each year's files
    for the_year, files in files_by_year.items():

        yearly_list = []

        # Loop through each daily file in the year
        for cdf_file in files:
            try:
                new_df = pd.DataFrame(extract_wind_data(cdf_file, variables))
                gsm = calc_B_GSM_angles(new_df)
                yearly_list.append(pd.concat([new_df, gsm], axis=1))
            except (AttributeError, ValueError, RuntimeError) as e:
                print('Known error.')
                log_missing_file(log_file_path, cdf_file, e)
            except Exception as e:
                print('Unknown error.')
                log_missing_file(log_file_path, cdf_file, e)

        if not yearly_list:
            print(f'No data for {the_year}')
            continue

        yearly_df = pd.concat(yearly_list, ignore_index=True)

        add_df_units(yearly_df)
        yearly_df = replace_inf(yearly_df, replace_large=True)
        yearly_df = reset_wind_indices(yearly_df, frequency, time_col) # Puts time stamps at beginning of bin

        output_file = os.path.join(data_directory, f'{directory_name}_{the_year}.cdf')
        attributes = {'data_frequency': frequency, 'time_col': time_col, 'R_E_km': R_E}
        write_to_cdf(yearly_df, output_file, attributes, overwrite)


def get_wind_files(directory, year=None):

    files_by_year = {}

    # If a specific year is passed, look within the year folder
    if year:
        year_folder = os.path.join(directory, str(year))
        if not os.path.isdir(year_folder):
            raise ValueError(f'Directory for the specified year {year} not found.')
        pattern = os.path.join(year_folder, '*.cdf')  # Match all .cdf files within the year folder
    else:
        pattern = os.path.join(directory, '*', '*.cdf')  # Match all .cdf files in all folders

    # Use glob to find files matching the pattern
    for cdf_file in sorted(glob.glob(pattern)):
        # Extract the year from the folder name (assuming folder name is YYYY)
        folder_year = os.path.basename(os.path.dirname(cdf_file))
        files_by_year.setdefault(folder_year, []).append(cdf_file)

    if year and str(year) not in files_by_year:
        raise ValueError(f"No files found for the specified year: {year}")

    return files_by_year



def extract_wind_data(cdf_file, variables):

    data_dict = {}

    with pycdf.CDF(cdf_file) as cdf:

        for var_name, var_code in variables.items():

            if var_code not in cdf:
                continue

            data = cdf[var_code].copy()
            if var_name not in ('epoch','epoch_pos'):
                spacepy.pycdf.istp.nanfill(data)
            data = data[...]

            if data.ndim == 2 and data.shape[1] == 3:  # Assuming a 2D array for vector components

                if 'GSE' in var_name:
                    coords = 'GSE'
                elif 'GSM' in var_name:
                    coords = 'GSM'
                else:
                    coords = 'GSE' # assume default
                field = var_name[0]
                # Split into components (e.g. x, y, z)
                data_dict[f'{field}_x_{coords}'] = data[:, 0]
                data_dict[f'{field}_y_{coords}'] = data[:, 1]
                data_dict[f'{field}_z_{coords}'] = data[:, 2]

            else:
                # Scalar data (not a vector)
                data_dict[var_name] = data.reshape(-1)  # flattens to 1D array

    return data_dict


def resample_wind_data(df, new_freq='1min'):
    """
    Resamples time series data in a DataFrame to specified intervals, calculating the mean for each bin,
    and adjusts the index based on the specified frequencies.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing time data (in CDF epoch format) and other data columns.
    new_freq : str, optional
        The target sampling interval for resampling. Default is '1min' (1-minute intervals).

    Returns
    -------
    pandas.DataFrame
        A DataFrame resampled to the specified interval with mean values calculated for each bin.
    """
    df = df.copy()  # Avoid modifying original DataFrame
    initial_freq = df.attrs['global'].get('data_frequency', None)

    # Define frequency mapping for clarity
    freq_map = {'3sec': '3S', '1min': '1min', '1hour': '1h'}

    if initial_freq not in freq_map or new_freq not in freq_map:
        raise ValueError("Both `initial_freq` and `new_freq` must be '3sec', '1min', or '1hour'.")

    new_pd_freq = freq_map[new_freq]

    # Ensure new frequency is lower than or equal to the initial frequency
    freq_order = ['3sec', '1min', '1hour']
    if freq_order.index(new_freq) < freq_order.index(initial_freq):
        raise ValueError("`new_freq` must be equal to or lower than `initial_freq`.")

    # Adjust index to the new frequency and calculate the mean
    df.index = df.index.floor(new_pd_freq)

    # Perform grouping and aggregation
    aggregated_data = {}
    df_grouped = df.groupby(df.index)
    for column in df.columns:
        if df.attrs['units'].get(column) == 'rad':
            # Use circular mean for 'rad' columns
            aggregated_data[column] = df_grouped[column].apply(lambda x: circular_mean(x))
        else:
            aggregated_data[column] = df_grouped[column].mean()

    df_aggregated = pd.DataFrame(aggregated_data)
    df_aggregated.dropna(inplace=True)
    df_aggregated.attrs['units'] = df.attrs['units']
    return df_aggregated

def reset_wind_indices(df, frequency, time_col='epoch'):
    """
    Adjusts the time indices of a DataFrame to the start of bins based on the initial frequency.
    The time is aligned to the start of the specified frequency by subtracting the required offset.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing time data (in CDF epoch format) and other data columns.

    frequency : str
        The sampling interval for resampling. Must be '3sec', '1min', or '1hour'.

    time_col : str, optional
        The name of the column containing time data to be resampled. Defaults to 'epoch'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the adjusted time indices (aligned to the start of the bins) in CDF epoch format.
    """
    # Define the frequency-to-offset mapping (in milliseconds)
    freq_map = {'3sec': '3S', '1min': '1min', '1hour': '1h'}

    # Ensure the provided frequency is valid
    if frequency not in freq_map:
        raise ValueError('"frequency" must be "3sec", "1min", or "1hour".')

    new_pd_freq = freq_map[frequency]
    df[time_col] = pd.to_datetime(df[time_col])
    df[time_col] = df[time_col].dt.floor(new_pd_freq)

    return df


