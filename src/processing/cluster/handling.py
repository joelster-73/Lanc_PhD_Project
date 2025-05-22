import os
import glob

import pandas as pd
from spacepy import pycdf

from ..handling import create_log_file, log_missing_file
from ..writing import write_to_cdf
from ..dataframes import add_df_units, resample_data

from ...coordinates.magnetic import calc_B_GSM_angles
from ...config import R_E

def process_cluster_files(directory, data_directory, variables, sample_interval='1min', time_col='epoch', year=None, sub_folders=False, overwrite=True):
    """
    Processes CDF files from a specified directory, extracts specified variables, resamples the data,
    and appends it to a new or existing CDF file. Files that could not be processed are logged.

    Parameters
    ----------
    directory : str
        The path to the directory containing the CDF files to process.

    data_directory : str
        The directory where processed CDF files will be saved.

    variables : dict
        A dictionary where keys are the variable names for the resulting DataFrame, and values are the
        variable codes used to extract data from the CDF files.

    sample_interval : str, optional
        The interval for resampling the data. The default is '1min' for one-minute intervals.

    time_col : str, optional
        The name of the column in the DataFrame containing time data (in CDF epoch format).
        The default is 'epoch'.

    year : str, optional
        The year to filter CDF files by. If None, all years in the directory are processed.

    overwrite : bool, optional
        Whether to overwrite existing CDF files. The default is True.

    Returns
    -------
    None
        This function processes the data and appends it to a CDF file in the `data_directory`.
        It logs any files that could not be processed.
    """
    print('Processing Cluster.')
    files_by_year = get_cluster_files(directory, year, sub_folders)

    directory_name = os.path.basename(os.path.normpath(directory))
    log_file_path = os.path.join(data_directory, f'{directory_name}_files_not_added.txt')  # Stores not loaded files
    create_log_file(log_file_path)

    # Process each year's files
    for the_year, files in files_by_year.items():
        print(f'Processing {the_year} data.')
        yearly_list = []

        # Loop through each daily file in the year
        for cdf_file in files:
            file_date = os.path.basename(cdf_file).split('__')[1].split('_')[0]
            try:  # Bad data check
                new_df = pd.DataFrame(extract_cluster_data(cdf_file, variables))
                gsm = calc_B_GSM_angles(new_df, time_col=time_col)
                yearly_list.append(pd.concat([new_df, gsm], axis=1))
                print(f'{file_date} file appended.')
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

        # write raw data to file - before resampling
        raw_dir = os.path.join(data_directory, 'raw')
        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)
        output_file = os.path.join(raw_dir, f'{directory_name}_{the_year}.cdf')
        if '5VPS' in directory_name:
            raw_sample = '5VPS'
        elif 'SPIN' in directory_name:
            raw_sample = 'SPIN'
        attributes = {'sample_interval': raw_sample, 'time_col': time_col, 'R_E_km': R_E}
        write_to_cdf(yearly_df, output_file, attributes, overwrite)

        print('Resampling.')
        # resample and write to file
        sampled_df = resample_data(yearly_df, time_col, sample_interval)
        add_df_units(sampled_df)

        samp_dir = os.path.join(data_directory, sample_interval)
        if not os.path.exists(samp_dir):
            os.makedirs(samp_dir)
        output_file = os.path.join(samp_dir, f'{directory_name}_{the_year}.cdf')
        attributes = {'sample_interval': sample_interval, 'time_col': time_col, 'R_E_km': R_E}
        write_to_cdf(sampled_df, output_file, attributes, overwrite)

        print(f'{the_year} processed.')


def get_cluster_files(directory=None, year=None, sub_folders=False):
    """
    Retrieves CDF files from a specified directory, optionally filtered by a specific year.

    Parameters
    ----------
    directory : str
        The path to the directory containing the CDF files to search.

    year : str, optional
        The year to filter the CDF files by. If not specified, all CDF files in the directory are retrieved.

    Returns
    -------
    dict
        A dictionary where the keys are years (as strings) and the values are lists of file paths
        for CDF files associated with that year.
    """
    files_by_year = {}
    
    if directory is None:
        directory = os.getcwd()

    if sub_folders:
        # Loop through subdirectories named as years
        for sub_folder in sorted(os.listdir(directory)):
            sub_folder_path = os.path.join(directory, sub_folder)
            if year and sub_folder != str(year):
                continue  # Skip folders not matching the specified year

            pattern = os.path.join(sub_folder_path, '*.cdf')  # Match all .cdf files in the folder
            for cdf_file in sorted(glob.glob(pattern)):
                files_by_year.setdefault(sub_folder, []).append(cdf_file)
    else:

        # Build the search pattern based on the presence of a specific year
        if year:
            pattern = os.path.join(directory, f'*__{year}*.cdf')  # Match only files with the given year
        else:
            pattern = os.path.join(directory, '*.cdf')  # Match all .cdf files if no year is specified

        # Use glob to find files matching the pattern
        for cdf_file in sorted(glob.glob(pattern)):
            # Extract the year from the filename assuming format data-name__YYYYMMDD_version.cdf
            file_year = os.path.basename(cdf_file).split('__')[1][:4]
            files_by_year.setdefault(file_year, []).append(cdf_file)

    if year and str(year) not in files_by_year:
        raise ValueError(f'No files found for {year}.')
    if not files_by_year:
        raise ValueError('No files found.')

    return files_by_year


def extract_cluster_data(cdf_file, variables):
    """
    Extracts specified variables from a CDF file and returns them in a dictionary.

    Parameters
    ----------
    cdf_file : str
        Path to the CDF file to read.

    variables : dict
        A dictionary where keys are the variable names (as strings) to be used in the output,
        and values are the corresponding variable codes (as strings) used to extract data
        from the CDF file.

    Returns
    -------
    dict
        A dictionary where the keys are the variable names, and the values are the corresponding
        data arrays. For vector variables, the components are stored as separate keys with '_x_GSE',
        '_y_GSE', and '_z_GSE' suffixes.
    """

    # Initialise a dictionary to store the data
    data_dict = {}

    # Load the CDF file (auto closes)
    with pycdf.CDF(cdf_file) as cdf:

        # Loop through the dictionary of variables and extract data
        for var_name, var_code in variables.items():
            data = cdf[var_code][...]  # Extract the data using the CDF variable code

            if data.ndim == 2 and data.shape[1] == 3:  # Assuming a 2D array for vector components
                if var_name == 'r':
                    data /= R_E  # Scales distances to multiples of Earth radii

                # Split into components (e.g. x, y, z)
                data_dict[f'{var_name}_x_GSE'] = data[:, 0]
                data_dict[f'{var_name}_y_GSE'] = data[:, 1]
                data_dict[f'{var_name}_z_GSE'] = data[:, 2]

            else:
                # Scalar data (not a vector)
                data_dict[var_name] = data  # pycdf extracts as datetime, no conversion from epoch needed

    return data_dict

