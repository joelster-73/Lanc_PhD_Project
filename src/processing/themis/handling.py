import os
import glob

import pandas as pd
import re
from spacepy import pycdf

from ..writing import write_to_cdf
from ..dataframes import add_df_units, resample_data
from ..handling import create_log_file, log_missing_file
from ..utils import create_directory

from ...coordinates.magnetic import calc_B_GSM_angles
from ...config import R_E


def process_themis_files(spacecraft, themis_directories, proc_directory, fgm_labels, pos_labels, data_resolution='1/128s',
                         sample_interval='1min', time_col='epoch', year=None, sub_folders=False, overwrite=True,
                         priority_suffixes = ['fgh', 'fgl', 'fge', 'fgs']):
    """

    Returns
    -------
    None
        This function processes the data and appends it to a CDF file in the `data_directory`.
        It logs any files that could not be processed.
    """

    spacecraft_dir = themis_directories[spacecraft]
    fgm_directory = f"{spacecraft_dir}/FGM"  # FGM directory inside the spacecraft folder
    pos_directory = f"{spacecraft_dir}/STATE"  # STATE directory inside the spacecraft folder
    out_directory = proc_directory[spacecraft]
    pos_variables = pos_labels[spacecraft]


    fgm_directory_name = os.path.basename(os.path.normpath(fgm_directory))
    pos_directory_name = os.path.basename(os.path.normpath(pos_directory))

    create_directory(out_directory)

    fgm_log_file_path = os.path.join(out_directory, f'{fgm_directory_name}_files_not_added.txt')  # Stores not loaded files
    pos_log_file_path = os.path.join(out_directory, f'{pos_directory_name}_files_not_added.txt')  # Stores not loaded files

    create_log_file(fgm_log_file_path)
    create_log_file(pos_log_file_path)

    print('Processing THEMIS.')
    fgm_files_by_year = get_themis_files(fgm_directory, year, sub_folders)
    pos_files_by_year = get_themis_files(pos_directory, year, sub_folders)

    # Process each year's files
    for (fgm_year, fgm_files), (pos_year, pos_files) in zip(fgm_files_by_year.items(), pos_files_by_year.items()):
        print(f'Processing {fgm_year} data.')


        ###---------------STATE DATA------------###

        pos_yearly_list = []

        # Loop through each daily file in the year
        for pos_file in pos_files:

            try:
                # position data at every minute
                pos_df = pd.DataFrame(extract_themis_data(pos_file, pos_variables))
                pos_yearly_list.append(pos_df)
            except:
                #print(f'{pos_file} empty.')
                log_missing_file(pos_log_file_path, pos_file)
                continue

        ###---------------FGM DATA------------###

        fgm_yearly_list = []

        for fgm_file in fgm_files:

            success = False
            for suffix in priority_suffixes:
                fgm_variables = fgm_labels[spacecraft][suffix]

                try:
                    fgm_dict = extract_themis_data(fgm_file, fgm_variables)
                    fgm_df = pd.DataFrame(fgm_dict)
                    if not fgm_dict:  # Check if the dictionary is empty
                        raise ValueError(f'{fgm_file} empty.')

                    # GSM angles
                    gsm = calc_B_GSM_angles(fgm_df, time_col=time_col)
                    fgm_df = pd.concat([fgm_df, gsm], axis=1)
                    fgm_yearly_list.append(fgm_df)

                    success = True
                    break  # exit suffix loop if successful

                except (ValueError, FileNotFoundError, pd.errors.EmptyDataError):
                    # Continue trying next suffix if specific expected errors
                    continue

                except Exception as e:
                    log_missing_file(fgm_log_file_path, fgm_file, e)
                    success = True  # treat this as handled and exit suffix loop
                    break

            if not success:
                # If none of the suffixes worked, log that the file was skipped
                log_missing_file(fgm_log_file_path, fgm_file, 'No valid suffix data found')

        ###---------------COMBINING------------###
        if not fgm_yearly_list and not pos_yearly_list:
            print(f'No data for {fgm_year}')
            continue

        pos_yearly_df = pd.concat(pos_yearly_list, ignore_index=True)

        fgm_yearly_df = pd.concat(fgm_yearly_list, ignore_index=True)
        add_df_units(fgm_yearly_df)

        # Resampling has to be done after as THEMIS files overlap on times
        fgm_yearly_df = resample_data(fgm_yearly_df, time_col, sample_interval, show_count=True)
        add_df_units(fgm_yearly_df)

        merged_df = pd.merge(pos_yearly_df, fgm_yearly_df, left_on=time_col, right_on=time_col, how='left')
        add_df_units(merged_df)

        output_file = os.path.join(out_directory, f'{fgm_directory_name}_{fgm_year}.cdf')
        attributes = {'sample_interval': data_resolution, 'time_col': time_col, 'R_E_km': R_E}
        write_to_cdf(merged_df, output_file, attributes, overwrite)

        print(f'{fgm_year} processed.')
        print(merged_df)


def get_themis_files(directory=None, year=None, sub_folders=False):
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
            files = sorted(glob.glob(pattern))
            latest_files = select_latest_versions(files)
            files_by_year[sub_folder] = latest_files
    else:

        # Build the search pattern based on the presence of a specific year
        if year:
            pattern = os.path.join(directory, f'*__{year}*.cdf')  # Match only files with the given year
        else:
            pattern = os.path.join(directory, '*.cdf')  # Match all .cdf files if no year is specified

        # Use glob to find files matching the pattern
        files = sorted(glob.glob(pattern))
        latest_files = select_latest_versions(files)
        for file in latest_files:
            # Extract the year from the filename assuming format data-name__YYYYMMDD_version.cdf
            file_year = os.path.basename(file).split('_')[3][:4]
            files_by_year.setdefault(file_year, []).append(file)

    if year and str(year) not in files_by_year:
        raise ValueError(f'No files found for {year}.')
    if not files_by_year:
        raise ValueError('No files found.')

    return files_by_year


def extract_themis_data(cdf_file, variables):
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
                system = 'GSE'
                if var_name == 'r':
                    data /= R_E  # Scales distances to multiples of Earth radii
                elif 'B' in var_name:
                    if 'GSM' in var_name:
                        system = 'GSM'
                    var_name = 'B'

                # Split into components (e.g. x, y, z)
                data_dict[f'{var_name}_x_{system}'] = data[:, 0]
                data_dict[f'{var_name}_y_{system}'] = data[:, 1]
                data_dict[f'{var_name}_z_{system}'] = data[:, 2]

            else:
                if var_name == 'time':
                    data = pd.to_datetime(data, unit='s', origin='unix')
                    var_name = 'epoch' # consistency with Cluster
                # Scalar data (not a vector)
                data_dict[var_name] = data  # pycdf extracts as datetime, no conversion from epoch needed

    return data_dict

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
