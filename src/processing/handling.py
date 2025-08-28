import os
import glob

from datetime import datetime

def create_log_file(log_file_path):
    """
    Creates a log file at the specified path with the current datetime as the first line,
    if the file does not already exist.

    Parameters
    ----------
    log_file_path : str
        The path where the log file will be created.

    Returns
    -------
        None: procedure just creates file.
    """
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Log created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")



def log_missing_file(log_file_path, file_path, e=None):
    """
    Appends the name of a missing file to the log file.

    Parameters
    ----------
    log_file_path : str
        The path to the log file where the missing file name will be appended.

    file_path : str
        The full path of the file that was not added. The file name will be extracted and logged.

    Returns
    -------
        None: procedure just updates file.
    """
    create_log_file(log_file_path)

    file_name = os.path.basename(file_path)
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{file_name} not added\n')
    if e is not None:
        print(f'{file_name} not added: {e}')


def get_processed_files(directory, year=None, keyword=None):
    """
    Retrieves a list of CDF files from the specified directory, optionally filtering by year and/or keyword.

    Parameters
    ----------
    directory : str
        The path to the directory where the CDF files are stored.

    year : str, optional
        A specific year to filter the files by (e.g., '2023').

    keyword : str, optional
        A keyword to filter the files by (e.g., 'OMNI').

    Returns
    -------
    list
        A sorted list of file paths that match the specified criteria.
    """
    if year and keyword:
        pattern = os.path.join(directory, f'*{keyword}*{year}*.cdf')
    elif year:
        pattern = os.path.join(directory, f'*{year}*.cdf')
    elif keyword:
        pattern = os.path.join(directory, f'*{keyword}*.cdf')
    else:
        pattern = os.path.join(directory, '*.cdf')

    # Use glob to find files matching the pattern
    files_processed = sorted(glob.glob(pattern))

    if not files_processed:
        raise ValueError(f"No files found in the directory: {directory}")

    return files_processed


def get_cdf_file(directory, filename=None):
    if filename:
        file_path = os.path.join(directory, filename)
        if not file_path.lower().endswith('.cdf'):
            raise ValueError(f"Provided file '{filename}' is not a .cdf file.")
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Specified file '{file_path}' not found.")
        return file_path

    # If no filename provided, look for CDF files in the directory
    cdf_files = glob.glob(os.path.join(directory, '*.cdf'))

    if len(cdf_files) != 1:
        raise ValueError(f"Expected one CDF file in the directory, found {len(cdf_files)}.")

    return cdf_files[0]



