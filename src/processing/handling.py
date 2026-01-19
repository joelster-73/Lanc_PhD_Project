import os
import glob

from collections import defaultdict

from datetime import datetime
from .utils import create_directory
from ..config import get_proc_directory

def create_log_file(log_file_path):

    create_directory(os.path.dirname(log_file_path))
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Log created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def log_missing_file(log_file_path, file_path, e=None):

    create_log_file(log_file_path)

    file_name = os.path.basename(file_path)
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'{file_name} not added\n')
    if e is not None:
        print(f'{file_name} not added: {e}')


def get_processed_files(directory, year=None, keyword=None):

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
        path = os.path.join(directory, filename)
        root, ext = os.path.splitext(path)
        if not ext:
           ext = '.cdf'
        file_path = root + ext

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

def get_file_keys(spacecraft, data, raw_res='raw'):
    file_keys = {}
    data_dir = get_proc_directory(spacecraft, dtype=data, resolution=raw_res)
    pattern = os.path.join(data_dir, '*.cdf')

    for cdf_file in sorted(glob.glob(pattern)):
        file_name = os.path.splitext(os.path.basename(cdf_file))[0]
        key = file_name.split('_')[-1] # splitext removes the extension
        file_keys[key] = file_name

    return file_keys

def refactor_keys(keys_dict, new_grouping='yearly'):
    if '-' in list(keys_dict.keys())[0]:
        grouping = 'monthly'
    else:
        grouping = 'yearly'

    if grouping==new_grouping:
        return {k: [v] for k, v in keys_dict.items()}
    elif grouping=='yearly' and new_grouping=='monthly':
        print('Yearly to monthly not implemented.')
        return {k: [v] for k, v in keys_dict.items()}

    out = defaultdict(list) # if key doesn't exist, it's automatically created with the value type passed in, list here

    for k, v in keys_dict.items():
        year = k.split('-')[0]
        out[int(year)].append(v)

    out = {k: sorted(v) for k, v in sorted(out.items())}

    return out

def rename_files(directory, old_string, new_string):
    for filename in os.listdir(directory):
        if old_string not in filename:
            continue

        old_path = os.path.join(directory, filename)
        new_filename = filename.replace(old_string, new_string)
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)