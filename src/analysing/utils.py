# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:17:23 2025

@author: richarj2
"""
import os
from datetime import datetime

from ..config import FIGURES_DIR

def array_to_string(arr, fmt='.2f'):
    if fmt == '.2f':
        return f'[{", ".join(f"{value:.2f}" for value in arr)}]'

def save_statistics(text, directory=None, sub_directory=None):
    """
    Appends a string with a timestamp to a text file in the 'figures' folder,
    creating the appropriate date folder if it does not exist.

    Parameters:
    directory (str): The root directory where 'figures' folder is located.
    text (str): The string to be appended to the text file, which may include newlines.

    Notes:
    The text file is saved in the 'figures' folder, under a subfolder named with the current date
    in the format 'yyMMdd', and the file is always named 'statistics.txt'.
    """

    now = datetime.now()
    folder_name = now.strftime('%y%m%d')
    if directory is None:
        directory = FIGURES_DIR
    if sub_directory is not None:
        if '/' in sub_directory:
            sub_directory = sub_directory.split('/')
            full_directory = os.path.join(directory, folder_name, *sub_directory)
        else:
            full_directory = os.path.join(directory, folder_name, sub_directory)

    else:
        full_directory = os.path.join(directory, folder_name)

    os.makedirs(full_directory, exist_ok=True)

    # Define the folder and file paths
    file_name = 'statistics.txt'
    timestamp = now.strftime('%Y-%m-%d %H:%M:%S')
    file_path = os.path.join(full_directory, file_name)

    with open(file_path, 'a') as file:
        file.write(f'{timestamp}\n')
        file.write(f'{text}\n')
        file.write('\n')

    print(f'Statistics saved to {file_path}\n\n')