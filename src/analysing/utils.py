# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:17:23 2025

@author: richarj2
"""
import os
from datetime import datetime

save_stats = False

def array_to_string(arr, fmt='.2f'):
    if fmt == '.2f':
        return f'[{", ".join(f"{value:.2f}" for value in arr)}]'

def save_statistics(text, directory='Figures'):
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
    if save_stats:
        # Get current timestamp and format it
        now = datetime.now()
        timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

        # Define the folder and file paths
        folder_name = now.strftime('%y%m%d')  # Folder based on the current date
        file_name = 'statistics.txt'  # The file is always named 'statistics.txt'

        # Determine the full path for the directory
        full_directory = os.path.join(directory, folder_name)

        # Ensure the directory exists
        os.makedirs(full_directory, exist_ok=True)

        # Create the full file path
        file_path = os.path.join(full_directory, file_name)

        # Open the file in append mode, adding timestamp and the text
        with open(file_path, 'a') as file:
            file.write(f'{timestamp}\n')  # Write the timestamp on the first line
            file.write(f'{text}\n')  # Write the provided text (may include \n)
            file.write('\n')  # Add a blank line after

        print(f'Statistics saved to {file_path}\n\n')