# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:27:58 2025

@author: richarj2
"""

import os
import numpy as np
import pandas as pd
import math

from pandas import Timedelta
from datetime import datetime

import imageio

#import config
from .config import save_fig
from ..config import FIGURES_DIR


def save_figure(figure, directory=None, sub_directory=None, file_name=None):
    """
    Save a matplotlib figure as a PNG file with a timestamped name (HHMMSS.png).
    Saves the file in a folder named YYMMDD within the specified directory.
    Creates the folder if it does not exist.

    Parameters:
    -----------
    figure : matplotlib.figure.Figure
        The figure object to be saved.
    directory : str, optional
        The parent directory to save the file in. Defaults to "Figures" in the current working directory.

    Returns:
    --------
    None
    """
    if save_fig:
        now = datetime.now()
        folder_name = now.strftime('%y%m%d')
        if directory is None:
            directory = FIGURES_DIR
        if sub_directory is not None:
            full_directory = os.path.join(directory, folder_name, sub_directory)
        else:
            full_directory = os.path.join(directory, folder_name)
        os.makedirs(full_directory, exist_ok=True)

        if file_name is None:
            file_name = now.strftime('%H%M%S')
        file_name += '.png'
        file_path = os.path.join(full_directory, file_name)
        base_name, extension = os.path.splitext(file_name)

        counter = 2
        while os.path.exists(file_path):
            file_path = os.path.join(full_directory, f'{base_name}_({counter}){extension}')
            counter += 1

        figure.savefig(file_path, format='png',bbox_inches='tight')
        print(f'\nFigure saved as {file_path}\n')

def save_frame(frame, i, frame_files, directory='Figures'):

    now = datetime.now()
    folder_name = now.strftime('%y%m%d')
    full_directory = os.path.join(directory, folder_name, 'Frames')
    os.makedirs(full_directory, exist_ok=True)

    file_name = f'frame_{i:03d}.png'
    file_path = os.path.join(full_directory, file_name)
    frame_files.append(file_path)

    # Save the figure
    frame.savefig(file_path, format='png', bbox_inches='tight', dpi=200)


def save_gif(frame_files, length=1, directory='Figures'):

    now = datetime.now()
    folder_name = now.strftime('%y%m%d')
    full_directory = os.path.join(directory, folder_name)

    gif_filename = now.strftime('%H%M%S') + '.gif'
    file_path = os.path.join(full_directory, gif_filename)

    # Save the figure
    frame_rate = 1/length
    with imageio.get_writer(file_path, mode='I', fps=frame_rate) as writer:
        for frame_file in frame_files:
            image = imageio.imread(frame_file)
            writer.append_data(image)

    # Clean up frame files
    for frame_file in frame_files:
        os.remove(frame_file)

    print(f'GIF saved as {file_path}\n')





def segment_dataframe(df, delta=Timedelta(minutes=1)):
    """
    Adds a 'segment' column to the DataFrame based on time gaps exceeding a threshold.
    This is useful for segmenting time-series data into separate chunks, typically for plotting purposes.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing time-series data to be segmented. The DataFrame must have a DateTimeIndex.
    delta : pd.Timedelta, optional
        The time gap threshold for segmenting the data. If the time difference between consecutive rows exceeds this value,
        a new segment is started. Defaults to 1 minute.

    Returns
    -------
    None : the input DataFrame is modified in place, adding a 'segment' column to represent the segmented data.
    """
    df = df.copy()
    if 'segment' in df.columns:
        df = df.drop(columns=['segment']) # if one already exists

    # Calculate the time differences between consecutive rows
    time_diffs = df.index.diff()

    # Create the 'segment' column in a single operation
    df.insert(0, 'segment', (time_diffs > delta).cumsum())

    return df



def datetime_to_decimal_year_vectorised(date_index):
    day_of_year = date_index.day_of_year

    # Calculate the number of days in the year (consider leap years)
    is_leap_year = date_index.is_leap_year
    days_in_year = np.where(is_leap_year, 366, 365)

    # Calculate the decimal year
    return date_index.year + (day_of_year - 1) / days_in_year

def calculate_bins(data, bin_width=None):
    """
    Calculate the number of bins such that:
    - Each bin corresponds to an integer range and is aligned at 0.
    - The number of bins is iteratively doubled until it exceeds 20.
    - The final number of bins is restricted to [40, 100].

    Parameters:
    - data (pd.Series or pd.DataFrame): Input data. If a DataFrame, a single column is expected.

    Returns:
    - int: The calculated number of bins.
    """
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError('DataFrame must contain exactly one column.')
        data = data.iloc[:, 0]
    elif isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError('NumPy array must be one-dimensional.')
    elif not isinstance(data, (pd.Series, np.ndarray)):
        raise TypeError('Input must be a Pandas Series, single-column DataFrame, or one-dimensional NumPy array.')
    if bin_width is None:
        # Calculate the range of the data
        data_min, data_max = np.min(data), np.max(data)
        bin_range = np.ceil(data_max) - np.floor(data_min)  # Ensure range covers full integers
        n_bins = int(bin_range)  # Initial bins aligned with integers

        # Ensure n_bins > 20 by doubling if necessary
        while n_bins < 20:
            n_bins *= 2

        # Restrict n_bins to the range [40, 100]
        return max(40, min(n_bins, 100))
    else:
        return math.ceil((np.max(data)-np.min(data)+1)/bin_width)


def calculate_bins_edges(data, bin_width=1):
    """
    Similar to a above but returns bin edges
    """
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError('DataFrame must contain exactly one column.')
        data = data.iloc[:, 0]

    elif isinstance(data, np.ndarray):
        if data.ndim != 1:
            raise ValueError('NumPy array must be one-dimensional.')

    elif not isinstance(data, (pd.Series, np.ndarray)):
        raise TypeError('Input must be a Pandas Series, single-column DataFrame, or one-dimensional NumPy array.')

    data_min, data_max = np.min(data), np.max(data)

    return np.arange(np.floor(data_min),np.ceil(data_max)+bin_width,bin_width)




