# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:08:31 2025

@author: richarj2
"""
import os
import numpy as np
import pandas as pd

from datetime import datetime
from spacepy import pycdf


from .dataframes import set_df_indices
from .utils import datetime_to_cdf_epoch, add_unit
from .handling import get_processed_files
from .reading import read_spacepy_object


def write_to_cdf(df, output_file, attributes=None, overwrite=True, append_rows=False):
    """
    Adds columns from a DataFrame to an existing CDF file, creating new variables or extending existing ones.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing data to be added to the CDF file. Each column in the DataFrame is added
        as a new variable or appended to an existing variable in the CDF file.

    output_file : pycdf.CDF
        Open CDF file to which the data will be added. New variables are created if they do not already
        exist in the file, or existing variables are extended with the new data.

    overwrite : bool, optional
        If True, overwrites existing columns; if False, appends to the columns. Default is False.

    Returns
    -------
        None: procedure just adds to attributes.
    """
    # Check if all columns have the same length
    column_lengths = df.apply(len)
    if column_lengths.nunique() > 1:
        raise ValueError(f'Columns have different lengths: {column_lengths.to_dict()}')

    with pycdf.CDF(output_file, create=not os.path.exists(output_file)) as cdf:
        cdf.readonly(False)
        if attributes is not None:
            for key, value in attributes.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        cdf.attrs[f'{key}_{sub_key}'] = sub_value
                else:
                    # If it's not a dictionary, just assign the value
                    cdf.attrs[key] = value
        # Write each column in the DataFrame to the CDF file
        for column in df.columns:
            new_data = df[column].to_numpy()
            if column not in cdf:
                try:
                    if column == 'epoch':
                        new_data = datetime_to_cdf_epoch(new_data)
                        cdf.new('epoch', data=new_data, type=pycdf.const.CDF_EPOCH)

                    elif isinstance(new_data[0], str):  # Check if data is a string (unicode or bytes)
                        max_len = max(len(s) for s in new_data)
                        padded_data = np.array([s.ljust(max_len) for s in new_data], dtype=f'U{max_len}')
                        # Store the data as CDF_CHAR with variable length
                        cdf.new(column, data=padded_data, type=pycdf.const.CDF_CHAR)
                        df.attrs['units'][column] = 'STRING'

                    else:
                        cdf.new(column, data=new_data)
                except:
                    print(f'Cannot add {column} to cdf file.')
                else:
                    cdf[column].attrs['units'] = df.attrs['units'].get(column,add_unit(column))
            elif overwrite:
                cdf[column] = new_data

            elif append_rows:
                # If the column already exists, extend the existing data
                cdf[column] = np.concatenate((cdf[column][...], new_data))
    print('Data written to file.\n')



def update_processed_cdf(directory, func, overwrite=False, year=None, batch=False, **kwargs):
    """
    Updates existing CDF files by applying a specified function to the data. The function can be applied
    to the entire dataset or in monthly batches, and the results are written back to the CDF file.

    Parameters
    ----------
    directory : str
        The path to the directory containing the CDF files to be updated.

    func : callable
        A function that processes the data in the DataFrame. The function should take the DataFrame
        and additional keyword arguments and return a modified DataFrame.

    overwrite : bool, optional
        If True, overwrites existing columns in the CDF file; otherwise, appends to them. Default is False.

    year : int, optional
        The year of the files to be processed. If None, processes all files in the directory.

    batch : bool, optional
        If True, processes the data in monthly batches; otherwise, processes the entire dataset at once.
        Default is True.

    **kwargs : additional keyword arguments
        Additional arguments to be passed to the function `func`.

    Returns
    -------
        None : updates the CDF files in place.
    """
    cdf_files = get_processed_files(directory, year)

    # Use glob to find files matching the pattern
    for cdf_file in cdf_files:

        with pycdf.CDF(cdf_file, create=False) as cdf:
            data_df = read_spacepy_object(cdf)

        set_df_indices(data_df, 'epoch')

        if data_df.empty:
            print(f"No data found in {cdf_file}.")

        else:
            all_batches = []

            if batch:
                # Resample by month and process each month's data
                for (year, month), group in data_df.groupby([data_df.index.year, data_df.index.month]):
                    print(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]} - Processing {year:04d}-{month:02d}")

                    all_batches.append(func(group, **kwargs))

                # Combine all monthly batches into a single DataFrame
                new_df = pd.concat(all_batches, axis=0)
            else:
                # Process the entire DataFrame at once
                new_df = func(data_df, **kwargs)

            write_to_cdf(new_df, cdf_file, overwrite)
            print(f"{datetime.now().strftime('%H:%M:%S.%f')[:-3]} - Finished processing file")


def update_cdf_attributes(directory, new_values, add=True):
    """
    Updates or adds a specific attribute in CDF files within a directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing the CDF files.

    attribute_key : str
        The key of the attribute to update or add (e.g., 'data_frequency').

    new_value : Any
        The new value to set for the specified attribute key.

    add : bool, optional
        If True, adds the attribute if it does not exist. Default is False.

    Returns
    -------
    None
    """
    # Iterate through all files in the directory
    cdf_files = get_processed_files(directory)

    # Use glob to find files matching the pattern
    for cdf_file in cdf_files:
            try:
                # Open the CDF file
                with pycdf.CDF(cdf_file, create=False) as cdf:
                    cdf.readonly(False)
                    # Check if the attribute exists
                    for key, value in new_values.items():
                        if key in cdf.attrs or add:
                            cdf.attrs[key] = value

            except Exception as e:
                print(f'Error processing {cdf_file}: {e}')

    print('Attribute update complete.')