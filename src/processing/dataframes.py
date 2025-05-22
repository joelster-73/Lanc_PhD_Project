import numpy as np
import pandas as pd

from .utils import add_unit, cdf_epoch_to_datetime
from ..analysing.calculations import circular_mean

def add_df_units(df):
    unit_attrs = {}
    for column in df:
        unit_attrs[column] = add_unit(column)
    df.attrs['units'] = unit_attrs

def merge_dataframes(df1, df2, suffix_1=None, suffix_2=None, clean=True, print_info=False):
    """
    Merges two DataFrames based on their indices and relabels their columns with specified suffixes.

    Parameters
    ----------
    df1 : pandas.DataFrame
        The first DataFrame to be merged.
    df2 : pandas.DataFrame
        The second DataFrame to be merged.
    suffix_1 : str
        The suffix to be added to the column names of the first DataFrame (`df1`).
    suffix_2 : str
        The suffix to be added to the column names of the second DataFrame (`df2`).

    Returns
    -------
    pandas.DataFrame
        A merged DataFrame with the columns from both input DataFrames,
        with the specified suffixes appended to their names.
    """
    if print_info:
        print(f'Length of df1: {len(df1):,}')
        print(f'Length of df2: {len(df2):,}')
    if suffix_1 is None and suffix_2 is None:
        raise ValueError('Both new suffices cannot be "None".')
    # Relabel columns with specified suffixes
    new_df1 = relabel_columns(df1, suffix_1)
    new_df2 = relabel_columns(df2, suffix_2)

    # Merge DataFrames based on their indices
    merged = new_df1.merge(new_df2, left_index=True, right_index=True)

    # Combine units attributes from both DataFrames
    merged.attrs['units'] = new_df1.attrs['units']
    merged.attrs['units'].update(new_df2.attrs['units'])

    merged.attrs['global'] = new_df1.attrs['global']
    merged.attrs['global'].update(new_df2.attrs['global'])

    if clean:
        cleaned = merged.dropna()
        if print_info:
            print(f'Length of merged df: {len(cleaned):,}\n')
        return cleaned
    if print_info:
        print(f'Length of merged df: {len(merged):,}\n')
    return merged


def relabel_columns(df, label):
    """
    Relabels the columns of a DataFrame by appending a suffix to each column name.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose columns are to be relabeled.

    label : str
        The suffix to be appended to each column name.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the columns renamed by appending the specified label.
        The `attrs['units']` attribute is also updated to match the new column names.
    """
    # Generate new column names by appending the label
    if label is not None:
        old_cols = list(df.columns)
        new_cols = [f'{col}_{label}' for col in old_cols]

        # Rename columns in the DataFrame
        new_df = df.rename(columns=dict(zip(old_cols, new_cols)))

        # Update units attribute with new column names
        try:
            units = df.attrs['units']
        except:
            add_df_units(df)
            units = df.attrs['units']
        new_units = {f'{key}_{label}': value for key, value in units.items()}

        new_df.attrs['units'] = new_units

        return new_df
    return df


def replace_inf(df, replace_large=False, threshold=1e28):
    # Replace infinities with NaN, without using inplace=True
    df = df.replace([np.inf, -np.inf], np.nan)

    if replace_large:
        numeric_cols = df.select_dtypes(include=[np.number])

        mask = np.abs(numeric_cols) > threshold
        df.loc[mask.index, mask.columns] = numeric_cols.where(~mask)

    return df

def resample_data(df, time_col='epoch', sample_interval='1min', show_count=False, show_print=False):
    """
    Resamples time series data in a DataFrame to specified intervals, calculating the mean for each bin
    and removing rows with NaN values. The time for each bin is set to the beginning of the bin in CDF epoch format.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing time data (in CDF epoch format) and other data columns.

    time_col : str, optional
        Name of the column containing time data to be resampled. Defaults to 'epoch'.

    sample_interval : str, optional
        The sampling interval for resampling. Default is '1min' (1-minute intervals).

    Returns
    -------
    pandas.DataFrame
        A DataFrame resampled to the specified interval with mean values calculated for each bin.
        The time for each bin corresponds to the beginning of the bin (not the midpoint) and is in CDF epoch format.
    """
    df = df.copy()
    if time_col == 'index':
        df['utc'] = df.index.floor(sample_interval)
    else:
        df['utc'] = df[time_col].dt.floor(sample_interval)

    aggregated_columns = {}
    grouped = df.groupby('utc')

    if len(grouped) == len(df):
        df.drop(columns=['utc'],inplace=True)
        return df

    for column in df.columns:
        if column == 'utc':
            continue
        elif df.attrs['units'].get(column) == 'rad':
            # Use circular mean for 'rad' columns
            if show_print:
                print('Doing circular mean.')
            aggregated_columns[column] = grouped[column].apply(lambda x: circular_mean(x))
        else:
            # Use standard mean for other columns
            aggregated_columns[column] = grouped[column].mean()

    if show_count:
        aggregated_columns['count'] = grouped.size().astype(int)

    resampled_df = pd.DataFrame(aggregated_columns)

    if time_col != 'index':
        resampled_df.drop(columns=[time_col], inplace=True)
    resampled_df.rename_axis('epoch', inplace=True)
    if time_col != 'index':
        resampled_df.reset_index(inplace=True)
    resampled_df.dropna(inplace=True)
    if 'utc' in resampled_df:
        resampled_df.drop(columns=['utc'], inplace=True)

    return resampled_df



def set_df_indices(df, time_col):
    """
    Converts epoch time to datetime and sets the 'time' column as the index of the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data with a time column in CDF epoch format.

    time_col : str
        The name of the column containing the time data in CDF epoch format.

    Returns
    -------
        None : The function modifies the DataFrame in-place, setting 'time' as the index and removing the original time column.
    """
    # Converts epoch time to datetimes
    df[time_col] = cdf_epoch_to_datetime(df[time_col])
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col, inplace=True)  # Set 'time_col' as the index

def next_index(df, index):
    if index in df.index:
        idx_position = df.index.get_loc(index)
        if idx_position + 1 < len(df):
            return df.index[idx_position + 1]

    return None

def previous_index(df, index):
    if index in df.index:
        idx_position = df.index.get_loc(index)
        if idx_position > 0:
            return df.index[idx_position - 1]

    return None

