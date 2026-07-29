# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:18:16 2025

@author: richarj2
"""
import numpy as np
import pandas as pd
import datetime as dt

# %% time

def filter_select_day(df, day):
    """
    Filters dataframe to return data from a certain day.
    """
    date = pd.Timestamp(day).normalize()  # midnight of that day
    return df.loc[date : date + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)]

def exclude_days(df, bad_data):
    """
    Excludes rows in a DataFrame where the index falls on any of the specified bad days.

    bad_data :
        A list or tuple specifying the dates to be excluded. Can contain:
        - Single datetime values for exact dates to exclude.
        - Tuples specifying date ranges or conditions for exclusion.
            - A tuple of length 2: (start_date, end_date) for a date range.
            - A tuple of length 3: (day, df_column, condition_value) to exclude a specific day based on a condition.
            - A tuple of length 4: (start_date, end_date, df_column, condition_value) to exclude rows based on a condition within a date range.

    """
    df.index = pd.to_datetime(df.index, errors='coerce')
    mask = pd.Series(True, index=df.index)

    for bad in bad_data:
        if isinstance(bad, (pd.Timestamp, dt.datetime, dt.date)):  # Single date
            bad_day = pd.Timestamp(bad)  # Convert to pd.Timestamp for consistency
            mask &= ~(df.index.normalize() == bad_day)

        elif isinstance(bad, tuple):
            if len(bad) == 2:  # Date range
                start_date, end_date = map(pd.Timestamp, bad)  # Convert to pd.Timestamp
                mask &= ~((df.index >= start_date) & (df.index <= end_date))

            elif len(bad) == 3:  # Single day with condition
                day, df_col, cond_val = bad
                day = pd.Timestamp(day)  # Convert to pd.Timestamp
                mask &= ~((df.index.normalize() == day) & (df[df_col] == cond_val))

            elif len(bad) == 4:  # Date range with condition
                start_date, end_date, df_col, cond_val = bad
                start_date, end_date = map(pd.Timestamp, (start_date, end_date))  # Convert
                mask &= ~((df.index >= start_date) & (df.index <= end_date) & (df[df_col] == cond_val))

            else:
                raise ValueError(f'Invalid tuple length in bad_data: {bad}')

        else:
            raise ValueError(f'Invalid entry in bad_data: {bad}')

    df.drop(df.loc[~mask].index, inplace=True)

# %% value

def filter_sign(df, col, sign='Positive'):
    if sign == 'Positive':
        return df[df[col]>0]
    elif sign == 'Negative':
        return df[df[col]<0]
    elif sign == 'Both':
        return df
    else:
        raise ValueError(f'{sign} not a valid filtering range.')

def filter_percentile(df, y_col, bins, p, filtering='Above'):
    """
    Filters a DataFrame to include only rows where the `y_col` values are above or below
    a specified percentile for each bin.
    """
    # Group the DataFrame by bins
    grouped = df.groupby(bins, observed=True)

    # Store percentile cutoffs for each bin
    bin_percentiles = {}

    # Apply the filtering to each bin
    def filter_group(group):
        cutoff = np.percentile(group[y_col], p)
        bin_percentiles[group.name] = cutoff  # Store the percentile for this bin
        if filtering == 'Below':
            return group[group[y_col] <= cutoff]
        elif filtering == 'Above':
            return group[group[y_col] > cutoff]
        else:
            raise ValueError(f"{filtering} is not a valid filtering range.")

    # Apply filtering to all groups and combine results
    filt_df = grouped.apply(filter_group).reset_index(drop=True)

    # Attach percentile cutoffs as an attribute
    filt_df.attrs['bin_limits'] = bin_percentiles

    return filt_df

# %% data

def filter_sw(df, method, **kwargs):
    """
    Filters a DataFrame based on spacecraft proximity to the bow shock and optional nose region.

    """
    df = df.copy()
    bad_data = kwargs.get('bad_data',None)
    return_mask = kwargs.get('return_mask',None)
    if bad_data is not None:
        exclude_days(df, bad_data)
    if method == 'model':
        # Apply mask based on the distance to the bow shock
        buffer = kwargs.get('buffer',0)
        df.attrs['global']['buffer'] = buffer
        mask = df['r_bs_diff'] > buffer

        # Optionally apply additional filtering based on the spacecraft being within the nose region
        if kwargs.get('nose',False):
            mask &= ((abs(df['r_y_GSE']) < 5) &
                     (abs(df['r_z_GSE']) < 5))

    elif method == 'GRMB':
        regions = kwargs.get('regions',(12,)) # 12 is the SWF region

        mask = df['GRMB_region'].isin(regions)

    elif method == 'combined':
        buffer = kwargs.get('buffer',0)
        df.attrs['global']['buffer'] = buffer
        buffer_mask = df['r_bs_diff'] > buffer

        regions = kwargs.get('regions',(12,))
        region_mask = df['GRMB_region'].isin(regions)

        mask = region_mask | buffer_mask

    else:
        raise ValueError('Please pass in a valid method.')

    if df.loc[mask].empty:
        raise RuntimeError('Filtered dataframe is empty.')
    print(f'Minutes before filtering: {len(df):,}')
    print(f'Minutes after filtering:  {np.sum(mask):,}, {100*np.sum(mask)/len(df):.3g}%')
    if return_mask:
        return mask
    return df.loc[mask]

def filter_data(df, *args):
    """
    Filters the DataFrame based on a specified column and value range.

    """
    column, min_val, max_val = args
    # Create a mask based on the filtering conditions
    if column == 'index':
        mask = (df.index >= min_val) & (df.index < max_val)
    else:
        mask = (df[column] >= min_val) & (df[column] <= max_val)

    # Drop rows that do not meet the mask condition
    df.drop(df.loc[~mask].index, inplace=True)


def filter_by_spacecraft(df, sc_col, sc_id, include=True):
    """
    Filters the DataFrame to include or exclude rows based on spacecraft ID.

    """
    if sc_col not in df:
        raise ValueError(f'"{sc_col}" not in dataframe.')

    # Create a mask based on the filtering condition
    if include:
        mask = df[sc_col] == sc_id
    else:
        mask = df[sc_col] != sc_id

    # Drop rows that do not meet the mask condition
    df.drop(df.loc[~mask].index, inplace=True)




