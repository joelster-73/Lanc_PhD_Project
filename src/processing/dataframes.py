import numpy as np
import pandas as pd
from uncertainties import unumpy as unp

from .utils import add_unit, cdf_epoch_to_datetime
from ..analysing.calculations import calc_mean_error, calc_average_vector

def add_df_units(df):
    unit_attrs = {}
    for column in df:
        unit_attrs[column] = add_unit(column)
    df.attrs['units'] = unit_attrs

def merge_dataframes(df1, df2, suffix_1=None, suffix_2=None, clean=True, print_info=False):

    if print_info:
        print(f'Length of df1: {len(df1):,}')
        print(f'Length of df2: {len(df2):,}')

    # Relabel columns with specified suffixes
    if suffix_1 is None and suffix_2 is None:
        print ('Both new suffices are "None"; duplicate columns will introduce errors.')
    new_df1 = relabel_columns(df1, suffix_1)
    new_df2 = relabel_columns(df2, suffix_2)

    # Merge DataFrames based on their indices
    merged = new_df1.merge(new_df2, left_index=True, right_index=True)

    # Combine units attributes from both DataFrames
    merged.attrs = {}

    # merge non-unit attrs
    merged.attrs.update({k: v for k, v in new_df1.attrs.items() if k != 'units'})
    merged.attrs.update({k: v for k, v in new_df2.attrs.items() if k != 'units'})

    # merge units explicitly
    merged.attrs['units'] = {}
    merged.attrs['units'].update(new_df1.attrs.get('units', {}))
    merged.attrs['units'].update(new_df2.attrs.get('units', {}))

    if clean:
        cleaned = merged.dropna(how='all')
        if print_info:
            print(f'Length of merged df: {len(cleaned):,}\n')
        return cleaned
    if print_info:
        print(f'Length of merged df: {len(merged):,}\n')
    return merged


def relabel_columns(df, label):

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

def safe_nominals(arr):
    if len(arr) == 0:  # empty list
        return [np.nan, np.nan, np.nan]
    else:
        return unp.nominal_values(arr)

def safe_stddevs(arr):
    if len(arr) == 0:  # empty list
        return [np.nan, np.nan, np.nan]
    else:
        return unp.std_devs(arr)

def resample_data(df, time_col='epoch', sample_interval='1min', inc_info=True, columns_to_skip=('quality','mode','flag', 'quality_esa','quality_fgm')):


    if sample_interval in ('none','NONE'):
        print('No valid sampling interval provided.')
        return df

    df = df.copy()
    if time_col == 'index':
        df['utc'] = df.index.floor(sample_interval)
    else:
        df['utc'] = df[time_col].dt.floor(sample_interval)

    grouped = df.groupby('utc')

    if len(grouped) == len(df):
        df.drop(columns=['utc'],inplace=True)
        df.rename(columns={time_col: 'epoch'},inplace=True)
        return df

    aggregated_columns = {}
    non_nan_counts = grouped.count()

    for column in df.columns:
        print(column)

        if column in ('utc',time_col) or column in columns_to_skip:
            # columns that are meaningless to average, e.g. quality, mode
            continue

        elif len(df[column].dropna())==0:
            # Handles problems with missing data
            continue

        elif '_GS' in column:

            field, _, coords = column.split('_')
            if column in aggregated_columns:
                # Skips y and z as done with x
                continue

            vector_columns = [f'{field}_{comp}_{coords}' for comp in ('x','y','z')]
            skip_x = False
            if vector_columns[0] not in df.columns: # when the x_GSM component isn't in the df, only x_GSE
                skip_x = True
                vector_columns[0] = vector_columns[0].replace('_GSM','_GSE')

            ufloat_series = grouped[vector_columns].apply(lambda x: calc_average_vector(x.dropna(), param=f'{field}_{coords}'))

            try:
                nom_vals = unp.nominal_values(ufloat_series.to_list())
                std_vals = unp.std_devs(ufloat_series.to_list())
            except:
                nom_vals = np.array(ufloat_series.apply(safe_nominals).to_list())
                std_vals = np.array(ufloat_series.apply(safe_stddevs).to_list())

            if len(nom_vals)==0:
                continue

            for i, comp in enumerate(('x','y','z')):
                if comp=='x' and skip_x:
                    continue

                aggregated_columns[f'{field}_{comp}_{coords}'] = nom_vals[:, i]
                if inc_info:
                    aggregated_columns[f'{field}_{comp}_{coords}_unc'] = std_vals[:, i]

            if inc_info:
                aggregated_columns[f'{field}_{coords}_count'] = non_nan_counts[vector_columns].min(axis=1)

        else:

            # Use standard mean for other columns
            unit = df.attrs['units'].get(column)
            ufloat_series = grouped[column].apply(lambda x: calc_mean_error(x.dropna(), unit=unit))

            aggregated_columns[column] = unp.nominal_values(ufloat_series.to_numpy())
            if inc_info:
                aggregated_columns[f'{column}_unc']   = unp.std_devs(ufloat_series.to_numpy())
                aggregated_columns[f'{column}_count'] = non_nan_counts[column].to_numpy()


    resampled_df = pd.DataFrame(aggregated_columns, index=grouped.groups.keys())
    resampled_df.rename_axis('epoch', inplace=True)

    if time_col != 'index':
        resampled_df.reset_index(inplace=True)
    resampled_df.dropna(how='all',inplace=True)

    if 'utc' in resampled_df:
        resampled_df.drop(columns=['utc'], inplace=True)

    return resampled_df

def set_df_indices(df, time_col):

    # Converts epoch time to datetimes
    if time_col not in df:
        raise Exception(f'"{time_col}" not in df.')
    try:
        df[time_col] = cdf_epoch_to_datetime(df[time_col])
    except:
        print('Data already in datetime format')

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

