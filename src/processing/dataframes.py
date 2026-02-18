import numpy as np
import pandas as pd
import warnings
from uncertainties import unumpy as unp

from .utils import add_unit, cdf_epoch_to_datetime
from ..analysing.calculations import calc_mean_error, calc_average_vector, average_of_averages

SKIP_COLUMNS = ('quality', 'mode', 'B_flag', 'flag', 'quality_esa', 'quality_fgm', 'sc_sw')

def add_df_units(df):
    unit_attrs = {}
    for column in df:
        unit_attrs[column] = add_unit(column)
    df.attrs['units'] = unit_attrs

def merge_dataframes(*dfs, suffices=None, clean=True, print_info=False):

    if len(dfs) < 2:
        raise ValueError('At least two dataframes are required.')

    if suffices is None:
        suffices = [None] * len(dfs)

    if len(suffices) != len(dfs):
        raise ValueError('suffixes length must match number of dataframes.')

    if all(s is None for s in suffices):
        warnings.warn('All suffices are None; column collisions may occur.')

    relabelled = [relabel_columns(df, suf) for df, suf in zip(dfs, suffices)]

    if print_info:
        for i, df in enumerate(relabelled, 1):
            print(f'Length of df{i}: {len(df):,}')

    merged = relabelled[0]
    for df in relabelled[1:]:
        merged = merged.merge(df, left_index=True, right_index=True)

    merged.attrs = {}

    for df in relabelled:
        for k, v in df.attrs.items():
            if k != 'units':
                merged.attrs[k] = v

    merged.attrs['units'] = {}
    for df in relabelled:
        merged.attrs['units'].update(df.attrs.get('units', {}))

    if clean:
        merged = merged.dropna(how='all')

    if print_info:
        print(f'Length of merged df: {len(merged):,}\n')

    return merged

def rename_columns(df, column_map):

    rename_map = {col: column_map[key] + col[len(key):] for col in df.columns for key in column_map if col.startswith(key)}
    df.rename(columns=rename_map, inplace=True)

    if df.attrs.get('units',{}):
        df.attrs['units'] = {rename_map.get(col, col): unit for col, unit in df.attrs['units'].items()}

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

def resample_data(df, time_col='epoch', sample_interval='1min', inc_info=True, columns_to_skip=SKIP_COLUMNS, drop_nans=True):

    if sample_interval in ('none','NONE'):
        print('No valid sampling interval provided.')
        return df

    print(f'Reprocessing to {sample_interval} resolution...\n')

    attributes = df.attrs

    df = df.copy()
    df.attrs = attributes
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

        if column in ('utc',time_col) or column in columns_to_skip or column.endswith('_unc') or column.endswith('_count'):

            print(f'Skipping {column}.')
            # columns that are meaningless to average, e.g. quality, mode
            continue

        elif df[column].count()==0:

            # Handles problems with missing data
            if drop_nans:
                print(f'Skipping {column} (all NaN).')
                continue
            print(f'Processing {column} (all NaN).')
            aggregated_columns[column] = np.nan
            aggregated_columns[f'{column}_unc']   = np.nan
            aggregated_columns[f'{column}_count'] = 0

        elif '_GS' in column:

            print(f'Processing {column} (vector).')

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

            print(f'Processing {column}.')

            # Use standard mean for other columns
            unit = df.attrs.get('units',{}).get(column,None)
            ufloat_series = grouped[column].apply(lambda x: calc_mean_error(x.dropna(), unit=unit))

            aggregated_columns[column] = unp.nominal_values(ufloat_series.to_numpy())
            if inc_info:
                aggregated_columns[f'{column}_unc']   = unp.std_devs(ufloat_series.to_numpy())
                aggregated_columns[f'{column}_count'] = non_nan_counts[column].to_numpy()

    resampled_df = pd.DataFrame(aggregated_columns, index=grouped.groups.keys())
    resampled_df.index.name = 'epoch'
    resampled_df.rename_axis('epoch', inplace=True)

    if time_col != 'index':
        resampled_df.reset_index(inplace=True)
    resampled_df.dropna(how='all',inplace=True)

    if 'utc' in resampled_df:
        resampled_df.drop(columns=['utc'], inplace=True)

    print(f'{sample_interval} done.')

    return resampled_df


def resample_data_weighted(df, time_col='epoch', sample_interval='1min', columns_to_skip=SKIP_COLUMNS):

    if sample_interval in ('none','NONE'):
        print('No valid sampling interval provided.')
        return df

    print(f'Reprocessing {len(df)} overlapping timestamps.\n')

    attributes = df.attrs

    df = df.copy()
    df.attrs = attributes
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

    for column in df.columns:

        if column in ('sc_sw','sc_msh'):
            print(f'{column} combined.')
            aggregated_columns[column] = grouped[column].apply(lambda x: x.iloc[0] if x.nunique() == 1 else ','.join(x.unique()))
            continue

        elif column in ('utc', time_col) or column in columns_to_skip:
            print(f'Skipping {column}.')
            continue

        elif column in aggregated_columns:
            print(f'    {column} processed.')
            continue

        ###----------Circular Averaging----------###
        if '_GS' in column:

            print(f'Processing {column} (vector).')

            field, _, coords = column.split('_')

            vector_columns = [f'{field}_{c}_{coords}' for c in ('x', 'y', 'z')]
            unc_columns = [f'{field}_{c}_{coords}_unc' for c in ('x', 'y', 'z')]
            count_column = f'{field}_{coords}_count'

            skip_x = False
            if vector_columns[0] not in df.columns:
                skip_x = True
                vector_columns[0] = vector_columns[0].replace('_GSM', '_GSE')
                unc_columns[0] = unc_columns[0].replace('_GSM', '_GSE')

            ufloat_series = grouped.apply(
                lambda g: average_of_averages(
                    g[vector_columns],
                    series_uncs=g[unc_columns],
                    series_counts=(g[count_column])
                ),
                include_groups=False
            )

            nom_vals = np.array(ufloat_series.apply(safe_nominals).to_list())
            std_vals = np.array(ufloat_series.apply(safe_stddevs).to_list())

            for i, comp in enumerate(('x', 'y', 'z')):
                if comp == 'x' and skip_x:
                    continue

                aggregated_columns[f'{field}_{comp}_{coords}'] = nom_vals[:, i]
                aggregated_columns[f'{field}_{comp}_{coords}_unc'] = std_vals[:, i]

            aggregated_columns[f'{field}_{coords}_count'] = (
                grouped[count_column].sum().to_numpy()
                if count_column in df.columns
                else grouped.size().to_numpy()
            )

        else:

            ###----------Scalar Averaging----------###
            print(f'Processing {column}.')

            if column == 'sc_sw':
                aggregated_columns[column] = grouped[column].first()
                continue

            unc_column = f'{column}_unc'
            count_column = f'{column}_count'

            if column == 'r_mag':
                count_column = 'r_GSE_count'

            ufloat_series = grouped.apply(
                lambda g: average_of_averages(
                    g[column],
                    series_uncs=(g[unc_column] if unc_column in g else None),
                    series_counts=(g[count_column] if count_column in g else None)
                ),
                include_groups=False
            ).to_numpy()

            aggregated_columns[column]     = unp.nominal_values(ufloat_series)
            aggregated_columns[unc_column] = unp.std_devs(ufloat_series)

            if count_column in grouped:
                aggregated_columns[count_column] = grouped[count_column].sum().to_numpy()
            else:
                aggregated_columns[count_column] = grouped.size().to_numpy()


    resampled_df = pd.DataFrame(aggregated_columns, index=grouped.groups.keys())
    resampled_df.rename_axis('epoch', inplace=True)
    resampled_df.index.name = 'epoch'

    if time_col != 'index':
        resampled_df.reset_index(inplace=True)
    resampled_df.dropna(how='all',inplace=True)

    if 'utc' in resampled_df:
        resampled_df.drop(columns=['utc'], inplace=True)

    print('Weighted average done.')

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



def print_head(df, nrows=5):

    with pd.option_context('display.max_columns', None,'display.width', None,'display.max_colwidth', None,'display.max_rows', None):
        print(df.head(nrows))