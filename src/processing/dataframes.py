import numpy as np
import pandas as pd
import warnings

from uncertainties import unumpy as unp, ufloat

from .utils import add_unit, cdf_epoch_to_datetime

from ..coordinates.spatial import cartesian_to_spherical, spherical_to_cartesian
from ..analysing.calculations import average_of_averages

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

    """
    Renames columns and the units in the attributes
    Works inplace
    """

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

# %% resampling

def resample_data(df, time_col='epoch', sample_interval='1min', inc_info=True, columns_to_skip=SKIP_COLUMNS):

    ### Consider changing this so x_GSM is always written as x_GSE


    if sample_interval in ('none','NONE'):
        print('No valid sampling interval provided.')
        return df

    print(f'Reprocessing to {sample_interval} resolution...\n')

    units = df.attrs.get('units', {})

    if time_col == 'index':
        df['utc'] = df.index.floor(sample_interval)
    else:
        if time_col != 'epoch':
            df.rename(columns={time_col: 'epoch'},inplace=True)
        df['utc'] = df[time_col].dt.floor(sample_interval)

    columns_to_drop = ['utc']

    if df['utc'].is_unique:
        df.drop(columns=columns_to_drop, inplace=True)
        return

    #---Sorting columns---#

    column_counts = df.count()

    scalar_columns  = []
    angular_columns = []
    vector_groups   = {}

    for column in df.columns:

        if column in ('utc',time_col) or column in columns_to_skip or column.endswith(('_unc','_count')):
            # columns that are meaningless to average, e.g. quality, mode
            print(f'Skipping {column}.')
            continue

        elif column_counts[column]==0:
            print(f'Skipping {column} (all NaN).')
            continue

        elif '_GS' in column:
            field, comp, coords = column.split('_')
            vector_groups.setdefault((field, coords), {})[comp] = column # if (field,coords) is not a key, it adds it with the default value {}

        else:
            unit = units.get(column)

            if unit in ('rad', 'deg', '°'):
                angular_columns.append(column)

            else:
                scalar_columns.append(column)

    #---Additions before grouping---#

    for column in angular_columns:

        values = df[column]

        if unit in ('deg', '°'):
            values = np.radians(values)

        df[f'{column}__sin'] = np.sin(values)
        df[f'{column}__cos'] = np.cos(values)

        columns_to_drop.extend([f'{column}__sin',f'{column}__cos'])

    skip_vectors = []

    for (field, coords), components in vector_groups.items():

        if 'x' not in components:
            if coords == 'GSM' and (field, 'GSE') in vector_groups:
                components['x'] = vector_groups[(field, 'GSE')]['x']
            else:
                skip_vectors.append((field, coords))
                continue

        vector_columns = [components['x'], components['y'], components['z']]

        vec = df[vector_columns].to_numpy()

        mag_col = f'{field}_mag'
        if mag_col not in df.columns:
            df[mag_col] = np.linalg.norm(vec, axis=1)
            scalar_columns.append(mag_col)

        valid = np.isfinite(vec).all(axis=1)

        r_col = f'{field}_{coords}__r'

        theta_sin_col = f'{field}_{coords}__theta_sin'
        theta_cos_col = f'{field}_{coords}__theta_cos'

        phi_sin_col = f'{field}_{coords}__phi_sin'
        phi_cos_col = f'{field}_{coords}__phi_cos'

        temp_columns = [r_col, theta_sin_col, theta_cos_col, phi_sin_col, phi_cos_col]

        columns_to_drop.extend(temp_columns)

        for column in temp_columns:
            df[column] = np.nan

        if np.any(valid):

            r, theta, phi = cartesian_to_spherical(vec[valid, 0], vec[valid, 1], vec[valid, 2])

            df.loc[valid, r_col] = r

            df.loc[valid, theta_sin_col] = np.sin(theta)
            df.loc[valid, theta_cos_col] = np.cos(theta)

            df.loc[valid, phi_sin_col] = np.sin(phi)
            df.loc[valid, phi_cos_col] = np.cos(phi)

    #---Grouping---#

    grouped = df.groupby('utc')

    aggregated_columns = {}

    #---Simple scalar columns---#

    if len(scalar_columns)>0:

        print('Processing scalar columns:',scalar_columns)

        means, sems, counts = grouped_mean(grouped[scalar_columns])

        for column in scalar_columns:

            aggregated_columns[column] = means[column].to_numpy()

            if inc_info:
                aggregated_columns[f'{column}_unc']   = sems[column].to_numpy()
                aggregated_columns[f'{column}_count'] = counts[column].to_numpy()

                units[f'{column}_unc']   = units.get(column)
                units[f'{column}_count'] = 'NUM'

    #---Angular columns---#

    if len(angular_columns)>0:

        print('Processing angular columns:',angular_columns)

        sin_cols = [f'{col}__sin' for col in angular_columns]
        cos_cols = [f'{col}__cos' for col in angular_columns]

        sin_mean = grouped[sin_cols].mean()
        cos_mean = grouped[cos_cols].mean()
        counts = grouped[angular_columns].count()

        for column in angular_columns:

            sin_col = f'{column}__sin'
            cos_col = f'{column}__cos'

            R = np.sqrt(sin_mean[sin_col]**2 + cos_mean[cos_col]**2)

            mean = np.arctan2(sin_mean[sin_col],cos_mean[cos_col])

            std = np.sqrt(-2 * np.log(R))
            sem = std / np.sqrt(counts[column])

            sem = sem.mask(counts[column] <= 1, 0)

            if units.get(column) in ('deg', '°'):
                mean = np.degrees(mean)
                sem  = np.degrees(sem)

            aggregated_columns[column] = mean.to_numpy()

            if inc_info:
                aggregated_columns[f'{column}_unc'] = sem.to_numpy()
                aggregated_columns[f'{column}_count'] = counts[column].to_numpy()

    #---Vector columns---#


    for (field, coords), components in vector_groups.items():

        if (field,coords) in skip_vectors:
            continue

        print(f'Processing {field}_{coords} (vector).')

        r_col = f'{field}_{coords}__r'

        theta_sin_col = f'{field}_{coords}__theta_sin'
        theta_cos_col = f'{field}_{coords}__theta_cos'

        phi_sin_col = f'{field}_{coords}__phi_sin'
        phi_cos_col = f'{field}_{coords}__phi_cos'

        r_mean, r_sem, r_count = grouped_mean(grouped[r_col])

        theta_mean, theta_sem, theta_count = grouped_circular_mean(grouped[theta_sin_col], grouped[theta_cos_col])

        phi_mean, phi_sem, phi_count = grouped_circular_mean(grouped[phi_sin_col], grouped[phi_cos_col])

        # Convert spherical averages back to Cartesian
        valid = (np.isfinite(r_mean) & np.isfinite(theta_mean) & np.isfinite(phi_mean))

        # Initialise outputs as NaN
        x = np.full(len(r_mean), np.nan)
        y = np.full(len(r_mean), np.nan)
        z = np.full(len(r_mean), np.nan)

        if np.any(valid):
            r_u = make_ufloat_array(r_mean[valid], r_sem[valid])
            theta_u = make_ufloat_array(theta_mean[valid], theta_sem[valid])
            phi_u = make_ufloat_array(phi_mean[valid], phi_sem[valid])

            x_u, y_u, z_u = spherical_to_cartesian(r_u, theta_u, phi_u)

            x[valid] = unp.nominal_values(x_u)
            y[valid] = unp.nominal_values(y_u)
            z[valid] = unp.nominal_values(z_u)

        vector_results = {'x': x, 'y': y, 'z': z}

        for comp, values in vector_results.items():

            if all((comp=='x', coords=='GSM', vector_groups.get((field,'GSE'),{}).get('x') in aggregated_columns)):
                print(f'Not adding {field}_{comp}_{coords}.')
                continue # skips x_GSM if x_GSE already in the new df

            column = components[comp]
            aggregated_columns[column] = unp.nominal_values(values)

            if inc_info:
                aggregated_columns[f'{column}_unc'] = unp.std_devs(values)
                units[f'{column}_unc'] = units.get(column)

        if inc_info:
            vector_columns = [components['x'], components['y'], components['z']]

            aggregated_columns[f'{field}_{coords}_count'] = (grouped[vector_columns].count().min(axis=1).to_numpy())

            units[f'{field}_{coords}_count'] = 'NUM'

    #---Combining the data---#

    print('Combining...')

    resampled_df = pd.DataFrame(aggregated_columns, index=grouped.groups.keys())
    resampled_df.index.name = 'epoch'
    resampled_df.rename_axis('epoch', inplace=True)

    if time_col != 'index':
        resampled_df.reset_index(inplace=True)
    resampled_df.dropna(how='all',inplace=True)

    resampled_df.attrs = df.attrs
    resampled_df.attrs['units'] = units
    resampled_df.attrs['sample_interval'] = sample_interval

    print(f'{sample_interval} done.')
    df.drop(columns=columns_to_drop,inplace=True)

    return resampled_df

# %% helpers

def make_ufloat_array(values, errors):
    """
    Convert value/error arrays into an array of ufloats.
    """

    return np.array([ufloat(v, e) for v, e in zip(values, errors)], dtype=object)

def grouped_mean(grouped_series):
    """
    Calculate mean, SEM, and count ignoring NaNs.
    Matches calc_mean_error() for non-angular data.
    """

    count = grouped_series.count()
    mean = grouped_series.mean()

    std = grouped_series.std(ddof=1)

    sem = (std.divide(np.sqrt(count)).mask(count <= 1, 0))

    return mean, sem, count

def grouped_circular_mean(grouped_sin, grouped_cos):
    """
    Circular mean and SEM from grouped sin/cos components.
    """

    sin_mean = grouped_sin.mean()
    cos_mean = grouped_cos.mean()

    mean = np.arctan2(sin_mean, cos_mean)

    R = np.sqrt(sin_mean**2 + cos_mean**2)

    std = np.sqrt(-2 * np.log(R))

    count = grouped_sin.count()

    sem = (std.divide(np.sqrt(count)).mask(count <= 1, 0))

    return mean, sem, count
# %% weighted
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

# %% utils

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

def print_dataframe(df, nrows=5, location='head'):

    with pd.option_context('display.max_columns', None,'display.width', None,'display.max_colwidth', None,'display.max_rows', None):
        if location in ('head','both'):
            print(df.head(nrows))
        if location in ('tail','both'):
            print(df.tail(nrows))