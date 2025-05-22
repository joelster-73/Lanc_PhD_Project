# -*- coding: utf-8 -*-
"""
Created on Mon May 12 12:02:37 2025

@author: richarj2
"""
import numpy as np
import pandas as pd
import speasy as spz
from speasy import amda
from datetime import timedelta

from .config import data_availability
from ..dataframes import resample_data


def retrieve_data(parameter, source, speasy_variables, start_time, end_time, downsample=True, print_bounds=False):

    available_start, available_end = data_availability.get(source)
    if start_time < available_start and end_time < available_start:
        if print_bounds:
            print(f'{source} outside available range: {available_start} to {available_end}')
        return None, None
    elif start_time > available_end and end_time > available_end:
        if print_bounds:
            print(f'{source} outside available range: {available_start} to {available_end}')
        return None, None

    data_id = speasy_variables.get(parameter).get(source)
    if data_id is None:
        print(f'No {parameter} data fpr {source}.')
        return pd.DataFrame()

    if isinstance(data_id, str):
        spz_data = amda.get_data(data_id, start_time, end_time)
        if spz_data is None:
            return None
        spz_data.replace_fillval_by_nan(inplace=True)

        times, values, unit = spz_data.time, spz_data.values, spz_data.unit
    elif isinstance(data_id, list):
        vec_data = spz.get_data(data_id, start_time, end_time)
        if spz_data is None:
            return None
        vec_data.replace_fillval_by_nan(inplace=True)

        times = vec_data[0].time
        values = np.column_stack((vec_data[0].values,vec_data[1].values,vec_data[2].values))
        unit = vec_data[0].unit

    else:
        spz_data = spz.get_data(data_id, start_time, end_time)
        if spz_data is None:
            return None
        spz_data.replace_fillval_by_nan(inplace=True)

        times, values, unit = spz_data.time, spz_data.values, spz_data.unit

    if source == 'IMP8':
        # seems to be problem with the fill value, metadata says 9999.9, but seems to be about double
        values[values>=9999.9] = np.nan

    times = pd.to_datetime(times, unit='ms')

    df = pd.DataFrame()

    if values.ndim == 1 or values.shape[1] == 1:
        values = values.flatten()
        df = pd.DataFrame(values, columns=[parameter], index=times)
        df.attrs['units'] = {}
        df.attrs['units'][parameter] = unit
    else:
        column_names = [f'{parameter}_x', f'{parameter}_y', f'{parameter}_z']
        df = pd.DataFrame(values, columns=column_names, index=times)
        df.attrs['units'] = {}
        for col in column_names:
            df.attrs['units'][col] = unit

    if source == 'OMNI':
        sc_ID = speasy_variables.get('OMNI_sc')
        id_data = spz.get_data(sc_ID, start_time, end_time)
        # spz_data.replace_fillval_by_nan(inplace=True)
        _, ids, _ = id_data.time, id_data.values, id_data.unit
        df['spacecraft'] = ids

    if downsample:
        attributes = df.attrs
        df = resample_data(df, 'index')
        df.attrs = attributes

    return df


def retrieve_datum(parameter, source, speasy_variables, time):

    if time is None:
        return None, None

    available_start, available_end = data_availability.get(source)
    if time < available_start or time > available_end:
        print(f'{source} outside available range: {available_start} to {available_end}')
        return None, None

    data_id = speasy_variables.get(parameter).get(source)
    window = 1 # minutes
    max_attempts = 60
    for counter in range(1, max_attempts + 1):
        interval = timedelta(minutes=window*counter)
        spz_data = amda.get_data(data_id, time-interval, time+interval)
        if spz_data:
            spz_data.replace_fillval_by_nan(inplace=True)

            times, values, unit = spz_data.time, spz_data.values, spz_data.unit
            times = pd.to_datetime(times, unit='ms')

            if values.ndim == 1 or values.shape[1] == 1:
                values = values.flatten()
                df = pd.DataFrame(values, columns=[parameter], index=times)
            else:
                column_names = [f'{parameter}_x', f'{parameter}_y', f'{parameter}_z']
                df = pd.DataFrame(values, columns=column_names, index=times)

            df.dropna(inplace=True)
            if not df.empty:
                if time in df.index:
                    datum = df.loc[time]
                    if isinstance(datum, pd.DataFrame):
                        datum = datum.iloc[0]
                    datum_values = datum.values
                    return datum_values.item() if datum_values.size == 1 else datum_values, unit
                elif len(df) > 1 and df.index.min() <= time <= df.index.max():
                    df = df.reindex(df.index.union([time])).sort_index()
                    df = df.interpolate(method='linear')
                    df = df.loc[~df.index.duplicated(keep='first')]
                    if time in df.index:
                        datum = df.loc[time]
                        datum_values = datum.values
                        return datum_values.item() if datum_values.size == 1 else datum_values, unit

    return None, None