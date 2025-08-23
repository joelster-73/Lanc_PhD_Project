# -*- coding: utf-8 -*-
"""
Created on Mon May 12 12:02:37 2025

@author: richarj2
"""
import numpy as np
import pandas as pd
import speasy as spz
import scipy

from uncertainties import unumpy as unp

from collections import Counter

from speasy import amda
from speasy.signal.resampling import interpolate
from datetime import timedelta

from .config import data_availability, data_availability_mag, speasy_variables
from ..dataframes import resample_data
from ...config import R_E

from ..omni.config import omni_spacecraft

def clean_retrieved_data(spz_data, upsample_data=None):
    if spz_data is None or len(spz_data.time) == 0:
        return None
    spz_data = spz_data.replace_fillval_by_nan()

    if spz_data.values is None or len(spz_data.values) == 0:
        return None

    if upsample_data is not None:
        if len(upsample_data.time) == 0 or len(spz_data.time) == 0:
            return spz_data
        spz_data = interpolate(upsample_data, spz_data)

    return spz_data


def retrieve_data(parameter, source, start_time, end_time, downsample=False, upsample=False, print_bounds=False, resolution='1min', add_omni_sc=True):
    if 'B_' in parameter:
        data_range = data_availability_mag
    else:
        data_range = data_availability
    available_start, available_end = data_range.get(source)

    if start_time < available_start and end_time < available_start:
        if print_bounds:
            print(f'{source} outside available range: {available_start} to {available_end}')
        return pd.DataFrame()
    elif start_time > available_end and end_time > available_end:
        if print_bounds:
            print(f'{source} outside available range: {available_start} to {available_end}')
        return pd.DataFrame()

    data_path = speasy_variables.get(parameter,None)
    if data_path is None:
        print(f'{parameter} not valid parameter variable.')
    data_id = data_path.get(source, None)
    if data_id is None:
        print(f'No {parameter} data for {source}.')
        return pd.DataFrame()

    upsample_data = None
    if upsample: # used to upsample to 1-minute resolution
        upsample_data = amda.get_data(speasy_variables.get('B_mag').get('OMNI'), start_time, end_time)

    if isinstance(data_id, list):
        vec_data = spz.get_data(data_id, start_time, end_time)
        vec_data = clean_retrieved_data(vec_data, upsample_data)

        if vec_data is None:
            return pd.DataFrame()

        times = vec_data[0].time
        values = np.column_stack((vec_data[0].values,vec_data[1].values,vec_data[2].values))
        unit = vec_data[0].unit

    else:
        if isinstance(data_id, str):
            spz_data = amda.get_data(data_id, start_time, end_time)
        else:
            spz_data = spz.get_data(data_id, start_time, end_time)

        spz_data = clean_retrieved_data(spz_data, upsample_data)
        if spz_data is None:
            return pd.DataFrame()

        times, values, unit = spz_data.time, spz_data.values, spz_data.unit

    if source == 'IMP8':
        # seems to be problem with the fill value, metadata says 9999.9, but seems to be about double
        values[values>=9999.9] = np.nan

    if unit=='km':
        values /= R_E
        unit = 'Re'

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
        df.attrs['units'][parameter] = unit
        for col in column_names:
            df.attrs['units'][col] = unit

    if add_omni_sc and source == 'OMNI':
        sc_ID = speasy_variables.get('OMNI_sc')
        id_data = spz.get_data(sc_ID, start_time, end_time)
        _, ids, _ = id_data.time, id_data.values, id_data.unit
        df['spacecraft'] = ids

    if downsample:
        attributes = df.attrs
        df = resample_data(df, 'index', sample_interval=resolution)
        df.attrs = attributes

    # In case duplicate times are returned
    df = df.loc[~df.index.duplicated(keep='first')]

    return df


def retrieve_datum(parameter, source, time, print_bounds=False, add_omni_sc=False):

    if time is None:
        return None, None

    available_start, available_end = data_availability.get(source)
    if time < available_start or time > available_end:
        if print_bounds:
            print(f'{source} outside available range: {available_start} to {available_end}')
        return None, None

    window = 1 # minutes
    max_attempts = 60
    for counter in range(1, max_attempts + 1):
        interval = timedelta(minutes=window*counter)
        df = retrieve_data(parameter, source, time-interval, time+interval, add_omni_sc=add_omni_sc)
        df.dropna(inplace=True)
        if df.empty:
            continue

        if time not in df.index and (len(df) > 1 and df.index.min() <= time <= df.index.max()):
            df = df.reindex(df.index.union([time])).sort_index()
            df = df.interpolate(method='linear')
            df = df.loc[~df.index.duplicated(keep='first')]

        if time in df.index:
            datum = df.loc[time]
            if isinstance(datum, pd.DataFrame):
                datum = datum.iloc[0]
            datum_values = datum.values
            unit = df.attrs['units'].get(parameter,None)
            return datum_values.item() if datum_values.size == 1 else datum_values, unit

    return None, None

def retrieve_position_unc(source, time, time_unc):

    position_var = 'R_GSE'

    position, _ = retrieve_datum(position_var, source, time, add_omni_sc=False)
    if position is None:
        return None, None

    pos_left, _ = retrieve_datum(position_var, source, time-timedelta(seconds=time_unc), add_omni_sc=False)
    pos_right, _ = retrieve_datum(position_var, source, time+timedelta(seconds=time_unc), add_omni_sc=False)
    for arr in (position, pos_left, pos_right):
        arr = np.array(arr)


    if pos_left is not None and pos_right is not None:
        unc = (pos_right - pos_left) / 2
    elif pos_left is not None:
        unc = (position - pos_left)
    elif pos_right is not None:
        unc = (pos_right - position)
    else:
        unc = np.zeros(3)


    return position, np.abs(unc)

def retrieve_modal_omni_sc(start_time, end_time, return_counts=False):
    if start_time==end_time:
        start_time, end_time = start_time-timedelta(minutes=10), start_time+timedelta(minutes=10)
    sc_ID = speasy_variables.get('OMNI_sc')
    id_data = spz.get_data(sc_ID, start_time, end_time)
    if id_data is None:
        return None
    _, ids, _ = id_data.time, id_data.values, id_data.unit

    modal_id = scipy.stats.mode(ids, axis=None)[0]
    modal_sc = omni_spacecraft.get(modal_id,modal_id)

    if return_counts:
        counts_dict = Counter(np.ravel(ids))
        counts_dict = dict(sorted(counts_dict.items(), key=lambda item: item[1], reverse=True))
        return modal_sc, {omni_spacecraft.get(key,key): value for key, value in counts_dict.items()}

    return modal_sc

def retrieve_omni_value(omni_time, omni_var='OMNI_lag'):

    omni_ID = speasy_variables.get(omni_var)
    omni_time = omni_time.replace(second=0, microsecond=0)
    omni_datum = spz.get_data(omni_ID, omni_time-timedelta(seconds=30), omni_time+timedelta(seconds=30))
    if omni_datum is None:
        return np.nan if omni_var=='OMNI_lag' else None
    datum = omni_datum.values[0][0]
    if omni_var=='OMNI_sc':
        sc = omni_spacecraft.get(datum,datum).upper()
        if sc=='WIND-V2':
            sc = 'WIND'
        return sc
    return datum


def get_shock_position(shock, sc):


    try:
        sc_pos = shock[[f'{sc}_r_{comp}_GSE' for comp in ('x','y','z')]].to_numpy()
        sc_pos_unc = shock[[f'{sc}_r_{comp}_GSE_unc' for comp in ('x','y','z')]].to_numpy()
    except:
        try:
            sc_time = shock[f'{sc}_time']
            sc_time_unc = shock[f'{sc}_time_unc_s']

            if pd.isnull(sc_time):
                return None

            sc_pos, sc_pos_unc = retrieve_position_unc(sc, sc_time, sc_time_unc)

        except:
            return None

    if sc_pos is None or np.isnan(sc_pos[0]):
        return None
    elif np.isnan(sc_pos_unc[0]):
        sc_pos_unc = np.zeros(len(sc_pos))

    return unp.uarray(sc_pos,sc_pos_unc)