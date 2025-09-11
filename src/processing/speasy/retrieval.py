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

from speasy.signal.resampling import interpolate
from datetime import timedelta

from .config import data_availability_plasma, data_availability_magnetic, data_availability_electric, speasy_variables
from .calculations import cross_product
from ..dataframes import resample_data

from ..omni.config import omni_spacecraft



# %% Multiple_parameters

def retrieve_multiple_data(source, parameters, intervals, print_error=False):

    products = []
    labels   = []
    tuples   = []

    for param in parameters:

        data_path = speasy_variables.get(source,None)
        if data_path is None:
            if print_error:
                print(f'{source} not valid source variable.')
            return pd.DataFrame()

        data_id = data_path.get(param, None)
        if data_id is None:
            if print_error:
                print(f'{param} data for {source} not implemented.')
            return pd.DataFrame()

        if isinstance(data_id, tuple):
            method, id1, id2 = data_id
            tuples.append({param: data_id})
            if id1 not in parameters:
                parameters.append(id1)
            if id2 not in parameters:
                parameters.append(id2)
        elif isinstance(data_id, str):
            products.append('amda/'+data_id)
            labels.append(param)
        else:
            products.append(data_id)
            labels.append(param)

    if len(products)>0:
        spz_data = spz.get_data(products, intervals)

    df_dict = {}

    for i, name in enumerate(labels):
        if len(spz_data[i])==0:
            continue

        df_param = pd.DataFrame()

        if not isinstance(spz_data[i][0],list):

            for interval_obj in spz_data[i]:
                if interval_obj is not None:
                    unit = interval_obj.unit
                    try:
                        convert = interval_obj.meta['SI_CONVERSION']
                    except:
                        convert = 1
                    #print_spz_data(interval_obj)
                    try:
                        df_new = interval_obj.replace_fillval_by_nan().to_dataframe()
                    except:
                        df_new = interval_obj.to_dataframe()
                    df_param = pd.concat([df_param,df_new])
        else:

            for j, comp in enumerate(('x','y','z')):

                df_comp = pd.DataFrame()

                for interval_obj in spz_data[i][j]:
                    if interval_obj is not None:
                        unit = interval_obj.unit
                        try:
                            convert = interval_obj.meta['SI_CONVERSION']
                        except:
                            convert = 1
                        #print_spz_data(interval_obj)
                        try:
                            df_new = interval_obj.replace_fillval_by_nan().to_dataframe()
                        except:
                            df_new = interval_obj.to_dataframe()
                        df_comp = pd.concat([df_comp,df_new])

                df_param = pd.concat([df_param, df_comp], axis=1)

        df_param.index.name = 'epoch'
        df_param.attrs = {'units': {}, 'conversions': {}}
        if not df_param.empty:

            if '_GS' in name:
                field, coords = name.split('_')
                df_param.columns = [f'{field}_{comp}_{coords}' for comp in ('x','y','z')]
            else:
                df_param.columns = [name]

            df_param.attrs['units'][name] = unit
            df_param.attrs['conversions'][name] = convert
            for col in df_param.columns:
                df_param.attrs['units'][col] = unit
                df_param.attrs['conversions'][name] = convert

        df_dict[name] = df_param

    for param, df in df_dict.items():
        df_dict[param] = df[~df.index.duplicated(keep='first')]

    for tuple_dict in tuples:
        for tuple_param, tuple_id in tuple_dict.items():
            method, id1, id2 = tuple_id

            if method=='CROSS':
                df_dict[tuple_param] = cross_product(df_dict[id1],df_dict[id2],tuple_param)
            else:
                print(f'{method} not implemented for {tuple_param}')

    return df_dict



def print_spz_data(spz_data):

    print("===========================================")
    print(f"Name:         {spz_data.name}")
    print(f"Columns:      {spz_data.columns}")
    print(f"Values Unit:  {spz_data.unit}")
    print(f"Memory usage: {spz_data.nbytes} Bytes")
    print(f"Axes Labels:  {spz_data.axes_labels}")
    print("-------------------------------------------")
    print(f"Meta-data:    {spz_data.meta}")
    print(f"Conversion    {spz_data.meta['SI_CONVERSION']}")
    print("-------------------------------------------")
    print(f"Time Axis:    {spz_data.time[:3]}")
    print("-------------------------------------------")
    print(f"Values:       {spz_data.values[:3]}")
    print("===========================================")

# %% Single_parameter_time

def clean_retrieved_data(spz_data, upsample_data=None):
    if spz_data is None or len(spz_data.time) == 0:
        return pd.DataFrame()
    spz_data = spz_data.replace_fillval_by_nan()

    if spz_data.values is None or len(spz_data.values) == 0:
        return pd.DataFrame()

    if upsample_data is not None:
        if len(upsample_data.time) == 0 or len(spz_data.time) == 0:
            return spz_data
        spz_data = interpolate(upsample_data, spz_data)

    return spz_data



def retrieve_data(parameter, source, start_time, end_time, downsample=False, upsample=False, resolution='1min', print_error=False):
    if 'B_' in parameter:
        data_range = data_availability_magnetic
    elif 'E_' in parameter:
        data_range = data_availability_electric
    else:
        data_range = data_availability_plasma
    available_range = data_range.get(source,None)

    if available_range is not None:
        available_start, available_end = available_range

        if start_time < available_start and end_time < available_start:
            if print_error:
                print(f'{source} outside available range: {available_start} to {available_end}')
            return pd.DataFrame()
        elif start_time > available_end and end_time > available_end:
            if print_error:
                print(f'{source} outside available range: {available_start} to {available_end}')
            return pd.DataFrame()

    df_dict = retrieve_multiple_data(source, [parameter], [[str(start_time), str(end_time)]], print_error)
    df = df_dict[parameter]

    if upsample: # used to upsample to 1-minute resolution
        df = df.resample('1min').interpolate(method='time')

    elif downsample:
        attributes = df.attrs
        df = resample_data(df, 'index', sample_interval=resolution)
        df.attrs = attributes


    return df


def retrieve_datum(parameter, source, time, print_error=False, add_omni_sc=False):

    if time is None:
        return None, None

    available_start, available_end = data_availability_magnetic.get(source)
    if time < available_start or time > available_end:
        if print_error:
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
    sc_ID = speasy_variables.get('OMNI').get('sc')
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

    var = omni_var.split('_')[1]
    omni_ID = speasy_variables.get('OMNI').get(var)
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