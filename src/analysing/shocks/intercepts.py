# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:20:40 2025

@author: richarj2
"""
import os
import numpy as np
import pandas as pd

from datetime import timedelta, datetime
from scipy import stats
from uncertainties import ufloat

from .discontinuities import find_peak_cross_corr
from .in_sw import is_in_solar_wind

from ...processing.speasy.retrieval import retrieve_data, retrieve_datum
from ...processing.speasy.config import speasy_variables, few_spacecraft
from ...processing.utils import add_unit

from ...config import R_E



def find_all_shocks(shocks, parameter, time=None, **kwargs):

    shocks = shocks.copy()
    shocks_attrs = shocks.attrs

    new_columns = {}

    for region in ['L1', 'Earth']:
        for sc in few_spacecraft.get(region, []):

            new_columns[f'{sc}_time'] = pd.NaT
            new_columns[f'{sc}_time_unc_s'] = np.nan

            if sc == 'OMNI':
                new_columns['OMNI_sc'] = np.nan

            for comp in ('x', 'y', 'z'):
                new_columns[f'{sc}_r_{comp}_GSE'] = np.nan

    # Add all new columns to the dataframe at once
    shocks = pd.concat([shocks, pd.DataFrame(new_columns, index=shocks.index)], axis=1)
    shocks.attrs = shocks_attrs

    for key in shocks.columns:
        if key not in shocks.attrs['units']:
            shocks.attrs['units'][key] = add_unit(key)

    if time is not None:
        time_shock = time
        nearest_idx = shocks.index.searchsorted(time_shock, side='right')
        nearest_time = shocks.index[nearest_idx]
        shock = shocks.loc[nearest_time].copy()
        sc_dict = find_shock_times(shock, parameter, **kwargs)

        for key, value in sc_dict.items():
            shock[key] = value

        return shock

    else:
        script_dir = os.getcwd() # change to location of script __file__
        file_name = 'Shocks_not_processed.txt'
        file_path = os.path.join(script_dir, file_name)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as my_file:
                my_file.write(f'Log created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

        for index, shock in shocks.iterrows():

            # sc_dict = find_shock_times(shock, parameter, time_window, position_var, R_E)

            try:
                sc_dict = find_shock_times(shock, parameter, **kwargs)

                for key, value in sc_dict.items():
                    shocks.at[index, key] = value

            except Exception as e:
                print(f'Issue with shock at time {index}: {e}')
                with open(file_path, 'a') as my_file:
                    sc = shock['spacecraft'].upper()
                    my_file.write(f'{index.strftime("%Y-%m-%d %H:%M:%S")} not added ({sc}): {e}\n')

        return shocks

# check find_times and move some stuff into parent function

def find_shock_times():

    # in this function, go through all possible spacecraft or keys
    # decide if in solar wind or other things like this
    # then call find_propagation_time

    return

def find_propagation_time(shock_time, detector, interceptor, parameter, position=None, **kwargs):


    position_var   = kwargs.get('position_var','R_GSE')
    buffer_up      = kwargs.get('buffer_up',33)
    buffer_dw      = kwargs.get('buffer_dw',35)
    resolution     = kwargs.get('resolution',None)
    intercept_pos  = kwargs.get('intercept_pos',None)

    distance_buff  = kwargs.get('distance_buff',50)
    max_neg_delay  = kwargs.get('max_neg_delay',40)
    max_pos_delay  = kwargs.get('max_neg_delay',90)

    if resolution is None:
        if set(['OMNI','DSC']).intersection([detector,interceptor]):
            resolution = 60
        elif set(['ACE','IMP8']).intersection([detector,interceptor]):
            resolution = 30
        else:
            resolution = 15
    sampling_interval = f'{resolution}s'


    # Time around shock front
    start_up = shock_time-timedelta(minutes=buffer_up)
    end_up   = shock_time+timedelta(minutes=buffer_up)

    ###----------DETECTOR POSITION----------###
    if position is None:
        position, _ = retrieve_datum(position_var, detector, speasy_variables, shock_time, add_omni_sc=False)
        if position is None:
            raise Exception(f'No location data for detector {detector}.')

    r_diff = position
    if intercept_pos is not None:
        r_diff = (position-intercept_pos)
    x_diff = r_diff[0]

    # Start/end at shock
    if x_diff > distance_buff:         # confident positive delay
        start_dw_mins = 10
        end_dw_mins   = max_pos_delay
    elif x_diff < -distance_buff:      # confident negative delay
        start_dw_mins = -max_neg_delay
        end_dw_mins   = 0
    else:
        start_dw_mins = -buffer_dw
        end_dw_mins   = buffer_dw

    # Possible lags from shock time
    start_lag = int(start_dw_mins*60/resolution)*resolution
    end_lag   = int(end_dw_mins*60/resolution)*resolution
    lags      = range(start_lag, end_lag+1, resolution)

    # Data needed to compare
    start_dw = shock_time + timedelta(minutes=start_dw_mins) - timedelta(minutes=buffer_up)
    end_dw   = shock_time + timedelta(minutes=end_dw_mins)   + timedelta(minutes=buffer_up)



    ###-------------------PARAMETER-------------------###
    data1 = retrieve_data(parameter, detector, speasy_variables, start_up, end_up, downsample=True, resolution=sampling_interval, add_omni_sc=False)

    data2 = retrieve_data(parameter, interceptor, speasy_variables, start_dw, end_dw, downsample=True, resolution=sampling_interval, add_omni_sc=False)

    if data1.empty or data2.empty:
        return None

    aligned = pd.merge(data1[[parameter]].rename(columns={parameter: detector}),
                       data2[[parameter]].rename(columns={parameter: interceptor}),
                       left_index=True, right_index=True, how='outer')

    series1 = aligned[detector]
    series2 = aligned[interceptor]

    lag, coeff = find_peak_cross_corr(parameter, series1, series2, detector, interceptor, shock_time, lags, resolution, **kwargs)
    if not np.isnan(lag):
        return ufloat(lag,resolution), coeff

    return None




