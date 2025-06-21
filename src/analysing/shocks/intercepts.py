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


def find_all_shocks(shocks, parameter, time=None, **kwargs):

    shocks = shocks.copy()
    shocks_attrs = shocks.attrs

    new_columns = {}

    for region in ['L1', 'Earth']:
        for sc in few_spacecraft.get(region, []):

            new_columns[f'{sc}_time'] = pd.NaT
            new_columns[f'{sc}_time_unc_s'] = np.nan
            new_columns[f'{sc}_coeff'] = np.nan

            if sc == 'OMNI':
                new_columns['OMNI_sc'] = np.nan

            for comp in ('x', 'y', 'z'):
                new_columns[f'{sc}_r_{comp}_GSE'] = np.nan
                new_columns[f'{sc}_r_{comp}_GSE_unc'] = np.nan

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

def train_propagation_time(shock_time, detector, interceptor, parameter, position=None, **kwargs):

    # Function is specifically to find time and do nothing else

    R_E            = kwargs.get('R_E',6370)
    position_var   = kwargs.get('position_var','R_GSE')
    buffer_up      = kwargs.get('buffer_up',15)
    buffer_dw      = kwargs.get('buffer_dw',30)
    region         = kwargs.get('region','Earth')
    resolution     = kwargs.get('resolution',None)
    intercept_pos  = kwargs.get('intercept_pos',None)
    overlap_mins   = buffer_up



    # Time around shock front
    start_up = shock_time-timedelta(minutes=buffer_up)
    end_up   = shock_time+timedelta(minutes=buffer_up)

    ###----------DETECTOR POSITION----------###
    if position is None:
        position, _ = retrieve_datum(position_var, detector, speasy_variables, shock_time, add_omni_sc=False)
        if position is None:
            raise Exception(f'No location data for detector {detector}.')


    if position is None:
        delay_time = 3600 # if no position, take delay time as an hour
    else:
        displacement = position
        if intercept_pos is not None:
            displacement = position-intercept_pos

        delay_time = np.linalg.norm(displacement)*R_E/500
        delay_time *= np.sign(displacement[0]) #if downstream, assume negative delay

    # Start/end at shock
    if region == 'Earth' and delay_time>buffer_up*60:
        start_dw = shock_time
        end_dw   = shock_time + timedelta(seconds=int(delay_time)) + timedelta(minutes=buffer_dw)
    elif region == 'Earth' and delay_time<-buffer_up*60:
        start_dw = shock_time + timedelta(seconds=int(delay_time)) - timedelta(minutes=buffer_dw)
        end_dw   = shock_time
    else:
        start_dw = shock_time-timedelta(minutes=buffer_dw)
        end_dw   = shock_time+timedelta(minutes=buffer_dw)


     # Buffer around delay only
    # if region == 'Earth' and delay_time>buffer_up*60:
    #     start_dw = shock_time + timedelta(seconds=int(delay_time)) - timedelta(minutes=buffer_dw)
    #     start_dw = max(shock_time, start_dw)
    #     end_dw   = shock_time + timedelta(seconds=int(delay_time)) + timedelta(minutes=buffer_dw)
    # elif region == 'Earth' and delay_time<-buffer_up*60:
    #     start_dw = shock_time + timedelta(seconds=int(delay_time)) - timedelta(minutes=buffer_dw)
    #     end_dw   = shock_time + timedelta(seconds=int(delay_time)) + timedelta(minutes=buffer_dw)
    #     end_dw   = min(shock_time, end_dw)
    # else:
    #     start_dw = shock_time-timedelta(minutes=buffer_dw)
    #     end_dw   = shock_time+timedelta(minutes=buffer_dw)

    ### Alternative: consider the quadrant of 3D space and whether infinite plane would have to intercept first or not - use only spacecraft positions

    if resolution is None:
        if set(['OMNI','DSC']).intersection([detector,interceptor]):
            resolution = 60
        elif set(['ACE','IMP8']).intersection([detector,interceptor]):
            resolution = 30
        else:
            resolution = 15
    sampling_interval = f'{resolution}s'

    parameters = []
    time_lags  = []
    lag_coeffs = []

    if parameter=='field':

        ###-------------------VECTOR-------------------###
        parameter = 'B_GSE'

        data1 = retrieve_data(parameter, detector, speasy_variables, start_up, end_up, downsample=True, resolution=sampling_interval, add_omni_sc=False)

        data2 = retrieve_data(parameter, interceptor, speasy_variables, start_dw, end_dw, downsample=True, resolution=sampling_interval, add_omni_sc=False)

        if data1.empty or data2.empty:
            return None, None, None

        for comp in ('x','y','z'):
            param_comp = f'{parameter}_{comp}'

            data1_comp = data1[[param_comp]]
            data2_comp = data2[[param_comp]]

            lag, unc, coeff = find_peak_cross_corr(param_comp, data1_comp, data2_comp, detector, interceptor, shock_time, resolution, overlap_mins=overlap_mins)
            if not np.isnan(lag):
                time_lags.append(ufloat(lag,unc))
                lag_coeffs.append(coeff)
                parameters.append(param_comp)

        parameter = 'B_mag'

    ###-------------------MAGNITUDE OR OTHER-------------------###
    data1 = retrieve_data(parameter, detector, speasy_variables, start_up, end_up, downsample=True, resolution=sampling_interval, add_omni_sc=False)

    data2 = retrieve_data(parameter, interceptor, speasy_variables, start_dw, end_dw, downsample=True, resolution=sampling_interval, add_omni_sc=False)

    if data1.empty or data2.empty:
        return None, None, None

    lag, unc, coeff = find_peak_cross_corr(parameter, data1, data2, detector, interceptor, shock_time, resolution, overlap_mins=overlap_mins)
    if not np.isnan(lag):
        time_lags.insert(0,ufloat(lag,unc))
        lag_coeffs.insert(0,coeff)
        parameters.insert(0,parameter)


    ###-------------------FIND MAX-------------------###
    if len(time_lags)==0:
        return None, None, None

    time_lags  = np.array(time_lags)
    lag_coeffs = np.array(lag_coeffs)
    parameters = np.array(parameters)

    time_lag = time_lags[lag_coeffs.argmax()]
    parameter = parameters[lag_coeffs.argmax()]
    lag_coeff = np.max(lag_coeffs)



    return time_lag, lag_coeff, parameter


