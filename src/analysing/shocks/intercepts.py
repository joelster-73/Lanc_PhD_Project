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

from .discontinuities import find_time_lag
from .in_sw import is_in_solar_wind

from ...processing.speasy.retrieval import retrieve_data, retrieve_position_unc
from ...processing.speasy.config import speasy_variables, few_spacecraft
from ...processing.utils import add_unit


def find_all_shocks(shocks, parameter, time=None, time_window=60, position_var='R_GSE', R_E=6370):

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
        sc_dict = find_shock_times(shock, parameter, time_window, position_var, R_E)

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
                sc_dict = find_shock_times(shock, parameter, time_window, position_var, R_E)

                for key, value in sc_dict.items():
                    shocks.at[index, key] = value

            except Exception as e:
                print(f'Issue with shock at time {index}: {e}')
                with open(file_path, 'a') as my_file:
                    sc = shock['spacecraft'].upper()
                    my_file.write(f'{index.strftime("%Y-%m-%d %H:%M:%S")} not added ({sc}): {e}\n')

        return shocks


def find_shock_times(shock, parameter, time_window=60, position_var='R_GSE', R_E=6370):

    #time_window is to check position data and whether in the solar wind or not

    return_dict = {}
    shock_time = shock.name.to_pydatetime()

    print(shock_time)

    # Time around shock front
    time_window_up = time_window
    start_time_up = shock_time-timedelta(minutes=time_window_up)
    end_time_up   = shock_time+timedelta(minutes=time_window_up)

    sc_L1 = shock['spacecraft'].upper()

    ###----------SHOCK TIME----------###
    param = parameter if parameter!='field' else 'B_mag'

    df_detect = retrieve_data(param, sc_L1, speasy_variables, start_time_up, end_time_up, downsample=True)
    if df_detect.empty: # In case issue with retrieving data
        e = f'No {sc_L1} {parameter} data available.'
        raise Exception(e)

    shock_time_unc = shock['time_s_unc']
    if np.isnan(shock_time_unc):
        shock_time_unc = 0

    return_dict[f'{sc_L1}_time'] = pd.to_datetime(shock_time)
    return_dict[f'{sc_L1}_time_unc_s'] = shock_time_unc
    return_dict[f'{sc_L1}_coeff'] = 1.1 # >1 to be clear this is an exact match

    ###----------DETECTOR POSITION----------###
    sc_pos, sc_pos_unc = retrieve_position_unc(sc_L1, speasy_variables, shock_time, shock_time_unc, shock_time_unc)
    if sc_pos is None:
        e = f'No location data for detector {sc_L1}.'
        raise Exception(e)

    delay_time = np.linalg.norm(sc_pos)*R_E/500

    return_dict[f'{sc_L1}_r_x_GSE'], return_dict[f'{sc_L1}_r_y_GSE'], return_dict[f'{sc_L1}_r_z_GSE'] = sc_pos
    return_dict[f'{sc_L1}_r_x_GSE_unc'], return_dict[f'{sc_L1}_r_y_GSE_unc'], return_dict[f'{sc_L1}_r_z_GSE_unc'] = sc_pos_unc

    ###-------------------OTHER SPACECRAFT-------------------###
    for region in ['L1', 'Earth']:
        for source in few_spacecraft.get(region, []):

            if source == sc_L1:
                continue

            if region == 'L1':
                buffer_dw = 40
                start = shock_time-timedelta(minutes=buffer_dw)
                end   = shock_time+timedelta(minutes=buffer_dw)

            elif region == 'Earth':
                arrival_time = shock_time + timedelta(seconds=int(delay_time))
                buffer_dw = 120 # large search window
                start = arrival_time-timedelta(minutes=buffer_dw)
                end   = arrival_time+timedelta(minutes=buffer_dw)
                start = max(start, shock_time) # can't arrive downstream before shock measured
            else:
                continue

            df_sc_pos = retrieve_data(position_var, source, speasy_variables, start, end, upsample=True)
            if df_sc_pos.empty:
                continue

            if source=='OMNI':
                df_omni_sc = df_sc_pos['spacecraft']
                modal_sc = stats.mode(df_omni_sc).mode
                return_dict['OMNI_sc'] = modal_sc
                if modal_sc==99:
                    continue
            else:
                in_sw = is_in_solar_wind(source, speasy_variables, start, end, pos_df=df_sc_pos)
                if np.any(~in_sw):
                    continue

            if set(['OMNI','DSC']).intersection([sc_L1,source]):
                resolution = 60
            elif set(['ACE','IMP8']).intersection([sc_L1,source]):
                resolution = 30
            else:
                resolution = 15
            sampling_interval = f'{resolution}s'

            if parameter=='field':
                time_lags = []
                lag_coeffs = []

                ###-------------------MAGNITUDE-------------------###
                param = 'B_mag'

                data1 = retrieve_data(param, sc_L1, speasy_variables,
                                          start_time_up, end_time_up, downsample=True, resolution=sampling_interval)


                data2 = retrieve_data(param, source, speasy_variables,
                                          start, end, downsample=True, resolution=sampling_interval, add_omni_sc=False)

                if not data1.empty and not data2.empty:

                    lag, unc, coeff = find_time_lag(param, data1, data2, sc_L1, source, shock_time, resolution, sampling_interval, buffer_dw)
                    if not np.isnan(lag):
                        time_lags.append(ufloat(lag,unc))
                        lag_coeffs.append(coeff)

                ###-------------------VECTOR-------------------###
                param = 'B_GSE'

                data1 = retrieve_data(param, sc_L1, speasy_variables,
                                          start_time_up, end_time_up, downsample=True, resolution=sampling_interval)


                data2 = retrieve_data(param, source, speasy_variables,
                                          start, end, downsample=True, resolution=sampling_interval, add_omni_sc=False)

                if not data1.empty and not data2.empty:

                    for comp in ('x','y','z'):
                        param_comp = f'{param}_{comp}'

                        data1_comp = data1[[param_comp]]
                        data2_comp = data2[[param_comp]]

                        lag, unc, coeff = find_time_lag(param_comp, data1_comp, data2_comp, sc_L1, source, shock_time, resolution, sampling_interval, buffer_dw)
                        if not np.isnan(lag):
                            time_lags.append(ufloat(lag,unc))
                            lag_coeffs.append(coeff)

                ###-------------------FIND MAX-------------------###
                if len(time_lags)==0:
                    continue
                else:
                    time_lags = np.array(time_lags)
                    lag_coeffs = np.array(lag_coeffs)

                    time_lag_u = time_lags[lag_coeffs.argmax()]
                    time_lag = time_lag_u.n
                    lag_unc  = time_lag_u.s
                    lag_coeff = np.max(lag_coeffs)

            else:
                data1 = retrieve_data(param, sc_L1, speasy_variables,
                                          start_time_up, end_time_up, downsample=True, resolution=sampling_interval)


                data2 = retrieve_data(param, source, speasy_variables,
                                          start, end, downsample=True, resolution=sampling_interval, add_omni_sc=False)

                if not data1.empty and not data2.empty:

                    time_lag, lag_unc, lag_coeff = find_time_lag(param, data1, data2, sc_L1, source, shock_time, resolution, sampling_interval, buffer_dw)
                    if np.isnan(time_lag):
                        continue


            front_time = shock_time + timedelta(seconds=time_lag)
            front_time_unc = (ufloat(time_lag, lag_unc) - ufloat(0,shock_time_unc)).s

            return_dict[f'{source}_time'] = front_time
            return_dict[f'{source}_time_unc_s'] = front_time_unc
            return_dict[f'{source}_coeff'] = lag_coeff

            sc_pos, sc_pos_unc = retrieve_position_unc(source, speasy_variables, front_time, front_time_unc, front_time_unc)
            if sc_pos is not None:
                return_dict[f'{source}_r_x_GSE'], return_dict[f'{source}_r_y_GSE'], return_dict[f'{source}_r_z_GSE'] = sc_pos
                return_dict[f'{source}_r_x_GSE_unc'], return_dict[f'{source}_r_y_GSE_unc'], return_dict[f'{source}_r_z_GSE_unc'] = sc_pos_unc

    return return_dict