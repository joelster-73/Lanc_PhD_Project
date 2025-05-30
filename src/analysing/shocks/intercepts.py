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
from ...processing.speasy.config import speasy_variables, all_spacecraft
from ...processing.utils import add_unit


def find_all_shocks(shocks, parameter, time=None, time_window=20, position_var='R_GSE', R_E=6370):

    shocks = shocks.copy()
    shocks_attrs = shocks.attrs

    new_columns = {}

    for region in ['L1', 'Earth']:
        for sc in all_spacecraft.get(region, []):

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
        sc_dict = find_shock(shock, parameter, time_window, position_var, R_E)

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

            # find_shock(shock, parameter, time_window, position_var, R_E)

            try:
                sc_dict = find_shock(shock, parameter, time_window, position_var, R_E)

                for key, value in sc_dict.items():
                    shocks.at[index, key] = value

            except Exception as e:
                print(f'Issue with shock at time {index}.')
                with open(file_path, 'a') as my_file:
                    sc = shock['spacecraft'].upper()
                    my_file.write(f'{index.strftime("%Y-%m-%d %H:%M:%S")} not added ({sc}): {e}\n')

        return shocks

def find_shock(shock, parameter, time_window=20, position_var='R_GSE', R_E=6370):

    shock_speed = shock['v_sh']
    shock_speed_unc = shock['v_sh_unc'] # Likely issue with shock in database
    if pd.isna(shock_speed):
        e = 'Shock speed is NaN.'
        raise Exception(e)
    elif pd.isna(shock_speed_unc):
        e = 'Shock speed uncertainty is NaN.'
        raise Exception(e)
    elif shock_speed <= 0:
        e = f'Shock speed is negative/zero ({shock_speed}).'
        raise Exception(e)
    elif shock_speed <= shock_speed_unc:
        e = f'Shock speed ({shock_speed}) is smaller than uncertainty ({shock_speed_unc}).'
        raise Exception(e)

    return find_shock_times(shock, parameter, time_window, position_var, R_E)


def find_shock_times(shock, parameter, time_window=20, position_var='R_GSE', R_E=6370):

    return_dict = {}
    shock_time = shock.name.to_pydatetime()
    shock_time_unc = shock['time_s_unc']
    if np.isnan(shock_time_unc):
        shock_time_unc = 0
    sc_L1 = shock['spacecraft'].upper()

    try:
        arrival_time     = shock_time + timedelta(seconds=shock['delay_s'])
    except:
        arrival_time     = shock_time + timedelta(hours=1)

    start_time_up = shock_time-timedelta(minutes=time_window)
    end_time_up   = shock_time+timedelta(minutes=time_window)

    start_time_dw = arrival_time-timedelta(minutes=time_window)
    end_time_dw   = arrival_time+timedelta(minutes=time_window)

    ###----------Detector Data----------###
    df_detect = retrieve_data(parameter, sc_L1, speasy_variables, start_time_up, end_time_up, downsample=True)
    if df_detect.empty: # In case issue with retrieving data
        e = f'No {sc_L1} data available.'
        raise Exception(e)

    return_dict[f'{sc_L1}_time'] = pd.to_datetime(shock_time)
    return_dict[f'{sc_L1}_time_unc_s'] = shock_time_unc
    return_dict[f'{sc_L1}_coeff'] = 1

    for comp in ('x','y','z'):
        return_dict[f'{sc_L1}_r_{comp}_GSE'] = shock[f'r_{comp}_GSE']

    _, sc_pos_unc = retrieve_position_unc(sc_L1, speasy_variables, shock_time, shock_time_unc, shock_time_unc)
    if sc_pos_unc is not None:
        return_dict[f'{sc_L1}_r_x_GSE_unc'], return_dict[f'{sc_L1}_r_y_GSE_unc'], return_dict[f'{sc_L1}_r_z_GSE_unc'] = sc_pos_unc


    ###-------------------OTHER SPACECRAFT-------------------###
    for region in ['L1', 'Earth']:
        for source in all_spacecraft.get(region, []):
            if source == sc_L1:
                continue
            elif source == 'OMNI':
                start = min(start_time_up,start_time_dw)
                end   = max(end_time_up,end_time_dw)
            elif region == 'L1':
                start, end = start_time_up, end_time_up
            elif region == 'Earth':
                start, end = start_time_dw, end_time_dw
            else:
                continue

            if source == 'OMNI':
                df_omni = retrieve_data(parameter, 'OMNI', speasy_variables, start, end)
                if ~df_omni.empty:
                    df_omni_sc = df_omni['spacecraft']
                    return_dict['OMNI_sc'] = stats.mode(df_omni_sc).mode

            else:
                df_sc_pos = retrieve_data(position_var, source, speasy_variables, start, end, upsample=True)
                if df_sc_pos.empty:
                    continue

                in_sw = is_in_solar_wind(source, speasy_variables, start, end, pos_df=df_sc_pos)
                if np.any(~in_sw):
                    continue

            time_lag, lag_unc, lag_coeff = find_time_lag(parameter, sc_L1, source, shock_time, arrival_time)
            if np.isnan(time_lag):
                continue

            front_time = shock_time + timedelta(seconds=time_lag)
            front_time_u = ufloat(time_lag, lag_unc) - ufloat(0,shock_time_unc)
            front_time_u = front_time_u.s

            return_dict[f'{source}_time'] = front_time
            return_dict[f'{source}_time_unc_s'] = front_time_u
            return_dict[f'{source}_coeff'] = lag_coeff

            sc_pos, sc_pos_unc = retrieve_position_unc(source, speasy_variables, front_time, front_time_u, front_time_u)

            if sc_pos is not None:
                return_dict[f'{source}_r_x_GSE'], return_dict[f'{source}_r_y_GSE'], return_dict[f'{source}_r_z_GSE'] = sc_pos
                return_dict[f'{source}_r_x_GSE_unc'], return_dict[f'{source}_r_y_GSE_unc'], return_dict[f'{source}_r_z_GSE_unc'] = sc_pos_unc

    return return_dict