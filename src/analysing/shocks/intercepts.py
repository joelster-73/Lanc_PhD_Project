# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:20:40 2025

@author: richarj2
"""
import os
import numpy as np
import pandas as pd

from datetime import timedelta, datetime
from uncertainties import ufloat

from .discontinuities import find_peak_cross_corr
from .in_sw import in_solar_wind

from ...processing.speasy.retrieval import retrieve_data, retrieve_datum, retrieve_modal_omni_sc, retrieve_position_unc
from ...processing.speasy.config import speasy_variables, sw_monitors, cluster_sc, themis_sc
from ...processing.dataframes import add_df_units

from ...config import R_E

def find_all_shocks(shocks, parameter, time=None, shocks_intercepts_started=None, starting_ID=None, **kwargs):

    # Shocks contains the known shock times
    # Dataframe to store shock times in shocks_intercepts

    if shocks_intercepts_started is None:
        shocks_intercepts = pd.DataFrame()
        new_columns = {}

        new_columns['detectors'] = ''
        for sc in sum((sw_monitors, ('OMNI',)), ()):

            new_columns[f'{sc}_time']       = pd.NaT
            new_columns[f'{sc}_time_unc_s'] = np.nan
            new_columns[f'{sc}_coeff']      = np.nan
            new_columns[f'{sc}_sc']         = ''

            for comp in ('x','y','z'):
                new_columns[f'{sc}_r_{comp}_GSE'] = np.nan
                new_columns[f'{sc}_r_{comp}_GSE_unc'] = np.nan

        new_columns['OMNI_sc'] = ''

        eventID_max = np.max(shocks['eventNum'].astype(int))
        eventIDs = range(1,eventID_max+1)
        shocks_intercepts = pd.concat([shocks_intercepts, pd.DataFrame(new_columns, index=eventIDs)], axis=1)
    else:
        shocks_intercepts = shocks_intercepts_started

    add_df_units(shocks_intercepts)

    shocks_intercepts.attrs['units']['detectors'] = 'LIST'
    shocks_intercepts.attrs['units']['OMNI_sc'] = 'STRING'

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
        # Store info for each shock event
        script_dir = os.getcwd() # change to location of script __file__
        file_name = 'Processing_events.txt'
        file_path = os.path.join(script_dir, file_name)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as my_file:
                my_file.write(f'Log created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

        try:
            for eventID, event in shocks.groupby(lambda x: int(shocks.loc[x, 'eventNum'])):

                if starting_ID is not None and eventID < starting_ID:
                    continue

                try:
                    find_shock_times(eventID, event, shocks_intercepts, **kwargs)

                except Exception as e:
                    print(e)
                    with open(file_path, 'a') as my_file:
                        my_file.write(f'{e}\n')

                else:
                    print(f'#{eventID}: Found OMNI time.')
                    with open(file_path, 'a') as my_file:
                        my_file.write(f'#{eventID}: Found OMNI time.\n')

        except KeyboardInterrupt:
            print(f'\n#{eventID}: Manual interrupt detected. Returning partial results...')
            with open(file_path, 'a') as my_file:
                my_file.write(f'\n#{eventID}: Manual interrupt detected. Returning partial results.\n')

        return shocks_intercepts


def find_shock_times(eventID, event, df_shocks, **kwargs):

    # Event contains the known shock times we're using
    # df_shocks is where we store the correlated times

    position_var = kwargs.get('position_var','R_GSE')

    ###-------------------INITIAL CHECKS-------------------###

    # Info for all shocks in this event
    times       = event.index.tolist()
    uncs        = event['time_unc'].tolist()
    detectors   = event['spacecraft'].tolist()
    detector_dict = dict(zip(detectors,list(zip(times,uncs))))
    max_time    = max(times)

    # Adds shocks in the database to dataframe
    df_shocks.at[eventID,'detectors'] = detectors
    for _, row in event.iterrows():
        sc = row['spacecraft']
        df_shocks.at[eventID,f'{sc}_time'] = row.name
        df_shocks.at[eventID,f'{sc}_time_unc_s'] = row['time_unc']

        pos, unc = retrieve_position_unc(sc, speasy_variables, row.name, row['time_unc'])

        if pos is None:
            pos = row[['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
            unc = np.array([np.nan,np.nan,np.nan])
            if np.isnan(pos[0]):
                continue

        df_shocks.loc[eventID,[f'{sc}_r_x_GSE',f'{sc}_r_y_GSE',f'{sc}_r_z_GSE']] = pos
        df_shocks.loc[eventID,[f'{sc}_r_x_GSE_unc',f'{sc}_r_y_GSE_unc',f'{sc}_r_z_GSE_unc']] = unc

    # Spacecraft OMNI used in its propagation in this time period
    start, end = min(times), max(times)+timedelta(minutes=90)
    sc_info    = retrieve_modal_omni_sc(speasy_variables, start, end, return_counts=True)
    if sc_info is None:
        raise Exception(f'#{eventID}: No OMNI info.')

    modal_sc, counts_dict = sc_info

    ###-------------------FIND WHEN SHOCK ARRIVES AT BSN ACCORDING TO OMNI-------------------###

    # Shocks in the event recorded by spacecraft used by OMNI
    omni_time = pd.NaT

    # Counts_dict is sorted in descending order
    for sc in counts_dict:
        sc = sc.upper()
        if sc in detectors:
            omni_sc = sc
        elif sc=='WIND-V2' and 'WIND' in detectors:
            omni_sc = 'WIND'
        else:
            continue

        detect_time, detect_unc = detector_dict[omni_sc]
        detect_pos = df_shocks.loc[eventID,[f'{omni_sc}_r_x_GSE',f'{omni_sc}_r_y_GSE',f'{omni_sc}_r_z_GSE']].to_numpy()

        # Approximate BSN location
        approx_time = detect_time + timedelta(seconds=((detect_pos[0]-15)*R_E/500))
        omni_pos, _ = retrieve_datum(position_var, 'OMNI', speasy_variables, approx_time, add_omni_sc=False)

        # Find time lag
        time_lag = find_propagation_time(detect_time, omni_sc, 'OMNI', 'B_mag', detect_pos, intercept_pos=omni_pos)
        if time_lag is None:
            continue
        delay, coeff = time_lag
        if not pd.isnull(df_shocks.at[eventID,'OMNI_time']) and coeff<=df_shocks.at[eventID,'OMNI_coeff']:
            continue
        df_shocks.at[eventID,'OMNI_coeff'] = coeff

        # Found suitable lag
        lagged_unc = delay - ufloat(0,detect_unc)
        omni_time  = detect_time + timedelta(seconds=delay.n)
        omni_unc   = lagged_unc.s
        df_shocks.at[eventID,'OMNI_sc'] = sc

        break

    # If there are no shocks in the event recorded by a spacecraft used by OMNI
    # Use the shocks we have to find when they intercept the OMNI spacecraft
    # Currently not implemented to see sample size
    if pd.isnull(omni_time):
        raise Exception(f'#{eventID}: Need to interpolate to find shock in OMNI.')


    df_shocks.at[eventID,'OMNI_time'] = omni_time
    df_shocks.at[eventID,'OMNI_time_unc_s'] = omni_unc

    pos, unc = retrieve_position_unc('OMNI', speasy_variables, omni_time, omni_unc)

    df_shocks.loc[eventID,['OMNI_r_x_GSE','OMNI_r_y_GSE','OMNI_r_z_GSE']] = pos
    df_shocks.loc[eventID,['OMNI_r_x_GSE_unc','OMNI_r_y_GSE_unc','OMNI_r_z_GSE_unc']] = unc

    ###-------------------FIND WHEN SHOCKS INTERCEPT DOWNSTREAM SPACECRAFT-------------------###
    intercept_sc = []
    for interceptor in sw_monitors:
        if interceptor in detectors:
            # Already have time
            continue
        elif interceptor in cluster_sc and (set(intercept_sc) & set(cluster_sc)):
            # If found shock in one Cluster sc, don't need to check others
            continue
        elif interceptor in themis_sc and (set(intercept_sc) & set(themis_sc)):
            # If found shock in one THEMIS sc, don't need to check others
            continue

        # Initial estimate of position
        intercept_pos, _ = retrieve_datum(position_var, interceptor, speasy_variables, max_time, add_omni_sc=False)
        if intercept_pos is None or not in_solar_wind(interceptor, max_time, speasy_variables):
            continue

        # When shock intercepts downstream monitors
        intercept_time  = pd.NaT
        intercept_unc   = np.nan
        intercept_coeff = np.nan
        intercept_sc    = None
        for i, row in event.iterrows(): # All the monitors
            detector = row['spacecraft']

            detector_pos = row[['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
            detector_time = row.name

            # Find how long shocks takes to intercept spacecraft
            time_lag = find_propagation_time(detector_time, detector, interceptor, 'B_mag', detector_pos, intercept_pos=intercept_pos)
            if time_lag is None:
                continue

            delay, coeff = time_lag
            if not pd.isnull(intercept_time) and coeff<=intercept_coeff:
                continue
            df_shocks.at[eventID,'OMNI_coeff'] = coeff

            lagged_unc = delay - ufloat(0,row['time_unc'])
            intercept_time = detector_time + timedelta(seconds=delay.n)
            intercept_unc  = lagged_unc.s
            intercept_coeff  = coeff
            intercept_sc  = detector

        # If shock isn't found to interept the spacecraft
        if pd.isnull(intercept_time):
            continue

        df_shocks.at[eventID,f'{interceptor}_time'] = intercept_time
        df_shocks.at[eventID,f'{interceptor}_time_unc_s'] = intercept_unc
        df_shocks.at[eventID,f'{interceptor}_coeff'] = intercept_coeff
        df_shocks.at[eventID,f'{interceptor}_sc'] = intercept_sc

        pos, unc = retrieve_position_unc(interceptor, speasy_variables, intercept_time, intercept_unc)

        df_shocks.loc[eventID,[f'{interceptor}_r_x_GSE',f'{interceptor}_r_y_GSE',f'{interceptor}_r_z_GSE']] = pos
        df_shocks.loc[eventID,[f'{interceptor}_r_x_GSE_unc',f'{interceptor}_r_y_GSE_unc',f'{interceptor}_r_z_GSE_unc']] = unc

        # Used to track for Cluster and THEMIS
        intercept_sc.append(interceptor)

    if len(intercept_sc)==0:
        raise Exception(f'#{eventID}: No downstream monitors recorded shock.')

    #return



def find_propagation_time(shock_time, detector, interceptor, parameter, position=None, **kwargs):


    position_var   = kwargs.get('position_var','R_GSE')
    buffer_up      = kwargs.get('buffer_up',33)
    buffer_dw      = kwargs.get('buffer_dw',40)
    resolution     = kwargs.get('resolution',None)
    intercept_pos  = kwargs.get('intercept_pos',None)

    distance_buff  = kwargs.get('distance_buff',70)
    max_neg_delay  = kwargs.get('max_neg_delay',40)
    max_pos_delay  = kwargs.get('max_neg_delay',90)

    if resolution is None:
        if set(['OMNI','ACE','IMP8']).intersection([detector,interceptor]):
            resolution = 60
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




