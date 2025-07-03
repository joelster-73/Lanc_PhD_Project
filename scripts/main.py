# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""

from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data

all_processed_shocks = import_processed_data(PROC_SHOCKS_DIR)

# %% Initialise

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from uncertainties import ufloat

from src.config import R_E

from src.analysing.shocks.intercepts import find_propagation_time
from src.analysing.shocks.in_sw import in_solar_wind

from src.processing.dataframes import add_df_units
from src.processing.speasy.config import speasy_variables
from src.processing.speasy.retrieval import retrieve_modal_omni_sc, retrieve_datum, retrieve_position_unc

position_var = 'R_GSE'
coeff_lim = 0.7

sw_monitors = ('WIND','ACE','DSC','C1','C2','C3','C4','THA','THB','THC','GEO','IMP8')
cluster_sc = ('C1','C2','C3','C4')
themis_sc  = ('THA','THB','THC')


# Dataframe to store shock times
shocks = pd.DataFrame()
new_columns = {}

new_columns['detectors'] = None
for sc in sw_monitors:

    new_columns[f'{sc}_time'] = pd.NaT
    new_columns[f'{sc}_time_unc_s'] = np.nan

    for comp in ('x','y','z'):
        new_columns[f'{sc}_r_{comp}_GSE'] = np.nan
        new_columns[f'{sc}_r_{comp}_GSE_unc'] = np.nan


new_columns['OMNI_sc'] = None
new_columns['OMNI_time'] = pd.NaT
new_columns['OMNI_time_unc_s'] = np.nan
for comp in ('x','y','z'):
    new_columns[f'OMNI_r_{comp}_GSE'] = np.nan
    new_columns[f'OMNI_r_{comp}_GSE_unc'] = np.nan


eventID_max = np.max(all_processed_shocks['eventNum'].astype(int))
eventIDs = range(1,eventID_max+1)
shocks = pd.concat([shocks, pd.DataFrame(new_columns, index=eventIDs)], axis=1)
add_df_units(shocks)

shocks.attrs['units']['detectors'] = 'STRING'
shocks.attrs['units']['OMNI_sc'] = 'STRING'

# Store info for each shock event
script_dir = os.getcwd() # change to location of script __file__
file_name = 'Processing_events.txt'
file_path = os.path.join(script_dir, file_name)
if not os.path.exists(file_path):
    with open(file_path, 'w') as my_file:
        my_file.write(f'Log created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')


# %% Process

# Each event has a row for each shock detected by Wind, ACE, DSC
for eventID, event in all_processed_shocks.groupby(lambda x: int(all_processed_shocks.loc[x, 'eventNum'])):

    if eventID<1462:
        continue

    ###-------------------INITIAL CHECKS-------------------###

    # Info for all shocks in this event
    times       = event.index.tolist()
    uncs        = event['time_unc'].tolist()
    detectors   = event['spacecraft'].tolist()
    detector_dict = dict(zip(detectors,list(zip(times,uncs))))
    max_time    = max(times)

    # Adds shocks in the database to dataframe
    shocks.at[eventID,'detectors'] = detectors
    for _, row in event.iterrows():
        sc = row['spacecraft']
        shocks.at[eventID,f'{sc}_time'] = row.name
        shocks.at[eventID,f'{sc}_time_unc_s'] = row['time_unc']

        pos, unc = retrieve_position_unc(sc, speasy_variables, row.name, row['time_unc'])

        if pos is None:
            pos = row[['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
            unc = np.array([np.nan,np.nan,np.nan])
            if np.isnan(pos[0]):
                continue

        shocks.loc[eventID,[f'{sc}_r_x_GSE',f'{sc}_r_y_GSE',f'{sc}_r_z_GSE']] = pos
        shocks.loc[eventID,[f'{sc}_r_x_GSE_unc',f'{sc}_r_y_GSE_unc',f'{sc}_r_z_GSE_unc']] = unc

    # Spacecraft OMNI used in its propagation in this time period
    start, end = min(times), max(times)+timedelta(minutes=90)
    sc_info    = retrieve_modal_omni_sc(speasy_variables, start, end, return_counts=True)
    if sc_info is None:
        event_info = f'#{eventID}: No OMNI info.'
        print(event_info)
        with open(file_path, 'a') as my_file:
            my_file.write(event_info+'\n')
        continue

    modal_sc, counts_dict = sc_info
    # if modal_sc == 'Bad Data':
    #     event_info = f'#{eventID}: No good OMNI data for event.'
    #     print(event_info)
    #     with open(file_path, 'a') as my_file:
    #         my_file.write(event_info+'\n')
    #     continue


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
        detect_pos = shocks.loc[eventID,[f'{omni_sc}_r_x_GSE',f'{omni_sc}_r_y_GSE',f'{omni_sc}_r_z_GSE']].to_numpy()

        # Approximate BSN location
        approx_time = detect_time + timedelta(seconds=((detect_pos[0]-15)*R_E/500))
        omni_pos, _ = retrieve_datum(position_var, 'OMNI', speasy_variables, approx_time, add_omni_sc=False)

        # Find time lag
        time_lag = find_propagation_time(detect_time, omni_sc, 'OMNI', 'B_mag', detect_pos, intercept_pos=omni_pos)
        if time_lag is None:
            continue

        delay, coeff = time_lag
        if coeff<=coeff_lim:
            continue

        # Found suitable lag
        lagged_unc = delay - ufloat(0,detect_unc)
        omni_time  = detect_time + timedelta(seconds=delay.n)
        omni_unc   = lagged_unc.s
        shocks.at[eventID,'OMNI_sc'] = sc

        break

    # If there are no shocks in the event recorded by a spacecraft used by OMNI
    # Use the shocks we have to find when they intercept the OMNI spacecraft
    # Currently not implemented to see sample size
    if pd.isnull(omni_time):
        event_info = f'#{eventID}: Need to interpolate to find shock in OMNI.'
        print(event_info)
        with open(file_path, 'a') as my_file:
            my_file.write(event_info+'\n')
        continue

    event_info = f'#{eventID}: Found OMNI time.'
    print(event_info)
    with open(file_path, 'a') as my_file:
        my_file.write(event_info+'\n')
    shocks.at[eventID,'OMNI_time'] = omni_time
    shocks.at[eventID,'OMNI_time_unc_s'] = omni_unc

    pos, unc = retrieve_position_unc('OMNI', speasy_variables, omni_time, omni_unc)

    shocks.loc[eventID,['OMNI_r_x_GSE','OMNI_r_y_GSE','OMNI_r_z_GSE']] = pos
    shocks.loc[eventID,['OMNI_r_x_GSE_unc','OMNI_r_y_GSE_unc','OMNI_r_z_GSE_unc']] = unc

    ###-------------------FIND WHEN SHOCKS INTERCEPT DOWNSTREAM SPACECRAFT-------------------###
    intercept_sc = []
    for interceptor in sw_monitors:
        # Don't want spacecraft that are used by OMNI - looking for those to compare with
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
        intercept_times = []
        intercept_uncs  = []
        for i, row in event.iterrows(): # All the monitors
            detector = row['spacecraft']

            detector_pos = row[['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
            detector_time = row.name

            # Find how long shocks takes to intercept spacecraft
            time_lag = find_propagation_time(detector_time, detector, interceptor, 'B_mag', detector_pos, intercept_pos=intercept_pos)
            if time_lag is None:
                continue

            delay, coeff = time_lag
            if coeff<=coeff_lim:
                continue

            lagged_unc = delay - ufloat(0,row['time_unc'])
            intercept_times.append(detector_time + timedelta(seconds=delay.n))
            intercept_uncs.append(lagged_unc.s)

        # If shock isn't found to interept the spacecraft
        if len(intercept_times)==0:
            continue
        elif len(intercept_times)==1:
            pred_time, pred_unc = intercept_times[0], intercept_uncs[0]
        else:
            # Average of times
            min_time = min(intercept_times)
            times_u = np.array([ufloat((time-min_time).total_seconds(), unc) for time, unc in zip(intercept_times,intercept_uncs)])
            avg_time = np.mean(times_u)
            pred_time = min_time + timedelta(seconds=avg_time.n)
            pred_unc = avg_time.s

        shocks.at[eventID,f'{interceptor}_time'] = pred_time
        shocks.at[eventID,f'{interceptor}_time_unc_s'] = pred_unc

        pos, unc = retrieve_position_unc(interceptor, speasy_variables, pred_time, pred_unc)

        shocks.loc[eventID,[f'{interceptor}_r_x_GSE',f'{interceptor}_r_y_GSE',f'{interceptor}_r_z_GSE']] = pos
        shocks.loc[eventID,[f'{interceptor}_r_x_GSE_unc',f'{interceptor}_r_y_GSE_unc',f'{interceptor}_r_z_GSE_unc']] = unc

        # Used to track for Cluster and THEMIS
        intercept_sc.append(interceptor)


# %%
from src.processing.writing import write_to_cdf
from src.config import PROC_INTERCEPTS_DIR

output_file = os.path.join(PROC_INTERCEPTS_DIR, 'shocks_with_intercepts.cdf')

write_to_cdf(shocks, output_file)


# %%
import warnings
warnings.filterwarnings('error')

from src.plotting.shocks import plot_time_differences, plot_time_histogram

plot_time_differences(shocks, selection='all', x_axis='x_comp', colouring='spacecraft')
plot_time_differences(shocks, selection='closest', x_axis='x_comp', colouring='spacecraft')
plot_time_differences(shocks, selection='earth', x_axis='x_comp', colouring='spacecraft')
plot_time_differences(shocks, selection='closest', x_axis='x_comp', colouring='spacecraft', max_dist=60)

plot_time_histogram(shocks, selection='all')
plot_time_histogram(shocks, selection='closest')
plot_time_histogram(shocks, selection='earth')
plot_time_histogram(shocks, selection='closest', max_dist=60)

