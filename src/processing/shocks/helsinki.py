# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""

import os
import pandas as pd
import numpy as np
from ..utils import add_unit

from ..speasy.retrieval import retrieve_omni_value
from ..speasy.config import sw_monitors

from collections import Counter
from ..speasy.retrieval import retrieve_position_unc


# %% Importing
def process_helsinki_shocks(directory, file_name, time_col='epoch'):

    data_file = os.path.join(directory,file_name)


    headers = ['year','month','day','hour','minute','second','spacecraft',
               'r_x','r_y','r_z','pos_system','type',
               'B_mag_up','B_mag_up_unc','B_x_up','B_x_up_unc','B_y_up','B_y_up_unc','B_z_up','B_z_up_unc',
               'B_mag_dw','B_mag_dw_unc','B_x_dw','B_x_dw_unc','B_y_dw','B_y_dw_unc','B_z_dw','B_z_dw_unc',
               'B_ratio','B_ratio_unc',
               'V_flow_up','V_flow_up_unc','V_x_up','V_x_up_unc','V_y_up','V_y_up_unc','V_z_up','V_z_up_unc',
               'V_flow_dw','V_flow_dw_unc','V_x_dw','V_x_dw_unc','V_y_dw','V_y_dw_unc','V_z_dw','V_z_dw_unc',
               'V_jump','V_jump_unc',
               'n_p_up','n_p_up_unc','n_p_dw','n_p_dw_unc','n_p_ratio','n_p_ratio_unc',
               'T_p_up','T_p_up_unc','T_p_dw','T_p_dw_unc','T_p_ratio','T_p_ratio_unc',
               'V_cs_up','V_cs_up_unc','V_A_up','V_A_up_unc','V_ms_up','V_ms_up_unc','Beta_up','Beta_up_unc',
               'N_x','N_x_unc','N_y','N_y_unc','N_z','N_z_unc','normal_system',
               'theta_Bn','theta_Bn_unc','v_sh','v_sh_unc','M_A','M_A_unc','M_ms','M_ms_unc',
               'radial_vel','interval','res_B','res_p']


    df  = pd.read_csv(
        data_file,
        sep=',',
        skiprows=84,
        header=None
    )
    df.columns = headers
    df.attrs = {}
    df['spacecraft'] = df['spacecraft'].str.strip()

    df[time_col] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
    df.set_index(time_col,inplace=True)
    df.sort_index(inplace=True)

    df = df.loc[
        ~(df['spacecraft'] == df['spacecraft'].shift()) |
        (df.index.to_series().diff() > pd.Timedelta(minutes=1))
    ]

    df.drop(columns=['year','month','day','hour','minute','second'],inplace=True)

    for col in ('T_p_up','T_p_up_unc','T_p_dw','T_p_dw_unc'):
        df[col] *= 1000 # converts to K from 10^4K

    for col in ('theta_Bn','theta_Bn_unc'):
        df[col] = np.deg2rad(df[col]) # converts to radians

    df['interval'] *= 60 # converts to s

    df = df[df['spacecraft'].isin(['ACE', 'OMNI', 'Wind', 'DSCOVR', 'Cluster-1', 'Cluster-3', 'Cluster-4'])]
    df['spacecraft'] = df['spacecraft'].replace({'Wind': 'WIND', 'DSCOVR': 'DSC', 'Cluster-1': 'C1', 'Cluster-3': 'C3', 'Cluster-4': 'C4'})

    position_coordinates = np.unique(df['pos_system'])
    if len(position_coordinates) == 1:
        system = position_coordinates[0].strip()
        df.rename(columns={f'r_{comp}': f'r_{comp}_{system}' for comp in ('x','y','z')},inplace=True)
        df.drop(columns=['pos_system'],inplace=True)

    normal_coordinates = np.unique(df['normal_system'])
    if len(normal_coordinates) == 1:
        system = normal_coordinates[0].strip()
        columns_to_rename = {}
        for comp in ('x','y','z'):
            columns_to_rename[f'N_{comp}'] = f'N_{comp}_{system}'
            columns_to_rename[f'N_{comp}_unc'] = f'N_{comp}_{system}_unc'
            for vec in ('B','V'):
                columns_to_rename[f'{vec}_{comp}_up']     = f'{vec}_{comp}_{system}_up'
                columns_to_rename[f'{vec}_{comp}_dw']     = f'{vec}_{comp}_{system}_dw'
                columns_to_rename[f'{vec}_{comp}_up_unc'] = f'{vec}_{comp}_{system}_up_unc'
                columns_to_rename[f'{vec}_{comp}_dw_unc'] = f'{vec}_{comp}_{system}_dw_unc'

        df.rename(columns=columns_to_rename,inplace=True)

        df.drop(columns=['normal_system'],inplace=True)

    radial_vels = np.unique(df['radial_vel'])
    if len(radial_vels) == 1:
        df.drop(columns=['radial_vel'],inplace=True)

    unit_attrs = {}
    for column in df:
        unit_attrs[column] = add_unit(column)
    df.attrs['units'] = unit_attrs

    for col in ('interval','res_B','res_p'):
        df.attrs['units'][col] = 's'

    return df

# %%

def convert_helsinki_df_training(df_helsinki):

    helsinki_shocks = df_helsinki[['spacecraft','r_x_GSE','r_y_GSE','r_z_GSE','res_B','res_p']]
    event_list = get_list_of_events_helsinki(helsinki_shocks)

    key_counts = Counter()
    for d in event_list:
        key_counts.update(d.keys())

    columns = ['epoch','eventNum','time_unc','spacecraft','r_x_GSE','r_y_GSE','r_z_GSE']
    event_shocks = pd.DataFrame(columns=columns)

    for i, event_dict in enumerate(event_list):
        for sc, info in event_dict.items():
            time = info[0]
            time_unc = info[1]
            position = helsinki_shocks.loc[time,['r_x_GSE','r_y_GSE','r_z_GSE']]
            if isinstance(position,pd.DataFrame):
                position = position.iloc[0].to_numpy()
            else:
                position = position.to_numpy()

            rad_dist = np.linalg.norm(position)
            if np.isnan(rad_dist):
                position, _ = retrieve_position_unc(sc, time, time_unc)
                if position is None:
                    continue
            event_shocks.loc[len(event_shocks)] = [time, str(i+1), time_unc, sc, position[0], position[1], position[2]]

    event_shocks.set_index('epoch',inplace=True)

    return event_shocks

# %%

def convert_helsinki_df_plotting(helsinki_shocks):
    event_list = get_list_of_events_helsinki(helsinki_shocks)

    unique_spacecraft = list({key for d in event_list for key in d})
    all_spacecraft = [sc for sc in sw_monitors if sc in unique_spacecraft]

    helsinki_events = pd.DataFrame()
    new_columns = {}

    new_columns['detectors'] = ''
    for sc in (all_spacecraft + ['OMNI']):

        new_columns[f'{sc}_time'] = pd.NaT
        new_columns[f'{sc}_time_unc_s'] = np.nan
        new_columns[f'{sc}_coeff']      = np.nan
        new_columns[f'{sc}_sc']         = ''

        for comp in ('x','y','z'):
            new_columns[f'{sc}_r_{comp}_GSE'] = np.nan
            new_columns[f'{sc}_r_{comp}_GSE_unc'] = np.nan

    eventIDs = range(1,len(event_list)+1)
    helsinki_events = pd.concat([helsinki_events, pd.DataFrame(new_columns, index=eventIDs)], axis=1)


    for ind, shock_dict in enumerate(event_list):

        event_num = ind+1

        detectors = [sc for sc in (sw_monitors + ('OMNI',)) if sc in list(shock_dict.keys())]
        #helsinki_events.at[event_num,'detectors'] = ','.join(detectors)
        helsinki_events.at[event_num,'detectors'] = detectors

        for sc, sc_info in shock_dict.items():
            helsinki_events.at[event_num, f'{sc}_time'] = sc_info[0]
            helsinki_events.at[event_num, f'{sc}_time_unc_s'] = sc_info[1]
            position = helsinki_shocks.loc[sc_info[0],['r_x_GSE','r_y_GSE','r_z_GSE']]
            if isinstance(position, pd.DataFrame):
                position = position.iloc[0].to_numpy()
            else:
                position = position.to_numpy()
            if np.isnan(position[0]):
                continue
            helsinki_events.loc[event_num,[f'{sc}_r_x_GSE',f'{sc}_r_y_GSE',f'{sc}_r_z_GSE']] = position
            if sc=='OMNI':
                helsinki_events.at[event_num,'OMNI_sc'] = retrieve_omni_value(sc_info[0], omni_var='OMNI_sc')


    return helsinki_events


# %%
def get_list_of_events_helsinki(df_shocks,reverse=False):

    event_list = []
    iterator = df_shocks.iterrows()

    prev_index, prev_shock = next(iterator)
    prev_sc     = prev_shock['spacecraft']
    prev_unc    = 0.5*max(prev_shock[['res_p','res_B']].to_numpy())
    event_dict  = {prev_sc: (prev_index,prev_unc)}

    while True:

        try:
            index, shock = next(iterator)
            sc     = shock['spacecraft']
            unc    = 0.5*max(shock[['res_p','res_B']].to_numpy())

            time_diff = (index-prev_index).total_seconds()

            if sc in event_dict or time_diff>=(90*60):
                event_list.append(event_dict)
                event_dict = {sc: (index,unc)}
                prev_index = index
            else:
                event_dict[sc] = (index,unc)

        except StopIteration:
            break

    if reverse:
        return event_list[::-1]

    return event_list

# %%

def get_list_of_events_all(df_shocks,reverse=False):

    event_list = []
    iterator = df_shocks.iterrows()

    prev_index, prev_shock = next(iterator)
    prev_sc     = prev_shock['spacecraft']
    prev_source = prev_shock['source']
    prev_unc    = prev_shock['time_unc']
    event_dict  = {prev_sc: (prev_index,prev_unc,prev_source)}

    while True:
        new_event = False
        try:
            index, shock = next(iterator)
            sc     = shock['spacecraft']
            source = shock['source']
            unc    = shock['time_unc']

            if (index-prev_index).total_seconds()>=(90*60):
                new_event = True

            elif sc in event_dict:
                if (index-event_dict[sc][0]).total_seconds()>=300:
                    new_event = True

                elif unc<event_dict[sc][1]:
                    event_dict[sc] = (index,unc,source)

            else:
                event_dict[sc] = (index,unc,source)

            if new_event:
                event_list.append(event_dict)
                event_dict = {sc: (index,unc,source)}
                prev_index = index


        except StopIteration:
            break

    if reverse:
        return event_list[::-1]

    return event_list