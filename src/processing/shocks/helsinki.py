# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""

import os
import pandas as pd
import numpy as np
from ..utils import add_unit

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
