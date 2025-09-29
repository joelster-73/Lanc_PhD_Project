# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:21:25 2025

@author: richarj2
"""
import os
import pandas as pd

from src.config import CROSSINGS_DIR, PROC_CLUS_DIR_MSH, PROC_CLUS_DIR_5VPS, MSH_DIR, OMNI_DIR, PROC_THEMIS_DIR
from src.processing.reading import import_processed_data
from src.processing.dataframes import merge_dataframes
from src.processing.writing import write_to_cdf
from src.coordinates.boundaries import calc_msh_r_diff

column_names = {
    'r_x_name'  : 'r_x_GSE',
    'r_y_name'  : 'r_y_GSE',
    'r_z_name'  : 'r_z_GSE',
    'r_name'    : 'r_mag',
    'r_ax_name' : 'r_x_aGSE',
    'r_ay_name' : 'r_y_aGSE',
    'r_az_name' : 'r_z_aGSE',
    'v_x_name'  : 'V_x_GSE',
    'v_y_name'  : 'V_y_GSE',
    'v_z_name'  : 'V_z_GSE',
    'p_name'    : 'P_flow'
}


# %% Grison_MSH_Times

crossings = import_processed_data(CROSSINGS_DIR)
cross_labels = crossings.attrs['crossings']

mask = (crossings['loc_num']==10) & (crossings['quality_num']>=2)

msh_times = crossings.loc[mask].copy()
msh_times.loc[:,'end_time'] = msh_times.index + pd.to_timedelta(msh_times.loc[:,'region_duration'], unit='s')
msh_times.drop(columns=['loc_num','quality_num','complexity_num'], inplace=True)

msh_times['new_group'] = (msh_times.index != msh_times['end_time'].shift()).cumsum()

# Combine consecutive groups
result = msh_times.groupby('new_group').agg(
    start_time=('end_time', lambda x: msh_times.loc[x.index[0]].name),
    region_duration=('region_duration', 'sum'),
    end_time=('end_time', 'last')
).set_index('start_time')

msh_times = result.loc[result['region_duration']>60]

time_ranges = [[str(start), str(end)] for start, end in zip(
        msh_times.index,
        msh_times.index + pd.to_timedelta(msh_times['region_duration'], unit='s'))]

# %% OMNI_with_lag

sample_intervals = ('1min','5min')

lag    = 17
dt_lag = pd.Timedelta(minutes=lag)

for sample_interval in sample_intervals:

    # Solar wind data
    omni_dir = os.path.join(OMNI_DIR, sample_interval)
    df_sw    = import_processed_data(omni_dir)

    # Lagged AE
    if (dt_lag % pd.Timedelta(sample_interval)) == pd.Timedelta(0):

        df_sw[f'AE_{lag}m'] = df_sw['AE'].shift(freq=dt_lag)

    else:
        print('Interpolating lag.')

        target_index = df_sw.index + dt_lag
        full_index = df_sw.index.union(target_index)
        temp = df_sw['AE'].reindex(full_index)
        temp = temp.interpolate(method='time')
        df_sw[f'AE_{lag}m'] = temp.loc[target_index].values

    # Writes OMNI with lag to file
    output_file = os.path.join(OMNI_DIR, 'with_lag', f'omni_{sample_interval}.cdf')
    write_to_cdf(df_sw, output_file, reset_index=True)


# %% Combined

# With Plasma
data_pop = 'with_plasma'
cluster_directory = PROC_CLUS_DIR_MSH

# Field only
data_pop = 'field_only'
cluster_directory = PROC_CLUS_DIR_5VPS
themis_directory  = PROC_THEMIS_DIR


sample_intervals = ('1min','5min')
msh_keys = ('c1','tha','thb','thc','thd','the')
pos_cols = ['r_MP','r_BS','r_phi','r_F']

param_map_pc = {k: k.replace('_sw', '_pc') for k in ['AE_sw','AL_sw','AU_sw', 'AE_17m_sw','SYM_D_sw','SYM_H_sw','ASY_D_sw','ASY_H_sw','PSI_P_10_sw','PSI_P_30_sw','PSI_P_60_sw']}

for sample_interval in sample_intervals:

    # Solar wind data
    omni_dir = os.path.join(OMNI_DIR, 'with_lag')
    df_sw    = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

    dfs_combined = []

    for spacecraft in msh_keys:

        if spacecraft in ('c1',):
            sc_dir = cluster_directory
        elif spacecraft in ('tha','thb','thc','thd','the'):
            sc_dir = os.path.join(themis_directory, spacecraft)

        sc_dir = os.path.join(sc_dir, sample_interval)

        # Magnetosheath data
        df_msh = import_processed_data(sc_dir)
        df_msh = df_msh.loc[df_msh['r_x_GSE']>0]
        df_msh.dropna(subset=['B_avg'],inplace=True)

        # Combine
        df_merged = merge_dataframes(df_sw, df_msh, suffix_1='sw', suffix_2=spacecraft)
        df_merged.columns = [param_map_pc.get(col,col) for col in df_merged.columns] # changes sw to pc for some omni
        df_merged.attrs['units'] = {param_map_pc.get(col,col): df_merged.attrs['units'].get(col,col) for col in df_merged.attrs['units']}
        df_merged_attrs = df_merged.attrs # stores

        # Filter for MSH
        df_positions = calc_msh_r_diff(df_merged, 'BOTH', position_key=spacecraft, data_key='sw', column_names=column_names)
        df_merged = pd.concat([df_merged,df_positions[pos_cols]],axis=1)

        if spacecraft=='c1': # Grison
            interval_index = pd.IntervalIndex.from_tuples(
                [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in time_ranges],
                closed='both'
            )
            df_merged = df_merged.loc[interval_index.get_indexer(df_merged.index) != -1]

        else: # Position
            df_merged = df_merged.loc[(df_merged['r_F']>0)&(df_merged['r_F']<1)]

        df_merged.rename(columns={col: f'{col}_{spacecraft}' for col in pos_cols}, inplace=True) # adds _sc suffix
        df_merged.dropna(how='all',inplace=True)

        # Adds only sc to list to combine
        dfs_combined.append(df_merged[[col for col in df_merged if f'_{spacecraft}' in col]])

        # Writes individual to file with omni
        df_merged.attrs = df_merged_attrs
        output_file = os.path.join(MSH_DIR, data_pop, sample_interval, f'msh_times_{spacecraft}.cdf')
        write_to_cdf(df_merged, output_file, reset_index=True)

        print(spacecraft,'written to file')

    df_combined = pd.concat(dfs_combined, axis=1)
    df_combined.attrs = df_merged_attrs

    df_combined = merge_dataframes(df_sw, df_combined, suffix_1='sw', suffix_2=None)
    df_combined.columns = [param_map_pc.get(col,col) for col in df_combined.columns]
    df_combined.attrs['units'] = {param_map_pc.get(col,col): df_combined.attrs['units'].get(col,col) for col in df_combined.attrs['units']}
    df_combined.attrs = df_merged_attrs

    # Write
    output_file = os.path.join(MSH_DIR, data_pop, sample_interval, 'msh_times_combined.cdf')
    write_to_cdf(df_combined, output_file, reset_index=True)

    print('combined written to file')