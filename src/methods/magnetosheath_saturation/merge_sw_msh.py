# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:21:25 2025

@author: richarj2
"""
import os
import pandas as pd

from src.config import CROSSINGS_DIR, PROC_CLUS_DIR_MSH, MSH_DIR, OMNI_DIR
from src.processing.reading import import_processed_data
from src.processing.dataframes import merge_dataframes
from src.processing.writing import write_to_cdf


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


# %% SW_MSH_Data

lag    = 17
dt_lag = pd.Timedelta(minutes=lag)
param_map = {'n_p': 'N_tot', 'V_flow': 'V_mag', 'P_flow': 'P_tot'}
param_map_pc = {k: k.replace('_sw', '_pc') for k in ['AE_sw','AL_sw','AU_sw', 'AE_17m_sw','SYM_D_sw','SYM_H_sw','ASY_D_sw','ASY_H_sw','PSI_P_10_sw','PSI_P_30_sw','PSI_P_60_sw']}


for sample_interval in ('1min','5min'):

    # Solar wind data
    omni_dir = os.path.join(OMNI_DIR,sample_interval)
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

    df_sw.columns = [param_map.get(col,col) for col in df_sw.columns]
    df_sw.attrs['units'] = {param_map.get(col,col): df_sw.attrs['units'].get(col,col) for col in df_sw.attrs['units']}

    # Magnetosheath data
    clus_dir = os.path.join(PROC_CLUS_DIR_MSH,sample_interval)
    df_msh = import_processed_data(clus_dir)

    df_merged = merge_dataframes(df_sw, df_msh, suffix_1='sw', suffix_2='msh')

    df_merged.columns = [param_map_pc.get(col,col) for col in df_merged.columns]
    df_merged.attrs['units'] = {param_map_pc.get(col,col): df_merged.attrs['units'].get(col,col) for col in df_merged.attrs['units']}

    # Filter
    interval_index = pd.IntervalIndex.from_tuples(
        [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in time_ranges],
        closed='both'
    )

    idxr = interval_index.get_indexer(df_merged.index)
    mask = idxr != -1
    df_merged = df_merged[mask]

    # Write
    output_file = os.path.join(MSH_DIR, sample_interval, f'msh_times_{sample_interval}.cdf')
    write_to_cdf(df_merged, output_file, reset_index=True)