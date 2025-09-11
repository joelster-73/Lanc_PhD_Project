# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""

# %%
import os
import numpy as np
import pandas as pd

from uncertainties import unumpy as unp

from src.config import CROSSINGS_DIR, PROC_OMNI_DIR, PROC_CLUS_DIR_MSH, PROC_CLUS_DIR_MSHA

from src.analysing.calculations import calc_average_vector, calc_mean_error
from src.analysing.coupling import kan_lee_field

from src.processing.reading import import_processed_data
from src.processing.writing import write_to_cdf


# %% Grison MSH intervals

crossings = import_processed_data(CROSSINGS_DIR)
cross_labels = crossings.attrs['crossings']
msh_times = crossings.loc[(crossings['loc_num']==10)&(crossings['region_duration']>60)].copy()
msh_times.loc[:,'end_time'] = msh_times.index + pd.to_timedelta(msh_times.loc[:,'region_duration'], unit='s')

time_ranges = [[str(start), str(end)] for start, end in zip(
        msh_times.index,
        msh_times.index + pd.to_timedelta(msh_times['region_duration'], unit='s'))]

# %% Cluster_and_OMNI

omni = import_processed_data(PROC_OMNI_DIR)
kan_lee_field(omni)

# Removing erroneous values
omni.loc[omni['AE']>5000,'AE'] = np.nan
omni.loc[omni['E_y']>25,'E_y'] = np.nan


df_msh = import_processed_data(PROC_CLUS_DIR_MSH)

# Data want to process
df_regions = {'sw': omni, 'msh': df_msh, 'pc': omni}
var_regions = {'sw': ['B_mag', 'B_GSM','n_p', 'V_flow', 'P_flow', 'E_mag', 'E_GSM', 'E_y', 'E_R', 'MA', 'S_mag'],
               'msh': ['r_GSE', 'B_mag', 'B_GSM', 'N_tot', 'V_mag', 'E_mag', 'E_GSM', 'P_tot'],
               'pc': ['AE', 'AE_17m']}
# %% Average

# from datetime import datetime

for start, interval in msh_times.iterrows():

    end = interval['end_time']

    # if start<datetime(2006,1,13):
    #     continue

    for region, df_region in df_regions.items():

        time_mask = (df_region.index>=start) & (df_region.index<end)
        if region=='msh':
            time_mask &= (df_region['r_x_GSE']>0)
        if not time_mask.any():
            continue

        df_filtered = df_region[time_mask].dropna()
        df_filtered_lag = df_region[time_mask+pd.Timedelta(minutes=17)].dropna()

        for param in var_regions[region]:
            if '_GS' in param:
                # Vector parameter
                vec, coords = param.split('_')
                param_cols = [f'{vec}_{comp}_{coords}' for comp in ('x', 'y', 'z')]
                if coords == 'GSM' and f'{vec}_x_GSM' not in df_filtered:
                    param_cols[0] = f'{vec}_x_GSE'

                if not set(param_cols).issubset(df_filtered.columns):
                    print(f'Skipping {param} in {region} due to missing columns')
                    continue

                vector = calc_average_vector(df_filtered[param_cols], param)

                # Assign results
                msh_times.loc[start, [f'{vec}_{c}_{coords}_{region}' for c in ('x','y','z')]] = unp.nominal_values(vector)
                msh_times.loc[start, [f'{vec}_{c}_{coords}_unc_{region}' for c in ('x','y','z')]] = unp.std_devs(vector)
                msh_times.at[start, f'{param}_count_{region}'] = len(df_filtered)

            else:
                # Scalar parameter
                if param == 'AE_17m':
                    param_data = df_filtered_lag['AE'].dropna()
                else:
                    param_data = df_filtered[param].dropna()

                # Map parameter names
                param_map = {'n_p': 'N_tot', 'V_flow': 'V_mag', 'P_flow': 'P_tot'}
                param_name = param_map.get(param, param)

                if not param_data.empty:
                    mean = calc_mean_error(param_data)
                    msh_times.at[start, f'{param_name}_{region}'] = mean.n
                    msh_times.at[start, f'{param_name}_unc_{region}'] = mean.s
                    msh_times.at[start, f'{param_name}_count_{region}'] = len(param_data)
                else:
                    msh_times.at[start, f'{param_name}_count_{region}'] = 0

                    # Account for omni's error

# %%
output_file = os.path.join(PROC_CLUS_DIR_MSHA, 'msh_times.cdf')
write_to_cdf(msh_times, output_file)


