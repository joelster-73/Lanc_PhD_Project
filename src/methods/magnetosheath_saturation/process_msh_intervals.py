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


# %% Grison

crossings = import_processed_data(CROSSINGS_DIR)
cross_labels = crossings.attrs['crossings']
msh_times = crossings.loc[(crossings['loc_num']==10)&(crossings['region_duration']>60)].copy()
msh_times.loc[:,'end_time'] = msh_times.index + pd.to_timedelta(msh_times.loc[:,'region_duration'], unit='s')

time_ranges = [[str(start), str(end)] for start, end in zip(
        msh_times.index,
        msh_times.index + pd.to_timedelta(msh_times['region_duration'], unit='s'))]

# %% Imprts

omni = import_processed_data(PROC_OMNI_DIR)
kan_lee_field(omni)

cluster = import_processed_data(PROC_CLUS_DIR_MSH)

# Removing erroneous values
omni.loc[omni['AE']>5000,'AE'] = np.nan
omni.loc[omni['E_y']>25,'E_y'] = np.nan

cluster.loc[cluster['B_z_GSM']<-100,'B_z_GSM'] = np.nan
cluster.loc[cluster['B_avg']>200,'B_avg'] = np.nan
cluster.loc[cluster['E_y_GSM']<-100,'E_y_GSM'] = np.nan
cluster.loc[cluster['V_mag']>2e3,'V_mag'] = np.nan
cluster.loc[cluster['N_tot']>500,'N_tot'] = np.nan
cluster.loc[cluster['S_mag']>1e4,'S_mag'] = np.nan

lag = 17 # sw to pc lag

# Data want to process
df_regions = {'sw': omni, 'msh': cluster, 'pc': omni}
var_regions = {'sw': ['B_mag', 'B_GSM', 'B_clock', 'n_p', 'V_flow', 'P_flow', 'E_mag', 'E_GSM', 'E_y', 'E_R', 'MA', 'S_mag'],
               'msh': ['r_GSE', 'B_avg', 'B_GSM', 'B_clock', 'N_tot', 'V_mag', 'E_mag', 'E_GSM', 'P_tot', 'S_mag', 'beta'],
               'pc': ['AE', f'AE_{lag}m']}

# %% Average

# from datetime import datetime
time_lag = pd.Timedelta(minutes=lag)

for start, interval in msh_times.iterrows():

    end = interval['end_time']

    # if start<datetime(2006,1,13):
    #     continue

    for region, df_region in df_regions.items():

        time_mask = (df_region.index>=start) & (df_region.index<end)
        if region=='msh':
            time_mask &= (df_region['r_x_GSE']>0)
        if np.sum(time_mask)==0:
            continue

        df_filtered = df_region.loc[time_mask]

        time_mask_lag = (df_region.index >= (start + time_lag)) & (df_region.index < (end + time_lag))
        if region=='msh':
            time_mask_lag &= (df_region['r_x_GSE']>0)
        df_filtered_lag = df_region.loc[time_mask_lag]

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

                vector = calc_average_vector(df_filtered.loc[:,param_cols].dropna(), param)
                if len(vector)==0:
                    continue

                # Assign results
                msh_times.loc[start, [f'{vec}_{comp}_{coords}_{region}' for comp in ('x','y','z')]] = unp.nominal_values(vector)
                msh_times.loc[start, [f'{vec}_{comp}_{coords}_unc_{region}' for comp in ('x','y','z')]] = unp.std_devs(vector)
                msh_times.at[start, f'{param}_count_{region}'] = len(df_filtered)

            else:
                # Scalar parameter
                if param == f'AE_{lag}m':
                    param_data = df_filtered_lag.loc[:,'AE'].dropna()
                else:
                    param_data = df_filtered.loc[:,param].dropna()

                # Map parameter names
                param_map = {'n_p': 'N_tot', 'V_flow': 'V_mag', 'P_flow': 'P_tot', 'B_mag': 'B_avg'}
                param_name = param_map.get(param, param)

                if not param_data.empty:
                    mean = calc_mean_error(param_data)
                    msh_times.at[start, f'{param_name}_{region}'] = mean.n
                    msh_times.at[start, f'{param_name}_unc_{region}'] = mean.s
                    msh_times.at[start, f'{param_name}_count_{region}'] = len(param_data)
                else:
                    msh_times.at[start, f'{param_name}_count_{region}'] = 0

                    # Account for omni's error

# %% Write
output_file = os.path.join(PROC_CLUS_DIR_MSHA, 'msh_times.cdf')
write_to_cdf(msh_times, output_file, reset_index=True)

# %%
