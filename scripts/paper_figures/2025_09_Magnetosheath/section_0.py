# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %%
import numpy as np

from src.analysing.coupling import kan_lee_field

from src.config import PROC_OMNI_DIR, PROC_CLUS_DIR_MSH, PROC_CLUS_DIR_MSHA
from src.processing.reading import import_processed_data


# %% Imports
df_sw  = import_processed_data(PROC_OMNI_DIR)
kan_lee_field(df_sw)

# Removing erroneous values
df_sw.loc[df_sw['AE']>5000,'AE'] = np.nan
df_sw.loc[df_sw['E_y']>20,'E_y'] = np.nan

df_msh = import_processed_data(PROC_CLUS_DIR_MSH)

msh_times = import_processed_data(PROC_CLUS_DIR_MSHA)


### Should create density vs field ratio plots like Nugyen to confirm msh dataset
### Look at relative strength in terms of E_Y = (-V x B)_Y
### Do histograms of parameters in raw databases and then in MSH times
### Check whether intervals are correct and for outliers
### Consider plotting every interval sw and msh parameter

# %% Pulkkinen

from src.plotting.comparing.parameter import compare_columns

# Also include beta MA plot
for param in ('B_avg','B_z_GSM','B_clock','E_y_GSM','V_mag','N_tot','S_mag','V_x_GSM'):

    sw_param  = f'{param}_sw'
    msh_param = f'{param}_msh'
    sw_err    = f'{param}_unc_sw'
    msh_err   = f'{param}_unc_msh'
    sw_count  = f'{param}_count_sw'
    msh_count = f'{param}_count_msh'

    if '_GS' in param:
        field, _, coords = param.split('_')
        sw_count  = f'{field}_{coords}_count_sw'
        msh_count = f'{field}_{coords}_count_msh'

    if 'V_' in param:
        step = 50
    elif 'B_' in param:
        step = 2.5
    else:
        step = 1

    compare_columns(msh_times, sw_param, msh_param, col1_err=sw_err, col2_err=msh_err, col3='MA_sw', display='scatter', scatter_size=20, z_max=12)

    #compare_columns(msh_times, sw_param, msh_param, col1_err=sw_err, col2_err=msh_err, col3='MA_sw', display='scatter_binned', col1_counts=sw_count, col2_counts=msh_count, scatter_size=20, bin_step=step, z_max=12, fit_type='straight')


# %%
for param in ('E_y_GSM','E_R'):
    sw_param  = f'{param}_sw'
    msh_param = f'{param}_msh'
    sw_err    = f'{param}_unc_sw'
    msh_err   = f'{param}_unc_msh'

    if '_GS' in param:
        field, _, coords = param.split('_')
        sw_count  = f'{field}_{coords}_count_sw'
        msh_count = f'{field}_{coords}_count_msh'

    step = 1

    mask = np.ones(len(msh_times),dtype=bool)

    if param=='E_y_GSM':
        mask &= (msh_times.loc[:, sw_param]>0) & (msh_times.loc[:, msh_param]>0) & (np.abs(msh_times.loc[:, msh_param])<1e6)

    compare_columns(msh_times.loc[mask], sw_param, 'AE_17m_pc', col1_err=sw_err, col2_err='AE_17m_unc_pc', col3='MA_sw', col1_counts=sw_count, col2_counts='AE_17m_count_pc', display='scatter_binned', scatter_size=20, bin_step=step)

    compare_columns(msh_times.loc[mask], msh_param, 'AE_pc', col1_err=msh_err, col2_err='AE_unc_pc', col3='MA_sw', col1_counts=sw_count, col2_counts='AE_count_pc', display='scatter_binned', scatter_size=20, bin_step=step)


# %%
for i in range(3):

    for couple in ('E_y_GSM','E_R'):

        mask = np.ones(len(df_sw),dtype=bool)
        if i>=1:
            mask &= df_sw[couple]>0
        if i>=2:
            mask &= df_sw[couple]<16

        for z_param in ('MA','MA_med','V_flow'):

            if z_param=='V_flow':
                edges = [400,500,600,700,800]
            elif z_param=='MA':
                edges = [5]
            elif z_param=='MA_med':
                edges = [9.2]
                z_param = 'MA'

            compare_columns(df_sw[mask], couple, 'AE', col3=z_param, delay_time=17, display='scatter_binned_multiple', scatter_size=20, zs_edges=edges)


# Create similar plots looking at saturation against Electric field and how velocity and magnetic field affect
