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


# %%
from src.plotting.comparing.parameter import compare_columns

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

# %%


for param in ('E_y_GSM','B_mag','V_mag'):
    sw_param  = f'{param}_sw'
    msh_param = f'{param}_msh'
    sw_err    = f'{param}_unc_sw'
    msh_err   = f'{param}_unc_msh'
    sw_count  = f'{param}_count_sw'
    msh_count = f'{param}_count_msh'

    mask = np.ones(len(msh_times),dtype=bool)

    if param=='E_y_GSM':
        mask &= (msh_times.loc[:, sw_param]>0) & (msh_times.loc[:, msh_param]>0) & (np.abs(msh_times.loc[:, msh_param])<1e6)

    compare_columns(msh_times.loc[mask], sw_param, msh_param, col1_err=sw_err, col2_err=msh_err, col3='MA_sw', xs_counts=sw_count, ys_count=msh_count, display='scatter_binned', scatter_size=20)

    compare_columns(msh_times.loc[mask], sw_param, 'AE_17m_pc', col1_err=sw_err, col2_err='AE_17m_unc_pc', col3='MA_sw', xs_counts=sw_count, ys_count='AE_17m_count_pc', display='scatter_binned', scatter_size=20)

    compare_columns(msh_times.loc[mask], msh_param, 'AE_pc', col1_err=msh_err, col2_err='AE_unc_pc', col3='MA_sw', xs_counts=sw_count, ys_count='AE_count_pc', display='scatter_binned', scatter_size=20)

