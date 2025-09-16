# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %%
import numpy as np

from src.config import PROC_CLUS_OMNI_MSH5
from src.processing.reading import import_processed_data

from src.plotting.comparing.parameter import compare_columns, plot_nguyen



df_merged = import_processed_data(PROC_CLUS_OMNI_MSH5)



### Look at relative strength in terms of E_Y = (-V x B)_Y
### Do histograms of parameters in raw databases and then in MSH times
### Consider plotting every interval sw and msh parameter

### HERE DO LOOP FOR INDEPENDENT/DEPENDENT RELTIOSN
### SO AE BUT ALSO SATURATION IN MSH BASED ON FACTOR E.G. CLOCK ANGLE
### THEN CREATE FIGURES USING ENTIRE OMNI/CLUSTER DATASETS

# Create similar plots looking at saturation against Electric field and how velocity and magnetic field affect


# %% Nguyen

plot_nguyen(df_merged, 'N_tot', 'B_avg')


# %% Pulkkinen

rules = {'B_avg':   (2, (1, 5)),
         'B_z_GSM': (2, (1, 5)),
         'B_clock': (np.pi/5, np.pi/36),
         'E_y_GSM': (1, (0.5, 1)),
         'V_mag':   (50, 5),
         'N_tot':   (1, (1, 5)),
         'S_mag':   (10, (4, 100))
         }

# Also include beta MA plot
for param in ('B_avg','B_z_GSM','B_clock','E_y_GSM','V_mag','N_tot','S_mag'):

    if param!='B_z_GSM':
        continue

    sw_param  = f'{param}_sw'

    # Include sw error if B else ignore

    msh_param = f'{param}_msh'
    msh_err   = f'{param}_unc_msh'
    msh_count = f'{param}_count_msh'


    step, bin_width = rules.get(param,(1, (1, 5)))

    compare_columns(df_merged, sw_param, msh_param, display='heat', bin_width=bin_width)

    if '_GS' in param:
        field, _, coords = param.split('_')
        msh_count = f'{field}_{coords}_count_msh'

    compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='MA_sw', display='scatter', scatter_size=20, z_max=12, zero_lines=(param=='B_z_GSM'))

    if param=='B_z_GSM':
        compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='region_duration', display='scatter', scatter_size=20, z_max=60000, data3_name='region duration', zero_lines=(param=='B_z_GSM'))

    compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='beta_msh', display='scatter_binned', col2_counts=msh_count, scatter_size=20, bin_step=step, z_min=0, z_max=2)

# %% beta

# Different format
ma   = 'MA'
beta = 'beta'

sw_param  = f'{ma}_sw'

msh_param = f'{beta}_msh'
msh_err   = f'{beta}_unc_msh'
msh_count = f'{beta}_count_msh'


### NEED TO CLEAN UP THE MSH DATA FOR ANOMALOUS BETA THEN



compare_columns(df_merged[(df_merged['beta_msh']<50)&(df_merged['MA_sw']<60)], sw_param, msh_param, display='heat', bin_width=1)

compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='E_y_GSM_sw', display='scatter', scatter_size=20, z_min=0, z_max=8)

compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='E_y_GSM_sw', display='scatter_binned', col2_counts=msh_count, scatter_size=20, bin_step=1, z_min=0, z_max=8)

# %% Correlation

for param in ('E_y_GSM','E_R'):
    sw_param  = f'{param}_sw'
    msh_param = f'{param}_msh'
    msh_err   = f'{param}_unc_msh'

    if '_GS' in param:
        field, _, coords = param.split('_')
        msh_count = f'{field}_{coords}_count_msh'

    step = 1

    mask = np.ones(len(df_merged),dtype=bool)

    if param=='E_y_GSM':
        mask &= (df_merged.loc[:, sw_param]>0) & (df_merged.loc[:, msh_param]>0) & (np.abs(df_merged.loc[:, msh_param])<1e6)

    compare_columns(df_merged.loc[mask], sw_param, 'AE_17m_pc', col3='MA_sw', display='scatter_binned', scatter_size=20, bin_step=step, z_max=8)

    compare_columns(df_merged.loc[mask], msh_param, 'AE_pc', col1_err=msh_err, col3='MA_sw', display='scatter_binned', scatter_size=20, bin_step=step)


# %%
for i in range(3):

    for couple in ('E_y_GSM_sw','B_z_GSM_sw'):

        mask = np.ones(len(df_merged),dtype=bool)
        if i>=1:
            if 'E_' in couple:
                mask &= df_merged[couple]>0
            elif 'B_' in couple:
                mask &= df_merged[couple]<0

        if i>=2:
            if 'E_' in couple:
                mask &= df_merged[couple]<16
            elif 'B_' in couple:
                mask &= df_merged[couple]>-20

        for z_param in ('MA_sw','MA_med','beta_msh'):


            if z_param=='V_mag':
                edges = [400,500,600,700,800]
            elif z_param=='MA_sw':
                edges = [5]
            elif z_param=='beta_msh':
                edges = [1]
            elif z_param=='MA_med':
                edges = [9.2]
                z_param = 'MA_sw'

            compare_columns(df_merged[mask], couple, 'AE_17m_pc', col3=z_param, display='scatter_binned_multiple', scatter_size=20, zs_edges=edges)

