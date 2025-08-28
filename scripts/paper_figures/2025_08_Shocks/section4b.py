# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 10:30:06 2025

@author: richarj2
"""
# %% Import

from src.config import HELSINKI_DIR, PROC_CFA_DIR, PROC_INTERCEPTS_DIR
from src.processing.reading import import_processed_data

shocks_with_intercepts = import_processed_data(PROC_INTERCEPTS_DIR)

# %% Scatter_and_hist
from src.methods.shock_intercepts.plotting import plot_time_differences

plot_time_differences(shocks_with_intercepts, selection='all', x_axis='delta_x', colouring='spacecraft')
plot_time_differences(shocks_with_intercepts, selection='earth', x_axis='delta_x', colouring='spacecraft')

# %% Normals

import pandas as pd

helsinki_with_normals = import_processed_data(HELSINKI_DIR)
helsinki_with_normals = helsinki_with_normals.loc[:,['spacecraft', 'r_x_GSE', 'r_y_GSE', 'r_z_GSE', 'res_B', 'res_p', 'N_x_GSE', 'N_x_GSE_unc', 'N_y_GSE', 'N_y_GSE_unc', 'N_z_GSE', 'N_z_GSE_unc','v_sh', 'v_sh_unc']]
helsinki_with_normals.rename(columns={f'N_{comp}_GSE':f'N{comp}' for comp in ('x','y','z')},inplace=True)
helsinki_with_normals.rename(columns={f'N_{comp}_GSE_unc':f'N{comp}_unc' for comp in ('x','y','z')},inplace=True)
helsinki_with_normals['time_unc'] = helsinki_with_normals.apply(lambda row: 0.5*max(row['res_B'], row['res_p']), axis=1)
helsinki_with_normals.drop(columns=['res_B','res_p'],inplace=True)
helsinki_with_normals['source'] = 'H'

cfa_shocks = import_processed_data(PROC_CFA_DIR)
cfa_shocks = cfa_shocks.loc[:,['time_s_unc', 'spacecraft', 'r_x_GSE', 'r_y_GSE', 'r_z_GSE', 'Nx', 'Nx_unc', 'Ny', 'Ny_unc', 'Nz', 'Nz_unc', 'v_sh', 'v_sh_unc']]
cfa_shocks.rename(columns={'time_s_unc':'time_unc'},inplace=True)
cfa_shocks['spacecraft'] = cfa_shocks['spacecraft'].str.upper()
cfa_shocks = cfa_shocks.loc[
    ~(cfa_shocks['spacecraft'] == cfa_shocks['spacecraft'].shift()) |
    (cfa_shocks.index.to_series().diff() > pd.Timedelta(minutes=5))
]
cfa_shocks['source'] = 'C'

shocks_with_normals = pd.concat([helsinki_with_normals,cfa_shocks]).sort_index()



# %% With_normals

plot_time_differences(shocks_with_intercepts, bottom_panel=None, selection='earth', x_axis='delta_n', colouring='spacecraft', shock_normals=shocks_with_normals)

plot_time_differences(shocks_with_intercepts, bottom_panel=None, selection='earth', x_axis='delta_t', colouring='spacecraft', shock_normals=shocks_with_normals)











