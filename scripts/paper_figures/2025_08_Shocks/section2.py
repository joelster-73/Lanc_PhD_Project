# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 10:32:24 2025

@author: richarj2
"""

from src.config import HELSINKI_DIR
from src.processing.reading import import_processed_data
from src.processing.shocks.helsinki import convert_helsinki_df_plotting

# Need to reprocess data as getting stuck at #1402


helsinki_shocks = import_processed_data(HELSINKI_DIR)
helsinki_events = convert_helsinki_df_plotting(helsinki_shocks)

# %% Scatter_and_hist

from src.methods.shock_intercepts.plotting import plot_time_differences

plot_time_differences(helsinki_events, selection='all', x_axis='delta_x', colouring='spacecraft', right_fit=None, bottom_fit=None)


# %% Normals

helsinki_with_normals = helsinki_shocks.loc[:,['spacecraft', 'r_x_GSE', 'r_y_GSE', 'r_z_GSE', 'res_B', 'res_p', 'N_x_GSE', 'N_x_GSE_unc', 'N_y_GSE', 'N_y_GSE_unc', 'N_z_GSE', 'N_z_GSE_unc','v_sh', 'v_sh_unc']]
helsinki_with_normals.rename(columns={f'N_{comp}_GSE':f'N{comp}' for comp in ('x','y','z')},inplace=True)
helsinki_with_normals.rename(columns={f'N_{comp}_GSE_unc':f'N{comp}_unc' for comp in ('x','y','z')},inplace=True)
helsinki_with_normals.loc[:,'time_unc'] = helsinki_with_normals.apply(lambda row: 0.5*max(row['res_B'], row['res_p']), axis=1)
helsinki_with_normals.drop(columns=['res_B','res_p'],inplace=True)
helsinki_with_normals.loc[:,'source'] = 'H'


# %% With_normals
from src.methods.shock_intercepts.plotting import plot_time_differences

plot_time_differences(helsinki_events, bottom_panel=None, selection='earth', x_axis='delta_n', colouring='spacecraft', shock_normals=helsinki_with_normals)
plot_time_differences(helsinki_events, bottom_panel=None, selection='earth', x_axis='delta_t', colouring='spacecraft', shock_normals=helsinki_with_normals)

