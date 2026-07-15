# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""






# %% Import
from src.processing.reading import import_processed_data

sample_interval = '1min'
data_pop = 'with_plasma'

region = 'sw'

df_sc   = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_mms1')













# %% Import
import os

from src.config import MSH_DIR, OMNI_DIR
from src.processing.reading import import_processed_data

from src.methods.magnetosheath_saturation.plotting import plot_saturation_overview, plot_compare_responses, plot_grouping_cause, plot_different_lags

sample_interval = '5min'
data_pop = 'with_plasma'

# Solar wind data
omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_sw    = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

msh_dir = os.path.join(MSH_DIR, data_pop, sample_interval)
df_merged = import_processed_data(msh_dir, 'msh_times_combined.cdf')

if sample_interval=='1min':
    data_type = 'mins'
else:
    data_type = 'counts'

pulk_params = ('E_y_GSM','B_z_GSM','V_flow','N_tot','S_mag')
param_names  = {'E_y_GSM': 'E_y', 'V_flow': 'V', 'B_z_GSM': 'B_z', 'N_tot': 'N', 'AE_17m': 'AE', 'PCN_17m': 'PCN'}
group_params = (('Delta B_theta','msh'),('beta','msh'),('M_A','sw'))
responses = ('AE_17m','PCN_17m')



# %% Overview

for group_param in group_params:
    for param in pulk_params:
        for response in responses:
            plot_compare_responses(df_sw, df_merged, param, response, restrict=True, data1_name=param_names.get(param,param), data2_name=param_names.get(response,response))
        plot_saturation_overview(df_sw, df_merged, param, param, group_param[0], ind_src='sw', dep_src='msh', grp_src=group_param[1], data_type=data_type)



    plot_saturation_overview(df_sw, df_merged, 'E_y_GSM', 'E_parallel', group_param[0], ind_src='sw', dep_src='msh', grp_src=group_param[1], data_type=data_type, same_var=True)
    for response in responses:
        plot_saturation_overview(df_sw, df_merged, 'E_parallel', response, group_param[0], ind_src='msh', dep_src='pc', grp_src=group_param[1], data_type=data_type)

    plot_saturation_overview(df_sw, df_merged, 'S_mag', 'S_perp', group_param[0], ind_src='sw', dep_src='msh', grp_src=group_param[1], data_type=data_type, same_var=True, invert_y=True)
    for response in responses:
        plot_saturation_overview(df_sw, df_merged, 'S_perp', response, group_param[0], ind_src='msh', dep_src='pc', grp_src=group_param[1], data_type=data_type)

    plot_saturation_overview(df_sw, df_merged, 'M_A', 'beta', group_param[0], ind_src='sw', dep_src='msh', grp_src=group_param[1], data_type=data_type, same_var=True)
    for response in responses:
        plot_saturation_overview(df_sw, df_merged, 'beta', response, group_param[0], ind_src='msh', dep_src='pc', grp_src=group_param[1], data_type=data_type)
# %% Grouping

plot_grouping_cause(df_merged, 'M_A', 'Delta B_theta', want_legend=True, show_count=True, data_type=data_type, region='sem')

# %% Lags

for param in ('E_y_GSM','E_R'):
    name = param_names.get(param,None)
    latex_it = (param in param_names)

    plot_different_lags(df_sw, param, 'AE', data1_name=name, name1_latex=latex_it)
    plot_different_lags(df_sw, param, 'PCN', data1_name=name, name1_latex=latex_it)
    plot_different_lags(df_merged, param, 'PCN', ind_src='msh', df_type='msh', data1_name=name, name1_latex=latex_it)


# %% Test
param = 'E_y_GSM'
response = 'PCN'
plot_compare_responses(df_sw, df_merged, param, response, restrict=True, data1_name=param_names.get(param,param), data2_name=param_names.get(response,response), min_count=20, compare_sw_msh=True)