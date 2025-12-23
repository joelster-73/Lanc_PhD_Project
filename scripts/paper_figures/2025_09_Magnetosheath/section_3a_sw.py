# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2

Showing saturation present with in-situ, so not due to OMNI (solely)
"""

# %% Import
from src.processing.reading import import_processed_data
from src.methods.magnetosheath_saturation.plotting import plot_driver_multi_responses, plot_different_lags, plot_pulkkinen_grid
from src.methods.magnetosheath_saturation.merge_region_sc import update_omni
from src.methods.magnetosheath_saturation.sc_delay_time import add_dynamic_index_lag

sample_interval = '1min'
data_pop = 'with_plasma'

region = 'sw'

df_omni = import_processed_data('omni', resolution=sample_interval)
update_omni(df_omni)

spacecraft = 'combined'

df_sc = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_{spacecraft}')
df_pc = import_processed_data('indices', file_name=f'combined_{sample_interval}')
add_dynamic_index_lag(df_sc, df_pc)


# %% Params

param_names  = {'E_y_GSM': 'E_y',
                'V_flow' : 'V',
                'B_z_GSM': 'B_z',
                'N_tot'  : 'N',
                'AE_53m' : 'AE',
                'PCN_17m': 'PCN',
                'PCC_17m': 'PCC',
                'AA_60m' : 'AA',
                'SME_53m': 'SME',
                'SMP_17m': 'SMP'}

sc_index_map = {name: f'{name}_adj' for name in ('AE_53m','PCN_17m','PCC_17m','SMP_17m','AA_60m','SME_53m')}

# %% Verify

params = ('E_y_GSM','B_avg','B_z_GSM','beta','N_tot','V_flow','P_flow','B_clock')

plot_pulkkinen_grid(df_omni, df_sc, params, source='sw', restrict=True, data_name_map=param_names, display='scatter')
plot_pulkkinen_grid(df_omni, df_sc, params, source='sw', restrict=True, data_name_map=param_names)


# %% Papers

# Myllys results
param = 'E_R'
for bottom_axis in ('heat','scatter','hist'):
    plot_driver_multi_responses(df_omni, df_sc, df_pc, param, 'PCN_17m', 'PCC_17m', 'SMP_17m', index_map=sc_index_map, restrict=True, bottom_axis=bottom_axis, sharey='row', data1_name=param_names.get(param,param), data_name_map=param_names, min_count=50)


# Lakka results
param = 'E_y_GSM'
for bottom_axis in ('heat','scatter','hist'):
    plot_driver_multi_responses(df_omni, df_sc, df_pc, param, 'AE_53m', 'SME_53m','AA_60m', index_map=sc_index_map, restrict=True, bottom_axis=bottom_axis, sharey='row', data1_name=param_names.get(param,param), data_name_map=param_names, min_count=50)

# %% Lags

plot_different_lags(df_sc, df_pc, 'E_R', 'SMP', restrict=True, data1_name=param_names.get(param,param), data_name_map=param_names, min_count=50)

plot_different_lags(df_sc, df_pc, 'E_y_GSM', 'SME', restrict=True, data1_name=param_names.get(param,param), data_name_map=param_names, min_count=50)

# %% All
responses = ('SMP_17m','SME_53m','AA_60m')

for param in ('E_R','E_y_GSM','B_z_GSM','V_flow','beta','N_tot','P_flow','B_clock'):

    plot_driver_multi_responses(df_omni, df_sc, df_pc, param, *responses, index_map=sc_index_map, restrict=True, data1_name=param_names.get(param,param), data_name_map=param_names)
