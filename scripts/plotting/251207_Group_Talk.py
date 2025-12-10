# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 10:52:25 2025

@author: richarj2
"""


# %% Import
from src.processing.reading import import_processed_data
from src.methods.magnetosheath_saturation.plotting import plot_driver_multi_responses, plot_driver_response, plot_saturation_overview, plot_different_lags, plot_pulkkinen_grid
from src.methods.magnetosheath_saturation.plot_space_time import plot_sc_years
from src.methods.magnetosheath_saturation.merge_region_sc import update_omni
from src.methods.magnetosheath_saturation.sc_delay_time import add_dynamic_index_lag


sample_interval = '1min'
data_pop = 'plasma'
region = 'sw'

data_type = 'mins' if sample_interval == '1min' else 'counts'

df_omni = import_processed_data('omni', resolution=sample_interval)
update_omni(df_omni)

spacecraft = 'c1'

df_sc   = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_{spacecraft}')
df_pc   = import_processed_data('indices', file_name=f'combined_{sample_interval}')
add_dynamic_index_lag(df_sc, df_pc)


# %% Recreating_Saturation
param_names  = {'E_y_GSM': 'E_y',
                'V_flow' : 'V',
                'B_z_GSM': 'B_z',
                'N_tot'  : 'N',
                'AE_53m' : 'AE',
                'PCN_17m': 'PCN',
                'PCC_17m': 'PCC',
                'SME_53m': 'SME',
                'na_np_ratio': 'R_alpha_p'}

sc_index_map = {name: f'{name}_adj' for name in ('AE_53m','AEc_53m','PCN_17m','PCC_17m','AA_60m','SME_53m')}

params = ('E_y_GSM','B_avg','B_z_GSM','beta','N_tot','V_flow','P_flow','B_clock')

plot_pulkkinen_grid(df_omni, df_sc, params, source='sw', restrict=True, data_name_map=param_names, display='scatter')
plot_pulkkinen_grid(df_omni, df_sc, params, source='sw', restrict=True, data_name_map=param_names)

# Myllys results
param = 'E_R'
plot_driver_multi_responses(df_omni, None, df_pc, param, 'PCN_17m', 'PCC_17m', index_map=sc_index_map, restrict=True, data_type=data_type, sharey='row', data1_name=param_names.get(param,param), data_name_map=param_names)

plot_driver_multi_responses(df_omni, df_sc, df_pc, param, 'PCN_17m', 'PCC_17m', index_map=sc_index_map, restrict=True, data_type=data_type, sharey='row', data1_name=param_names.get(param,param), data_name_map=param_names)

# Lakka results
param = 'E_y_GSM'
plot_driver_multi_responses(df_omni, None, df_pc, param, 'AE_53m', 'SME_53m', index_map=sc_index_map, restrict=True, data_type=data_type, sharey='row', data1_name=param_names.get(param,param), data_name_map=param_names)

plot_driver_multi_responses(df_omni, df_sc, df_pc, param, 'AE_53m', 'SME_53m', index_map=sc_index_map, restrict=True, data_type=data_type, sharey='row', data1_name=param_names.get(param,param), data_name_map=param_names)

# %% Data_Distribution

sw_keys  = ('c1','mms1','thb')
msh_keys = ('c1','mms1','the')

plot_sc_years(sample_interval='1min', region='sw', data_pop='plasma', sc_keys=('c1',), combined=False, year_range=(2000,2026))
plot_sc_years(sample_interval='1min', region='sw', data_pop='plasma', sc_keys=sw_keys, combined=True)

plot_sc_years(sample_interval='1min', region='msh', data_pop='plasma', sc_keys=('c1',), combined=False, year_range=(2000,2026))
plot_sc_years(sample_interval='1min', region='msh', data_pop='plasma', sc_keys=msh_keys, combined=True)


# %% Lags

plot_different_lags(df_sc, df_pc, param, 'PCC', restrict=True, data_type=data_type, skip_zero=True, data1_name=param_names.get(param,param), data_name_map=param_names)


# %% Saturation_Cause

sample_interval = '5min'
data_pop = 'plasma'
region = 'msh'

data_type = 'mins' if sample_interval == '1min' else 'counts'

df_omni = import_processed_data('omni', resolution=sample_interval)
update_omni(df_omni)

spacecraft = 'c1'
df_sc   = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_{spacecraft}')
df_pc   = import_processed_data('indices', file_name=f'combined_{sample_interval}')
add_dynamic_index_lag(df_sc, df_pc)

# %%
param    = 'E_y_GSM'
response = 'SME_53m'

plot_driver_response(df_omni, df_sc, df_pc, param, response, restrict=True, data1_name=param_names.get(param,param), data2_name=param_names.get(response,response))

split = 'na_np_ratio'
plot_saturation_overview(df_omni, df_sc, df_pc, param, response, split, grp_src='sw', plot_test=True, max_expansions=0, data1_name=param_names.get(param,param), data2_name=param_names.get(response,response), data3_name=param_names.get(split,None), bounds=[0,10])
