# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2

Showing saturation present with in-situ, so not due to OMNI (solely)
"""


# %% Import
from src.processing.reading import import_processed_data
from src.methods.magnetosheath_saturation.plotting import plot_driver_multi_responses
from src.methods.magnetosheath_saturation.merge_region_sc import update_omni, update_pc_index
from src.processing.dataframes import merge_dataframes

sample_interval = '5min'
data_pop = 'with_plasma'

data_type = 'mins' if sample_interval == '1min' else 'counts'

region = 'sw'

df_sc   = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_combined')
df_omni = import_processed_data('omni', resolution=sample_interval)
df_pc   = import_processed_data('indices', file_name=f'combined_{sample_interval}')

update_omni(df_omni)
# Very small, nearly 0, PCC values
for df in (df_sc,df_pc): # CAN REMOVE DF_SC
    update_pc_index(df, minimum=0.15)

df_sw   = merge_dataframes(df_omni, df_pc)
# %% Params

param_names  = {'E_y_GSM': 'E_y',
                'V_flow' : 'V',
                'B_z_GSM': 'B_z',
                'N_tot'  : 'N',
                'AE_53m' : 'AE',
                'AEc_53m': 'AEc',
                'PCN_17m': 'PCN',
                'PCC_17m': 'PCC',
                'PCM_17m': 'PCM',
                'AA_60m' : 'AA',
                'SME_53m': 'SME'}

# %% Papers


### UPDATE FUNCTIONS TO TAKE IN DF_OMNI, DF_SC, AND DF_PC



# Myllys results
param = 'E_R'
plot_driver_multi_responses(df_sw, df_sc, param, 'PCN_17m', 'PCC_17m', 'PCM_17m', restrict=True, data_type=data_type, sharey='row', data1_name=param_names.get(param,param), data_name_map=param_names)

# Lakka results
param = 'E_y_GSM'
plot_driver_multi_responses(df_sw, df_sc, param, 'AE_53m', 'SME_53m','AA_60m', restrict=True, data_type=data_type, sharey='row', data1_name=param_names.get(param,param), data_name_map=param_names)

# %% All
responses = ('PCM_17m','SME_53m','AA_60m')

for param in ('P_flow','E_R','E_y_GSM','B_z_GSM','V_flow','N_tot','P_flow','B_clock'):

    plot_driver_multi_responses(df_sw, df_sc, param, *responses, restrict=True, data_type=data_type, data1_name=param_names.get(param,param), data_name_map=param_names)
    break