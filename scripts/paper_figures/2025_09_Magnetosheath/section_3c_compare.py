# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""
# Plotting why this occurs (so saturation overview with grouping)

# %% Import
import os
import itertools as it

from src.config import MSH_DIR, OMNI_DIR
from src.processing.reading import import_processed_data

from src.methods.magnetosheath_saturation.plotting import plot_compare_responses

sample_interval = '5min'
data_pop = 'with_plasma'

# Solar wind data
omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_omni  = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

msh_dir = os.path.join(MSH_DIR, data_pop, sample_interval)
df_msh  = import_processed_data(msh_dir, 'msh_times_combined.cdf')

if sample_interval=='1min':
    data_type = 'mins'
else:
    data_type = 'counts'


param_names  = {'E_y_GSM': 'E_y',
                'V_flow' : 'V',
                'B_z_GSM': 'B_z',
                'N_tot'  : 'N',
                'AE_53m' : 'AE',
                'AEc_53m': 'AEc',
                'PCN_17m': 'PCN',
                'AA_17m' : 'AA',
                'SME_53m': 'SME'}



# %% Test


plot_compare_responses(df_omni, df_msh, 'E_y_GSM', 'AA_17m', restrict=True, data1_name=param_names.get('E_y_GSM','E_y_GSM'), data2_name=param_names.get('AA_17m','AA_17m'), data_type=data_type, compare_sw_msh=True)

plot_compare_responses(df_omni, df_msh, 'E_y_GSM', 'AA_17m', restrict=True, data1_name=param_names.get('E_y_GSM','E_y_GSM'), data2_name=param_names.get('AA_17m','AA_17m'), data_type=data_type, compare_sw_msh=True)

# %% Pulkkinen

# Need to try and reduce the error in each bin - look at scatters (with sw and msh), too much variance
# Likely down to fixed 17-minute lag time of BS to PC


responses = ('PCN_17m', 'AA_17m', 'AE_53m', 'AEc_53m', 'SME_53m')

params = ('E_mag', 'E_y_GSM', 'E_R', 'V_flow', 'N_tot', 'B_avg', 'B_z_GSM', 'S_x_GSM')
regions = ('sem', 'med', 'max')
msh_maps = ({}, {'E_y_GSM': 'E_parallel', 'S_x_GSM': 'S_perp'})


params = ('E_y_GSM', 'E_R', 'B_z_GSM', 'V_flow')
regions = ('sem',)
msh_maps = ({},)

for param, response, region, msh_map in it.product(params, responses, regions, msh_maps):
    if msh_map and param not in msh_map:
        continue
    min_count = 50 if region=='max' else 40
    plot_compare_responses(df_omni, df_msh, param, response, restrict=True, data1_name=param_names.get(param,param), data2_name=param_names.get(response,response), min_count=min_count, compare_sw_msh=True, msh_map=msh_map, region=region)



            # plot_compare_responses(df_omni, df_msh, param, response, restrict=False, data1_name=param_names.get(param,param), data2_name=param_names.get(response,response), min_count=40, compare_sw_msh=True, msh_map=msh_map)
            # plot_compare_responses(df_omni, df_msh, param, response, restrict=True, data1_name=param_names.get(param,param), data2_name=param_names.get(response,response), min_count=40, compare_sw_msh=True, msh_map=msh_map, display='scatter')

