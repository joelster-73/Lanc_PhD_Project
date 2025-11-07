# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %% Import
import os

from src.config import MSH_DIR, OMNI_DIR
from src.processing.reading import import_processed_data

from src.methods.magnetosheath_saturation.plotting import plot_compare_responses

sample_interval = '1min'
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
                'AE_30m' : 'AE',
                'AEc_30m': 'AEc',
                'PCN_17m': 'PCN',
                'AA_17m' : 'AA',
                'SME_30m': 'SME'}


# %% Response


plot_compare_responses(df_omni, df_msh, 'E_y_GSM', 'AA_17m', restrict=True, data1_name=param_names.get('E_y_GSM','E_y_GSM'), data2_name=param_names.get('AA_17m','AA_17m'), data_type=data_type, compare_sw_msh=True)

plot_compare_responses(df_omni, df_msh, 'E_y_GSM', 'AA_17m', restrict=True, data1_name=param_names.get('E_y_GSM','E_y_GSM'), data2_name=param_names.get('AA_17m','AA_17m'), data_type=data_type, compare_sw_msh=True)


# Add in the Poynting Flux
# Extend to full range (so \pm Bz and Ey)

for response in ('PCN_17m','AA_17m','AE_30m','SME_30m'):
    for param in ('N_tot','E_mag','E_y_GSM','V_flow','B_avg','B_z_GSM'):
        for msh_map in ({},{'E_y_GSM': 'E_parallel'}):
            if msh_map and param not in msh_map:
                continue
            plot_compare_responses(df_omni, df_msh, param, response, restrict=True, data1_name=param_names.get(param,param), data2_name=param_names.get(response,response), min_count=40, compare_sw_msh=True, msh_map=msh_map)
            plot_compare_responses(df_omni, df_msh, param, response, restrict=True, data1_name=param_names.get(param,param), data2_name=param_names.get(response,response), min_count=40, compare_sw_msh=True, msh_map=msh_map, restrict=False)
            plot_compare_responses(df_omni, df_msh, param, response, restrict=True, data1_name=param_names.get(param,param), data2_name=param_names.get(response,response), min_count=40, compare_sw_msh=True, msh_map=msh_map, display='scatter')

