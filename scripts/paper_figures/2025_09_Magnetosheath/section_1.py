# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %% Import
import os

from src.config import SW_DIR, OMNI_DIR
from src.processing.reading import import_processed_data
from src.methods.magnetosheath_saturation.plotting import plot_compare_sc_omni

sample_interval = '1min'
data_pop = 'with_plasma'

omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_omni    = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

sw_dir = os.path.join(SW_DIR, data_pop, sample_interval)
df_sw = import_processed_data(sw_dir, 'sw_times_combined.cdf')

# %% Params
param_names  = {'E_y_GSM': 'E_y',
                'V_flow' : 'V',
                'B_z_GSM': 'B_z',
                'N_tot'  : 'N',
                'AE_30m' : 'AE',
                'AEc_30m': 'AEc',
                'PCN_17m': 'PCN',
                'AA_17m' : 'AA',
                'SME_30m': 'SME'}

param     = 'E_y_GSM'

if sample_interval=='1min':
    data_type = 'mins'
else:
    data_type = 'counts'
# %% Plots
responses = ('AE_30m','PCN_17m')
plot_compare_sc_omni(df_omni, None, param, *responses, restrict=True,  data_type=data_type, data1_name=param_names.get(param,param), data_name_map=param_names)

responses = ('AE_30m','AEc_30m')
plot_compare_sc_omni(df_omni, df_sw, param, *responses, restrict=True,  data_type=data_type, data1_name=param_names.get(param,param), data_name_map=param_names)


responses = ('PCN_17m','AA_17m','AE_30m','AEc_30m','SME_30m')
plot_compare_sc_omni(df_omni, df_sw, param, *responses, restrict=True,  data_type=data_type, data1_name=param_names.get(param,param), data_name_map=param_names)
