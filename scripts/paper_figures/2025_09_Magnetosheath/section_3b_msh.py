# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""
# Plotting pulkkinen figures (not explaining why yet)

# %% Import
import os
import itertools as it

from src.config import MSH_DIR, OMNI_DIR
from src.processing.reading import import_processed_data
from src.methods.magnetosheath_saturation.plotting import plot_compare_responses

sample_interval = '5min'
data_pop = 'with_plasma'

omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_omni    = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

msh_dir = os.path.join(MSH_DIR, data_pop, sample_interval)
df_msh = import_processed_data(msh_dir, 'msh_times_combined.cdf')

# %% Params
param_names  = {'E_y_GSM': 'E_y',
                'V_flow' : 'V',
                'B_z_GSM': 'B_z',
                'N_tot'  : 'N',
                'AE_53m' : 'AE',
                'AEc_53m': 'AEc',
                'PCN_17m': 'PCN',
                'AA_17m' : 'AA',
                'SME_53m': 'SME'}

if sample_interval=='1min':
    data_type = 'mins'
else:
    data_type = 'counts'


 # %% Grid
responses = ('PCN_17m','AA_17m','SME_53m','AE_53m')
params    = ('E_R','E_y_GSM','B_z_GSM','V_flow','N_tot','B_clock')


for param, response in it.product(params, responses):
    if param!='B_clock':
        continue
    plot_compare_responses(df_omni, df_msh, param, response, restrict=True, shift_centre=False, data1_name=param_names.get(param,param), data2_name=param_names.get(response,response), data_name_map=param_names)
    break
