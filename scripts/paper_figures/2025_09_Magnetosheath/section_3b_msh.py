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
from src.methods.magnetosheath_saturation.plotting import plot_compare_responses, plot_omni_vs_msh

sample_interval = '5min'
data_pop = 'with_plasma'

omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_omni    = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

msh_dir = os.path.join(MSH_DIR, data_pop, sample_interval)
df_msh = import_processed_data(msh_dir, 'msh_times_combined.cdf')




### ADD THIS TO DATA BEFORE SAMPLING
### NEED TO FIX PLASMA BETA TOO


import numpy as np
from scipy.constants import physical_constants, m_p
ma = physical_constants['alpha particle mass'][0]

for src in ('sw','msh'):

    m = m_p * (1+ma*df_msh[f'na_np_ratio_{src}']) / (1+df_msh[f'na_np_ratio_{src}'])
    rho = df_msh[f'N_tot_{src}'] * m

    df_msh[f'V_A_{src}'] = df_msh[f'B_avg_{src}'] * 1e-9 / np.sqrt(4*np.pi*1e-7*rho*1e6) / 1e3


# %% Params
param_names  = {'E_y_GSM' : 'E_y',
                'V_flow'  : 'V',
                'P_flow'  : 'P',
                'B_clock' : 'B_theta',
                'B_z_GSM' : 'B_z',
                'N_tot'   : 'N',
                'AE_53m'  : 'AE',
                'AEc_53m' : 'AEc',
                'PCN_17m' : 'PCN',
                'AA_17m'  : 'AA',
                'SME_53m' : 'SME'}

if sample_interval=='1min':
    data_type = 'mins'
else:
    data_type = 'counts'


 # %% Grid
responses = ('PCN','AA_17m','SME_53m','AE_53m')
params    = ('E_R','E_y_GSM','B_z_GSM','V_flow','N_tot','P_flow','B_clock')


for param, response in it.product(params, responses):

    plot_compare_responses(df_omni, df_msh, param, response, restrict=True, data1_name=param_names.get(param,param), data2_name=param_names.get(response,response), data_name_map=param_names)

# %% OMNI_vs_msh
params = ('E_R','E_y_GSM','B_z_GSM','V_flow','V_A','N_tot','P_flow','B_clock')

plot_omni_vs_msh(df_msh, params, restrict=True, data_name_map=param_names, msh_map={'E_y_GSM': 'E_parallel'})