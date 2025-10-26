# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %% Import
import os

from src.config import SW_DIR, OMNI_DIR
from src.processing.reading import import_processed_data
from src.methods.magnetosheath_saturation.plotting import plot_sc_years, plot_compare_sc_omni

sample_interval = '1min'
data_pop = 'with_plasma'

# Solar wind data
omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_omni    = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

sw_dir = os.path.join(SW_DIR, data_pop, sample_interval)
df_merged = import_processed_data(sw_dir, 'sw_times_combined.cdf')

if sample_interval=='1min':
    data_type = 'mins'
else:
    data_type = 'counts'

pulk_params = ('E_y_GSM','B_z_GSM','V_flow','N_tot','S_mag')
param_names  = {'E_y_GSM': 'E_y', 'V_flow': 'V', 'B_z_GSM': 'B_z', 'N_tot': 'N', 'AE_17m': 'AE', 'PCN_17m': 'PCN'}
group_params = (('Delta B_theta','msh'),('beta','msh'),('M_A','sw'))
responses = ('AE_17m','PCN_17m')

# %% Space_and_time


sc_keys = ('c1','m1')
plot_sc_years(sw_dir, 'sw', sc_keys=sc_keys, data_type=data_type)

param = 'E_R'

plot_compare_sc_omni(df_omni, df_merged, param, *responses, restrict=True, min_count=50, data_type=data_type, data1_name=param_names.get(param,param), data_name_map = param_names)
