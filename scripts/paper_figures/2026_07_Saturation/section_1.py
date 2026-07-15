# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %% Import

from src.processing.reading import import_processed_data
from src.methods.magnetosheath_saturation.plotting import plot_driver_multi_responses

sample_interval = '1min'
data_pop = 'with_plasma'

df_omni = import_processed_data('omni', resolution=sample_interval)
df_pc   = import_processed_data('indices', file_name=f'combined_{sample_interval}')

# %% Mapping
param_names  = {'E_y_GSM': 'E_y',
                'V_flow' : 'V',
                'B_z_GSM': 'B_z',
                'N_tot'  : 'N'}

# %% Plots
responses = ('PCN_17m','PCC_17m','SMP_17m','SME_53m','AE_53m','AA_60m')
param     = 'E_R'

for bottom_axis in ('heat','scatter','hist'):
    plot_driver_multi_responses(df_omni, None, df_pc, param, *responses, restrict=True, bottom_axis=bottom_axis, data1_name=param_names.get(param,param), data_name_map=param_names)