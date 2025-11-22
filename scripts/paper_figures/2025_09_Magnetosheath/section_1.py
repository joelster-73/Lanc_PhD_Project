# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %% Import
import os

from src.config import OMNI_DIR
from src.processing.reading import import_processed_data
from src.methods.magnetosheath_saturation.plotting import plot_driver_multi_responses

sample_interval = '1min'
data_pop = 'with_plasma'

if sample_interval=='1min':
    data_type = 'mins'
else:
    data_type = 'counts'

omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_omni    = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

# %% Mapping
param_names  = {'E_y_GSM': 'E_y',
                'V_flow' : 'V',
                'B_z_GSM': 'B_z',
                'N_tot'  : 'N'}

# %% Plots
responses = ('PCC','PCC_17m','PCN_17m','PCN','AA_17m','SME_53m','AE_53m','AEc_53m')
param     = 'E_R'

plot_driver_multi_responses(df_omni, None, param, *responses, restrict=True,  data_type=data_type, data1_name=param_names.get(param,param), data_name_map=param_names)