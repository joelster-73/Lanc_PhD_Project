# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %% Import
import os

from src.config import OMNI_DIR, SW_DIR, MSH_DIR
from src.processing.reading import import_processed_data
from src.methods.magnetosheath_saturation.plotting import plot_compare_indices

sample_interval = '1min'
data_pop = 'with_plasma'

if sample_interval=='1min':
    data_type = 'mins'
else:
    data_type = 'counts'

omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_omni = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

sw_dir = os.path.join(SW_DIR, data_pop, sample_interval)
df_sw  = import_processed_data(sw_dir, 'sw_times_combined.cdf')

msh_dir = os.path.join(MSH_DIR, data_pop, sample_interval)
df_msh = import_processed_data(msh_dir, 'msh_times_combined.cdf')

# %% Bias

indices = [['PCN','PCC'],['AA','PCC'],['SME','AE']]

plot_compare_indices(df_omni, None, indices, restrict=True, data_type=data_type, show_inverse=False, figure_title='OMNI', display='scatter')

plot_compare_indices(df_omni, df_sw, indices, restrict=True, data_type=data_type, figure_title='Solar wind times')
plot_compare_indices(df_omni, df_msh, indices, restrict=True, data_type=data_type, figure_title='Magnetosheath times')