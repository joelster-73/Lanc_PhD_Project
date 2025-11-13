# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %% Import
import os

from src.config import SW_DIR, MSH_DIR
from src.methods.magnetosheath_saturation.plot_space_time import plot_sc_years, plot_sc_orbits, plot_sc_sw_msh

sample_interval = '1min'
data_pop = 'with_plasma'

if sample_interval=='1min':
    data_type = 'mins'
else:
    data_type = 'counts'

sw_dir = os.path.join(SW_DIR, data_pop, sample_interval)
msh_dir = os.path.join(MSH_DIR, data_pop, sample_interval)

sw_keys = ('c1','m1','thb')
msh_keys = ('c1','m1','the')

# %% Plots

# Solar Wind
plot_sc_years(sw_dir, sc_keys=sw_keys, combined=False, data_type=data_type, region='sw')
plot_sc_orbits(sw_dir, sc_keys=sw_keys, data_type=data_type, region='sw')

# Magnetosheath
plot_sc_years(msh_dir, sc_keys=msh_keys, combined=False, data_type=data_type, region='msh')
plot_sc_orbits(msh_dir, sc_keys=msh_keys, data_type=data_type, region='msh')

# Both
plot_sc_sw_msh(sw_dir, msh_dir, sw_keys, msh_keys, data_type=data_type)

# %% Bias

# plot_bias_over_years(df_sw, df_merged, 'AE')
# plot_bias_in_parameters(df_sw, df_merged)

