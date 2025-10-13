# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %% Import
import os

from src.config import MSH_DIR, OMNI_DIR
from src.processing.reading import import_processed_data

sample_interval = '1min'
data_pop = 'field_only'

# Solar wind data
omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_sw    = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

msh_dir = os.path.join(MSH_DIR, data_pop, sample_interval)
df_merged = import_processed_data(msh_dir, 'msh_times_combined.cdf')

if sample_interval=='1min':
    data_type = 'mins'
else:
    data_type = 'counts'


# %% Space_and_time
from src.methods.magnetosheath_saturation.plotting import plot_sc_orbits, plot_sc_years

plot_sc_orbits(msh_dir, data_type=data_type)

plot_sc_years(msh_dir)
plot_sc_years(msh_dir, combined=False)


# %% Time_Activity
from src.plotting.comparing.space_time import plot_compare_datasets_with_activity

plot_compare_datasets_with_activity(df_sw, df_merged, df_colours=('orange','blue'), df_names=('All OMNI','Contemp MSH'))


# %% Bias
from src.methods.magnetosheath_saturation.plotting import plot_bias_over_years, plot_bias_in_parameters

plot_bias_over_years(df_sw, df_merged, 'AE')
plot_bias_in_parameters(df_sw, df_merged)


# %% Overview
from src.methods.magnetosheath_saturation.plotting import plot_saturation_overview

plot_saturation_overview(df_sw, df_merged, 'B_avg', 'AE_17m', 'B_clock', ind_src='msh', grp_src='msh', bounds=None, data_type=data_type, plot_test=False)
