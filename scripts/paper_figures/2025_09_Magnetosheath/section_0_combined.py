# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %% Import
import os
import numpy as np

from src.config import MSH_DIR, OMNI_DIR
from src.analysing.comparing import difference_series
from src.processing.reading import import_processed_data

sample_interval = '5min'
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

# Rotation of clock angle
# Ignoring sw uncertainty for time being
df_merged['Delta B_theta_msh'] = np.abs(difference_series(df_merged['B_clock_sw'],df_merged['B_clock_msh'],unit='rad'))
not_nan = ~np.isnan(df_merged['Delta B_theta_msh'])
df_merged.loc[not_nan,'Delta B_theta_unc_msh'] = df_merged.loc[not_nan,'B_clock_unc_msh']
df_merged.attrs['units']['Delta B_theta_msh'] = 'rad'

# Reversal of Bz
# Not interested in error
df_merged['Delta Bz_msh'] = df_merged['B_z_GSM_msh']/df_merged['B_z_GSM_sw']
df_merged['Delta Bz_msh'] = df_merged['Delta Bz_msh'].replace([np.inf, -np.inf], np.nan)
df_merged.attrs['units']['Delta Bz_msh'] = '1'


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

plot_saturation_overview(df_sw, df_merged, 'B_z_GSM', 'AE_17m', 'Delta B_theta', data_type=data_type)
