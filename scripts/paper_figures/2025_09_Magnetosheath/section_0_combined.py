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
from src.analysing.calculations import calc_angle_between_vecs

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

df_sw.loc[df_sw['M_A']>100,'M_A'] = np.nan
df_merged.loc[df_merged['M_A_sw']>100,'M_A_sw'] = np.nan

# Rotation of clock angle
# Ignoring sw uncertainty for time being
df_merged['Delta B_theta_msh'] = np.abs(difference_series(df_merged['B_clock_sw'],df_merged['B_clock_msh'],unit='rad'))
not_nan = ~np.isnan(df_merged['Delta B_theta_msh'])
df_merged.loc[not_nan,'Delta B_theta_unc_msh'] = df_merged.loc[not_nan,'B_clock_unc_msh']
df_merged.attrs['units']['Delta B_theta_msh'] = 'rad'

# Reversal of Bz
# Not interested in error
df_merged['Delta B_z_msh'] = df_merged['B_z_GSM_msh']/df_merged['B_z_GSM_sw'] - 1
df_merged['Delta B_z_msh'] = df_merged['Delta B_z_msh'].replace([np.inf, -np.inf], np.nan)
df_merged.loc[np.abs(df_merged['Delta B_z_msh'])>1000,'Delta B_z_msh'] = np.nan
df_merged.attrs['units']['Delta B_z_msh'] = '1'

# Theta Bn angle - quasi-perp/quasi-para
df_sw['theta_Bn'] = calc_angle_between_vecs(df_sw, 'B_GSE', 'R_BSN')
df_merged['theta_Bn_sw'] = calc_angle_between_vecs(df_merged, 'B_GSE_sw', 'R_BSN_sw')

df_sw.loc[df_sw['theta_Bn']>np.pi/2,'theta_Bn'] = np.pi - df_sw.loc[df_sw['theta_Bn']>np.pi/2,'theta_Bn']
df_merged.loc[df_merged['theta_Bn_sw']>np.pi/2,'theta_Bn_sw'] = np.pi - df_merged.loc[df_merged['theta_Bn_sw']>np.pi/2,'theta_Bn_sw']

df_sw.attrs['units']['theta_Bn'] = 'rad'
df_merged.attrs['units']['theta_Bn_sw'] = 'rad'


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

plot_saturation_overview(df_sw, df_merged, 'B_z_GSM', 'AE_17m', 'B_clock', grp_src='msh', ind_src='sw', bounds=[-20,0], data_type=data_type, plot_test=False)
