# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""

# %% lags

from src.methods.saturation.plotting.general import plot_delay_hists

plot_delay_hists('combined', 'sw', data_pop='plasma', resolution='5min')
plot_delay_hists('combined', 'msh', data_pop='plasma', resolution='5min')


# %% distrubtions

from src.methods.saturation.plotting.space_time import plot_sc_years, plot_sc_orbits, plot_sc_sw_msh, plot_data_inventory

resolution = '5min'

all_keys = ('c1','mms1','tha','thb','thc','thd','the')
sw_keys  = ('c1','mms1','thb','thc')
msh_keys = ('c1','mms1','tha','thd','the')


# Solar Wind
plot_sc_years(resolution, 'plasma', 'sw', sw_keys, combined=False)
plot_sc_orbits(resolution, 'plasma', 'sw', sw_keys)


# Magnetosheath
plot_sc_years(resolution, 'plasma', 'msh', msh_keys, combined=False)
plot_sc_orbits(resolution, 'plasma', 'msh', msh_keys)

# Both
plot_sc_sw_msh(resolution, 'plasma', sw_keys, msh_keys)

plot_data_inventory(*all_keys, region='all', resolution=resolution)
plot_data_inventory(*sw_keys, region='sw', resolution='5min')
plot_data_inventory(*msh_keys, region='msh', resolution='5min')


# %% counts

from src.methods.saturation.plotting.quality import plot_avg_counts, plot_compare_averaging


plot_avg_counts('mms1', 'msh', 'B_GSM', 'V_GSE', 'E_GSM')
plot_avg_counts('c1', 'msh', 'B_GSM', 'V_GSE', 'E_GSM')
plot_avg_counts('the', 'msh', 'B_GSM', 'V_GSE', 'E_GSM')


plot_compare_averaging('mms1', 'msh', 'B_z_GSM', 'V_x_GSE', 'E_y_GSM')

# %% sources

from src.methods.saturation.plotting.general import plot_pulkkinen_grid

sw_params = ('B_avg','B_z_GSM','B_clock','E_mag','E_y_GSM','E_R','N_tot','V_flow','P_flow')

plot_pulkkinen_grid(*sw_params, ind_reg='omni', dep_reg='sw', display='heat')

mah_params = ('B_avg','B_z_GSM','B_clock','E_mag','E_y_GSM','E_R','N_tot','V_flow','beta')

plot_pulkkinen_grid(*mah_params, ind_reg='sw',   dep_reg='msh', display='heat')
plot_pulkkinen_grid(*mah_params, ind_reg='omni', dep_reg='msh', display='heat')



# %% uncertainties

from src.methods.saturation.plotting.uncertainties import plot_independent_uncertainties

plot_independent_uncertainties('B_z_GSM', 'E_y_GSM', 'E_R', resolution='5min', spacecraft='omni')
plot_independent_uncertainties('B_z_GSM', 'E_y_GSM', 'E_R', resolution='5min', spacecraft='combined', region='sw')
plot_independent_uncertainties('B_z_GSM', 'E_y_GSM', 'E_R', resolution='5min', spacecraft='combined', region='msh')



