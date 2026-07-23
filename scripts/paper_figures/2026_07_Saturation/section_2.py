# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""

# %% lags

from src.methods.saturation.plotting.general import plot_delay_hists

plot_delay_hists('combined', 'sw', data_pop='plasma', sample_interval='5min')
plot_delay_hists('combined', 'msh', data_pop='plasma', sample_interval='5min')


# %% distrubtions

from src.methods.saturation.plotting.space_time import plot_sc_years, plot_sc_orbits, plot_sc_sw_msh

sample_interval = '1min'
data_pop = 'plasma'

sw_keys = ('c1','mms1','thb')
msh_keys = ('c1','mms1','the')


# Solar Wind
plot_sc_years(sample_interval, data_pop, 'sw', sw_keys, combined=False)
plot_sc_orbits(sample_interval, data_pop, 'sw', sw_keys)


# Magnetosheath
plot_sc_years(sample_interval, data_pop, 'msh', msh_keys, combined=False)
plot_sc_orbits(sample_interval, data_pop, 'msh', msh_keys)

# Both
plot_sc_sw_msh(sample_interval, data_pop, sw_keys, msh_keys)

# %% sources

from src.methods.saturation.plotting.general import plot_pulkkinen_grid

params = ('B_avg','B_z_GSM','E_mag','E_y_GSM','V_flow','B_clock','beta','N_tot','P_flow','T_tot')

plot_pulkkinen_grid(*params, ind_src='omni', dep_src='sw', resolution='5min', restrict=True, display='heat')
plot_pulkkinen_grid(*params, ind_src='sw', dep_src='msh', resolution='5min', restrict=True, display='heat')




# %% uncertainties

from src.methods.saturation.plotting.uncertainties import plot_independent_uncertainties

plot_independent_uncertainties('B_z_GSM', 'E_y_GSM', 'E_R', resolution='5min', spacecraft='omni')
plot_independent_uncertainties('B_z_GSM', 'E_y_GSM', 'E_R', resolution='5min', spacecraft='combined', region='sw')
plot_independent_uncertainties('B_z_GSM', 'E_y_GSM', 'E_R', resolution='5min', spacecraft='combined', region='msh')



