# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %% Import

from src.methods.magnetosheath_saturation.plot_space_time import plot_sc_years, plot_sc_orbits, plot_sc_sw_msh

sample_interval = '1min'
data_pop = 'with_plasma'

sw_keys = ('c1','mms1','thb')
msh_keys = ('c1','mms1','the')

# %% Plots

# Solar Wind
plot_sc_years(sample_interval, data_pop, 'sw', sw_keys, combined=False)
plot_sc_orbits(sample_interval, data_pop, 'sw', sw_keys)


# Magnetosheath
plot_sc_years(sample_interval, data_pop, 'msh', msh_keys, combined=False)
plot_sc_orbits(sample_interval, data_pop, 'msh', msh_keys)

# Both
plot_sc_sw_msh(sample_interval, data_pop, sw_keys, msh_keys)
