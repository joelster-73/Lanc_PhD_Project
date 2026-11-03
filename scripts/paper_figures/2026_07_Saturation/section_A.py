# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 21:49:36 2026

@author: richarj2

Other plots that may be useful but not directly needed.
"""

# %% inventory

from src.methods.saturation.plotting.general import plot_delay_hists
from src.methods.saturation.plotting.space_time import plot_data_inventory

# Plots not to be included, but numbers are.

sw_keys  = ('c1','c3','mms1')
msh_keys = ('c1','c3','mms1','tha','thd','the')

plot_data_inventory(*sw_keys, region='sw', display='monthly', print_info=True)
plot_data_inventory(*msh_keys, region='msh', display='monthly', print_info=True)

plot_delay_hists('combined', ('msh','sw'), resolution='5min')

from src.methods.saturation.plotting.quality import plot_avg_counts_grid, check_removed_counts, plot_remove_low_counts

# %% averaging

# Describe in manuscript but don't include

plot_avg_counts_grid('cluster', 'themis', 'mms')

# Create a summary table for the datasets at the end of the section
check_removed_counts('combined', 'drivers', resolution='5min', region='sw', restrict=True)
check_removed_counts('combined', 'drivers', resolution='5min', region='msh', restrict=True)

# Context for Thesis
plot_remove_low_counts('combined', 'drivers', resolution='5min', regions=('sw','msh'), restrict=True)



# %% distrubtions

from src.methods.saturation.plotting.space_time import plot_sc_years, plot_sc_orbits, plot_sc_sw_msh

# None of these figures are to be included, but are used when describing the final solar wind and magnetosheath datasets

resolution = '1min'

all_keys = ('c1','c3','mms1','tha','thb','thc','thd','the')
sw_keys  = ('c1','c3','mms1')
msh_keys = ('c1','c3','mms1','tha','thd','the')


# Solar Wind
plot_sc_years(resolution, 'plasma', 'sw', sw_keys, combined=False)
plot_sc_orbits(resolution, 'plasma', 'sw', sw_keys)


# Magnetosheath
plot_sc_years(resolution, 'plasma', 'msh', msh_keys, combined=False)
plot_sc_orbits(resolution, 'plasma', 'msh', msh_keys)

# Both
plot_sc_sw_msh(resolution, 'plasma', sw_keys, msh_keys)



