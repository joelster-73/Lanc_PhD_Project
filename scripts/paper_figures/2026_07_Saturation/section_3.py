# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2

Spacecraft data.
"""

# %% averaging

from src.methods.saturation.plotting.quality import plot_avg_counts_grid, plot_avg_distributions

# Describe in manuscript but don't include

plot_avg_counts_grid('cluster', 'themis', 'mms')

# Include unrestricted figure
# Point is to say whether averaging affects distribution; it doesn't
# Not imposing restrictions as only fitting to a certain x range
# See if any spacecraft under/over compares to other

plot_avg_distributions('combined', 'data')

plot_avg_distributions('combined', 'drivers', restrict=True)


plot_avg_distributions('c1', 'data')

plot_avg_distributions('c3', 'data')

plot_avg_distributions('mms1', 'data')

plot_avg_distributions(('tha','thd','the'), 'data', regions=('msh','msh','msh'))

plot_avg_distributions(('thb','thc'), 'data', regions=('sw','sw'))




# %% sensitivity

# Add plots that are to show sensitivty to filtering parameters
# Won't include, perhaps in appendix, but need to say they're not sensitive
# All in the `calibration.py' file

# %% uncertainties

from src.methods.saturation.plotting.quality import plot_independent_uncertainties

_ = plot_independent_uncertainties('B_z_GSM', 'E_R', 'E_yz', 'S_x_GSE', regions=('sw','msh'))



# %% inventory

from src.methods.saturation.plotting.general import plot_delay_hists
from src.methods.saturation.plotting.space_time import plot_data_inventory

# Plots not to be included, but numbers are.

sw_keys  = ('c1','c3','mms1')
msh_keys = ('c1','c3','mms1','tha','thd','the')

plot_data_inventory(*sw_keys, region='sw', display='monthly', print_info=True)
plot_data_inventory(*msh_keys, region='msh', display='monthly', print_info=True)

plot_delay_hists('combined', ('msh','sw'), resolution='5min')

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

