# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""

# %% averaging

from src.methods.saturation.plotting.quality import plot_avg_counts_grid, plot_avg_distributions

# Describe in manuscript but don't include

plot_avg_counts_grid('cluster', 'themis', 'mms')

# Include one of the figures below (probs restricted but to a lesser extent)

plot_avg_distributions('combined', 'V_x_GSE', 'B_z_GSM', 'E_yz', 'S_x_GSE', restrict=False)
plot_avg_distributions('combined', 'V_x_GSE', 'B_z_GSM', 'E_yz', 'S_x_GSE', restrict=True)

# %% sensitivity

### Add plots that are to show sensitivty to filtering parameters
### Won't include, perhaps in appendix, but need to say they're not sensitive

# %% uncertainties

from src.methods.saturation.plotting.quality import plot_independent_uncertainties

_ = plot_independent_uncertainties('B_z_GSM', 'E_R', 'E_yz', 'S_x_GSE', regions=('sw','msh'))


# %% distrubtions

from src.methods.saturation.plotting.space_time import plot_sc_years, plot_sc_orbits, plot_sc_sw_msh, plot_data_inventory

### None of these figures are to be included, but are used when describing the final solar wind and magnetosheath datasets

### Make sure to print out somewhere how much data there actually is (how many minutes and how many five minute samples over how many unique days)

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

plot_data_inventory(*all_keys, region='all')
plot_data_inventory(*sw_keys, region='sw', display='monthly')
plot_data_inventory(*msh_keys, region='msh', display='monthly')