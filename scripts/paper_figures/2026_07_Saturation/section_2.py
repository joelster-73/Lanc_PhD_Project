# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""

# %% lags

from src.methods.saturation.plotting.general import plot_delay_hists

# Plots not to be included, but numbers are.

plot_delay_hists('combined', ('msh','sw'), resolution='1min')
plot_delay_hists('combined', ('msh','sw'), resolution='5min')

