# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 08:43:58 2025

@author: richarj2
"""

# %% Imports
from src.config import HELSINKI_DIR
from src.processing.reading import import_processed_data
from src.processing.shocks.helsinki import convert_helsinki_df_plotting

helsinki_shocks = import_processed_data(HELSINKI_DIR)
helsinki_events = convert_helsinki_df_plotting(helsinki_shocks)

# %% Compression

from src.plotting.shocks import plot_compressions_both

# Do similar plot for angle between normal vectors
plot_compressions_both(helsinki_shocks, plot_type='hist')


# %% Scatter_and_hist
from src.methods.shock_intercepts.plotting import plot_propagations_both

plot_propagations_both(helsinki_events, x_axis='delta_x', colouring='spacecraft')
