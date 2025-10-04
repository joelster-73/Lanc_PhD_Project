# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 08:43:58 2025

@author: richarj2
"""

# %% FIND_ERRORS
import warnings
warnings.simplefilter('error', RuntimeWarning)



# %% Imports
from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data
from src.processing.shocks.helsinki import convert_helsinki_df_plotting

helsinki_shocks = import_processed_data(PROC_SHOCKS_DIR, file_name='helsinki_shocks.cdf')
helsinki_events = convert_helsinki_df_plotting(helsinki_shocks)

# %% Compression

from src.methods.shock_intercepts.plotting import plot_compressions_both, plot_omni_compressions

# Do similar plot for angle between normal vectors
plot_compressions_both(helsinki_shocks, plot_type='hist')

plot_omni_compressions(helsinki_shocks, plot_type='hist')
# %% Scatters
from src.methods.shock_intercepts.plotting import plot_propagations_both

plot_propagations_both(helsinki_events, x_axis='delta_x', colouring='spacecraft')

