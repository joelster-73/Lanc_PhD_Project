# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 10:30:06 2025

@author: richarj2
"""
# %% Imports
from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data
from src.processing.shocks.helsinki import convert_helsinki_df_plotting

helsinki_shocks = import_processed_data(PROC_SHOCKS_DIR, file_name='helsinki_shocks.cdf')
helsinki_events = convert_helsinki_df_plotting(helsinki_shocks)

 # %% Training

from src.methods.shock_intercepts.training import plot_grid_param_vary

coeff_lim = 0.8

plot_grid_param_vary(helsinki_events, 'dist_buff', 'buffer_dw', 'min_ratio', 'buffer_up', coeff_lim=coeff_lim)


# %% Comparing

from src.methods.shock_intercepts.training import analyse_all_events, plot_comparison

df_trained_params = analyse_all_events(helsinki_events)

coeff_lim = 0.8
colour_style = 'sc' # coeff or sc

plot_comparison(df_trained_params, coeff_lim=coeff_lim, colouring=colour_style)

# %%