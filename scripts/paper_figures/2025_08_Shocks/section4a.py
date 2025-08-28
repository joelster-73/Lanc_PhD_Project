# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 10:30:06 2025

@author: richarj2
"""
# %% Imports
from src.config import HELSINKI_DIR
from src.processing.reading import import_processed_data
from src.processing.shocks.helsinki import convert_helsinki_df_plotting

helsinki_shocks = import_processed_data(HELSINKI_DIR)
helsinki_events = convert_helsinki_df_plotting(helsinki_shocks)

 # %% Training

from src.methods.shock_intercepts.training import plot_grid_param_vary

coeff_lim = 0.9

buffer_up = 33
buffer_dw = 35

distance_buff = 80
min_ratio_change = 0.85


plot_grid_param_vary(helsinki_events, 'dist_buff', 'min_ratio', 'buffer_dw', 'buffer_up', buffer_up=buffer_up, buffer_dw=buffer_dw, distance_buff=distance_buff, min_ratio_change=min_ratio_change, coeff_lim=coeff_lim)


 # %% Optimal_parameters

from src.methods.shock_intercepts.training import analyse_all_events
buffer_up = 33
buffer_dw = 35

distance_buff = 80
min_ratio_change = 0.85

df_trained_params = analyse_all_events(helsinki_events, buffer_up=buffer_up, buffer_dw=buffer_dw, distance_buff=distance_buff, min_ratio_change=min_ratio_change)


# %% Comparing

from src.methods.shock_intercepts.training import plot_comparison

coeff_lim = 0.9

colour_style = 'sc' # coeff or sc

plot_comparison(df_trained_params, coeff_lim=coeff_lim, colouring=colour_style)