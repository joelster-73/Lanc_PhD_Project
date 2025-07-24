# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 15:16:19 2025

@author: richarj2
"""

# %% Imports
from src.config import HELSINKI_DIR
from src.processing.reading import import_processed_data
from src.processing.shocks.helsinki import convert_helsinki_df_training

helsinki_shocks = import_processed_data(HELSINKI_DIR)
helsinki_events = convert_helsinki_df_training(helsinki_shocks)

 # %% Algorithm_all_shocks_compare

from src.analysing.shocks.training import analyse_all_events
buffer_up = 33
buffer_dw = 35

distance_buff = 60
min_ratio_change = 0.8

correlated_delays, helsinki_delays, detection_times, coefficients, detectors, interceptors = analyse_all_events(helsinki_events, buffer_up=buffer_up, buffer_dw=buffer_dw, distance_buff=distance_buff, min_ratio_change=min_ratio_change)

# %% Plotting_comparison
from src.analysing.shocks.training import plot_comparison

coeff_lim = 0.9

colour_style = 'sc' # coeff or sc

plot_comparison(helsinki_delays, correlated_delays, coefficients, detectors, interceptors, coeff_lim=coeff_lim, colouring=colour_style, title_info_dict={'buffer_up': buffer_up, 'buffer_dw': buffer_dw, 'dist_buff': distance_buff, 'min_ratio': min_ratio_change})


 # %% Testing_distance_and_ratio
from src.analysing.shocks.training import train_algorithm_param, plot_single_param_vary

buffer_up = 33
buffer_dw = 35

distance_buff = 80
min_ratio_change = 0.85

for vary in ('dist_buff','min_ratio'):

    fit_slopes, fit_ints, fit_R2s, fit_counts, vary_array = train_algorithm_param(helsinki_events, vary=vary, vary_array=None, buffer_up=buffer_up, buffer_dw=buffer_dw, distance_buff=distance_buff, min_ratio_change=min_ratio_change, coeff_lim=coeff_lim)


    plot_single_param_vary(vary_array, slopes_fit=fit_slopes, slopes_int=fit_ints, counts=fit_counts, slopes_R2=fit_R2s, ind_var=vary, coeff_lim=coeff_lim, title_info_dict={'buffer_up': buffer_up, 'buffer_dw': buffer_dw, 'dist_buff': distance_buff})
