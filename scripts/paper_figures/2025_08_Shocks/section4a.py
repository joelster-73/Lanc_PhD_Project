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

from src.methods.shock_intercepts.training import train_algorithm_param, plot_single_param_vary

coeff_lim = 0.9

buffer_up = 33
buffer_dw = 35

distance_buff = 80
min_ratio_change = 0.85


### CHANGE TO BE A GRID PLOT

for vary in ('buffer_up','buffer_dw','min_ratio','dist_buff'):

    fit_slopes, fit_ints, fit_R2s, fit_counts, vary_array = train_algorithm_param(helsinki_events, vary=vary, vary_array=None, buffer_up=buffer_up, buffer_dw=buffer_dw, distance_buff=distance_buff, min_ratio_change=min_ratio_change, coeff_lim=coeff_lim)

    plot_single_param_vary(vary_array, slopes_fit=fit_slopes, slopes_int=fit_ints, counts=fit_counts, slopes_R2=fit_R2s, ind_var=vary, coeff_lim=coeff_lim)


 # %% Optimal_parameters

from src.analysing.shocks.training import analyse_all_events
buffer_up = 33
buffer_dw = 35

distance_buff = 80
min_ratio_change = 0.85

### CHANGE TO BE A DATAFRAME

correlated_delays, helsinki_delays, detection_times, coefficients, detectors, interceptors, event_numbers = analyse_all_events(helsinki_events, buffer_up=buffer_up, buffer_dw=buffer_dw, distance_buff=distance_buff, min_ratio_change=min_ratio_change)

# %% Comparing

from src.analysing.shocks.training import plot_comparison

coeff_lim = 0.9

colour_style = 'sc' # coeff or sc

### CHANGE TO USE COMPARE SERIES PROCEDURE

plot_comparison(helsinki_delays, correlated_delays, coefficients, detectors, interceptors, event_nums=event_numbers, coeff_lim=coeff_lim, colouring=colour_style)