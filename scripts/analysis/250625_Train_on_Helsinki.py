# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 15:16:19 2025

@author: richarj2
"""

# %% Imports
from src.config import HELSINKI_DIR
from src.processing.reading import import_processed_data

helsinki_shocks = import_processed_data(HELSINKI_DIR)

# %% Events
from src.processing.shocks.helsinki import get_list_of_events
event_list = get_list_of_events(helsinki_shocks)

# %%
from src.analysing.shocks.training import train_algorithm_buffers

buffer_up_range = range(10,36)
buffer_dw_range = range(20,41)
coeff_lim       = 0.7
num_events      = len(event_list)


slopes, counts, r2_val =  train_algorithm_buffers(helsinki_shocks, event_list, buffer_up_range, buffer_dw_range, coeff_lim)
# %%
from src.analysing.shocks.training import plot_buffer_training
structures = {'slopes': slopes, 'counts': counts, 'R2': r2_val}

for limits in (None,):
    plot_buffer_training(structures, limits, buffer_up_range, buffer_dw_range, coeff_lim, num_events)

 # %%
from src.analysing.shocks.training import analyse_all_events_more_info
buffer_up = 33
buffer_dw = 35

distance_buff = 60
min_ratio_change = 0.8

helsinki_delays, correlated_delays, coefficients, shock_times, detectors, interceptors, modal_omni_sc = analyse_all_events_more_info(helsinki_shocks, event_list, buffer_up, buffer_dw, distance_buff=distance_buff, min_ratio_change=min_ratio_change)

# %%
from src.analysing.shocks.training import plot_comparison

coeff_lim = 0.7

for colour_style in ('coeff','sc'):

    plot_comparison(helsinki_delays, correlated_delays, coefficients, detectors, interceptors, coeff_lim=coeff_lim, colouring=colour_style, modal_omni=modal_omni_sc, title_info_dict={'buffer_up': buffer_up, 'buffer_dw': buffer_dw, 'dist_buff': distance_buff, 'min_ratio': min_ratio_change})


 # %% Plot
from src.analysing.shocks.training import train_algorithm_param, plot_single_param_vary

buffer_up = 33
buffer_dw = 35

distance_buff = 60
min_ratio_change = 0.8

for vary in ('min_ratio','dist_buff'):

    slope_fits, num_shocks, slope_R2s, vary_array = train_algorithm_param(helsinki_shocks, event_list, vary=vary, vary_array=None, buffer_up=buffer_up, buffer_dw=buffer_dw, dist_buff=distance_buff, min_ratio_change=min_ratio_change, coeff_lim=coeff_lim)


    plot_single_param_vary(vary_array, slopes_fit=slope_fits, counts=num_shocks, slopes_R2=slope_R2s, ind_var='min_ratio', title_info_dict={'buffer_up': buffer_up, 'buffer_dw': buffer_dw, 'dist_buff': distance_buff})

# %%