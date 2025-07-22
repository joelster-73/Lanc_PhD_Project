# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 15:16:19 2025

@author: richarj2
"""

# %% Imports
from src.config import HELSINKI_DIR
from src.processing.reading import import_processed_data
from src.processing.shocks.helsinki import convert_helsinki_df_training
from src.analysing.shocks.training import analyse_all_events

helsinki_shocks = import_processed_data(HELSINKI_DIR)
helsinki_events = convert_helsinki_df_training(helsinki_shocks)

information = analyse_all_events(helsinki_events)

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



# %%