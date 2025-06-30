# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025


@author: richarj2
"""



# %% Imports
from src.config import HELSINKI_DIR
from src.processing.reading import import_processed_data
from src.processing.shocks.helsinki import get_list_of_events

helsinki_shocks = import_processed_data(HELSINKI_DIR)
event_list = get_list_of_events(helsinki_shocks)


# %% Find_delays
from src.analysing.shocks.training import analyse_all_events_more_info
buffer_up = 33
buffer_dw = 35

distance_buff = 60
min_ratio_change = 0.8

helsinki_delays, correlated_delays, coefficients, shock_times, detectors, interceptors, modal_omni_sc = analyse_all_events_more_info(helsinki_shocks, event_list, buffer_up, buffer_dw, distance_buff=distance_buff, min_ratio_change=min_ratio_change)

# %% Plot_comparison
from src.analysing.shocks.training import plot_comparison, plot_differences_over_time

coeff_lim = 0.7

for colour_style in ('coeff','sc'):

    plot_comparison(helsinki_delays, correlated_delays, coefficients, detectors, interceptors, coeff_lim=coeff_lim, colouring=colour_style, modal_omni=modal_omni_sc, title_info_dict={'buffer_up': buffer_up, 'buffer_dw': buffer_dw, 'dist_buff': distance_buff, 'min_ratio': min_ratio_change})

    plot_differences_over_time(helsinki_delays, correlated_delays, shock_times, coefficients, detectors, interceptors, coeff_lim=coeff_lim, colouring=colour_style, modal_omni=modal_omni_sc, title_info_dict={'buffer_up': buffer_up, 'buffer_dw': buffer_dw, 'dist_buff': distance_buff, 'min_ratio': min_ratio_change})

# %% Odd_times

from datetime import timedelta, datetime
from src.plotting.shocks import plot_dict_times



# Change slightly so that the same events are plotted together???
# Do stats to find the min time difference when detector is upstream for rough restriction on time upstream

bad_mask = (np.abs(helsinki_delays-correlated_delays)>=10) & (coefficients>=coeff_lim) # above coefficient threshold

shock_times_bad = shock_times[bad_mask]
corr_delays_bad  = correlated_delays[bad_mask]
helsinki_delays_bad = helsinki_delays[bad_mask]
coeffs_bad = coefficients[bad_mask]

detectors_bad = detectors[bad_mask]
interceptors_bad = interceptors[bad_mask]


for i in range(len(shock_times_bad)):

    shock_time  = shock_times_bad[i]
    detector    = detectors_bad[i]
    interceptor = interceptors_bad[i]
    bad_time    = shock_time[0]+timedelta(minutes=corr_delays_bad[i].n)
    good_time   = shock_time[0]+timedelta(minutes=helsinki_delays_bad[i].n)


    bad_dict = {detector: {'time': shock_time[0], 'time_unc': shock_time[1]}, interceptor: {'time': good_time, 'time_unc': 60*helsinki_delays_bad[i].s}, interceptor+' [Guess]': {'time': bad_time, 'time_unc': 60*corr_delays_bad[i].s, 'coeff': coeffs_bad[i]}}

    plot_dict_times(bad_dict, detector, 'B_mag', time_window=30, plot_in_sw=False, plot_full_range=True)
