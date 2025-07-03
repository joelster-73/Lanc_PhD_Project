# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 15:27:14 2025

@author: richarj2
"""

from src.processing.reading import import_processed_data
from src.config import PROC_INTERCEPTS_DIR

shocks_with_intercepts = import_processed_data(PROC_INTERCEPTS_DIR)
# %%

from src.plotting.shocks import plot_time_differences, plot_time_histogram

plot_time_differences(shocks_with_intercepts, selection='earth', x_axis='L1_rho', colouring='spacecraft')




# %%
plot_time_differences(shocks_with_intercepts, selection='all', x_axis='x_comp', colouring='spacecraft')
plot_time_differences(shocks_with_intercepts, selection='closest', x_axis='x_comp', colouring='spacecraft')
plot_time_differences(shocks_with_intercepts, selection='earth', x_axis='x_comp', colouring='spacecraft')
plot_time_differences(shocks_with_intercepts, selection='closest', x_axis='x_comp', colouring='spacecraft', max_dist=60)

plot_time_histogram(shocks_with_intercepts, selection='all')
plot_time_histogram(shocks_with_intercepts, selection='closest')
plot_time_histogram(shocks_with_intercepts, selection='earth')
plot_time_histogram(shocks_with_intercepts, selection='closest', max_dist=60)

