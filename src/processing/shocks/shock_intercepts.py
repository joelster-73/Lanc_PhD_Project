# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 14:52:14 2025

@author: richarj2
"""

from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data

all_processed_shocks = import_processed_data(PROC_SHOCKS_DIR)

# %% Initialise
from src.analysing.shocks.intercepts import find_all_shocks
shocks_intercepted = find_all_shocks(all_processed_shocks, 'B_mag')

# %%
import os
from src.processing.writing import write_to_cdf
from src.config import PROC_INTERCEPTS_DIR
output_file = os.path.join(PROC_INTERCEPTS_DIR, 'shocks_with_intercepts.cdf')

write_to_cdf(shocks_intercepted, output_file, attributes={'time_col': 'none'})

# %%

shocks_with_intercepts = import_processed_data(PROC_INTERCEPTS_DIR)
# %%

from src.plotting.shocks import plot_time_differences, plot_time_histogram

plot_time_differences(shocks_with_intercepts, selection='all', x_axis='x_comp', colouring='spacecraft')
plot_time_differences(shocks_with_intercepts, selection='closest', x_axis='x_comp', colouring='spacecraft')
plot_time_differences(shocks_with_intercepts, selection='earth', x_axis='x_comp', colouring='spacecraft')
plot_time_differences(shocks_with_intercepts, selection='closest', x_axis='x_comp', colouring='spacecraft', max_dist=60)

plot_time_histogram(shocks_with_intercepts, selection='all')
plot_time_histogram(shocks_with_intercepts, selection='closest')
plot_time_histogram(shocks_with_intercepts, selection='earth')
plot_time_histogram(shocks_with_intercepts, selection='closest', max_dist=60)

