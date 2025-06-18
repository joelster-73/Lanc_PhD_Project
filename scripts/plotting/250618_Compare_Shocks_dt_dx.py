# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""


#

# %% Importing
from src.config import PROC_CFA_DIR
from src.processing.reading import import_processed_data
from src.processing.cfa.donki import combine_cfa_donki

cfa_shocks = import_processed_data(PROC_CFA_DIR)
shocks = combine_cfa_donki(cfa_shocks)

# %% Importing
from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data
from src.analysing.shocks.discontinuities import find_closest

shocks_intercepts = import_processed_data(PROC_SHOCKS_DIR)
find_closest(shocks_intercepts)


# %%
from src.plotting.shocks import plot_time_differences, plot_time_histogram

plot_time_differences(shocks_intercepts, coeff_lim=0.7, selection='earth', x_axis='x_comp', colouring='spacecraft')
plot_time_differences(shocks_intercepts, coeff_lim=0.7, selection='earth', x_axis='x_comp', colouring='coeff')
plot_time_differences(shocks_intercepts, coeff_lim=0.7, selection='earth', x_axis='x_comp', colouring='angle',     show_errors=False, cfa_shocks=cfa_shocks)
plot_time_differences(shocks_intercepts, coeff_lim=0.7, selection='earth', x_axis='x_comp', colouring='sun_earth', show_errors=False, cfa_shocks=cfa_shocks)


plot_time_histogram(shocks_intercepts, coeff_lim=0.7, selection='earth', show_best_fit=False, show_errors=True, colouring='none')




