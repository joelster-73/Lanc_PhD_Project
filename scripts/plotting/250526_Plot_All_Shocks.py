# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""
from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data

shocks = import_processed_data(PROC_SHOCKS_DIR)


# %%
from datetime import datetime
from src.plotting.shocks import plot_all_shocks

plot_all_shocks(shocks,'B_mag')

# %%

from src.plotting.shocks import plot_shock_times

parameter = 'B_mag'

time_choice = datetime(2013,3,17)
nearest_idx = shocks.index.searchsorted(time_choice, side='right')
nearest_time = shocks.index[nearest_idx]
shock = shocks.loc[nearest_time]


parameter = 'B_mag'
plot_shock_times(shock, parameter)


# %%
from src.plotting.shocks import plot_time_differences

plot_time_differences(shocks, coeff_lim=0.8, selection='earth', x_axis='x_comp', colouring='spacecraft')

# %%dss
plot_all_shocks(shocks, 'B_mag', plot_in_sw=True, plot_positions=True, start_printing=datetime(2011,12,31))


