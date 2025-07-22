# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 10:26:21 2025

@author: richarj2
"""

# %% Imports
from src.config import HELSINKI_DIR
from src.processing.reading import import_processed_data
from src.processing.shocks.helsinki import convert_helsinki_df

helsinki_shocks = import_processed_data(HELSINKI_DIR)
helsinki_events = convert_helsinki_df(helsinki_shocks)

# %%
from src.plotting.shocks import plot_time_differences

plot_time_differences(helsinki_events, selection='all', x_axis='x_comp', colouring='spacecraft', histograms=True, histogram_fits=False)
plot_time_differences(helsinki_events, selection='earth', x_axis='x_comp', colouring='spacecraft', histograms=True)