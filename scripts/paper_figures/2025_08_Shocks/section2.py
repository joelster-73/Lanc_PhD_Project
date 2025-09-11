# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 10:32:24 2025

@author: richarj2
"""

from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data
from src.processing.shocks.helsinki import convert_helsinki_df_plotting

helsinki_shocks = import_processed_data(PROC_SHOCKS_DIR, file_name='helsinki_shocks.cdf')
helsinki_events = convert_helsinki_df_plotting(helsinki_shocks)

# %% Scatter_and_hist

from src.methods.shock_intercepts.plotting import plot_time_differences

plot_time_differences(helsinki_events, selection='all', x_axis='delta_x', colouring='spacecraft', right_fit=None, bottom_fit=None)


# %% Normals

from src.methods.shock_intercepts.plotting import plot_time_differences

helsinki_with_normals = import_processed_data(PROC_SHOCKS_DIR, file_name='shocks_with_normals.cdf')

plot_time_differences(helsinki_events, bottom_panel=None, selection='earth', x_axis='delta_n', colouring='spacecraft', shock_normals=helsinki_with_normals)
plot_time_differences(helsinki_events, bottom_panel=None, selection='earth', x_axis='delta_t', colouring='spacecraft', shock_normals=helsinki_with_normals)

