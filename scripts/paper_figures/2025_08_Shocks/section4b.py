# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 10:30:06 2025

@author: richarj2
"""
# %% Import

from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data

shocks_with_intercepts = import_processed_data(PROC_SHOCKS_DIR, 'shocks_with_intercepts.cdf')

# %% Scatter_and_hist
from src.methods.shock_intercepts.plotting import plot_time_differences

plot_time_differences(shocks_with_intercepts, selection='all', x_axis='delta_x', colouring='spacecraft')
plot_time_differences(shocks_with_intercepts, selection='earth', x_axis='delta_x', colouring='spacecraft')

# %% Normals

shocks_with_normals = import_processed_data(PROC_SHOCKS_DIR, file_name='shocks_with_normals.cdf')


plot_time_differences(shocks_with_intercepts, bottom_panel=None, selection='earth', x_axis='delta_n', colouring='spacecraft', shock_normals=shocks_with_normals)

plot_time_differences(shocks_with_intercepts, bottom_panel=None, selection='earth', x_axis='delta_t', colouring='spacecraft', shock_normals=shocks_with_normals)