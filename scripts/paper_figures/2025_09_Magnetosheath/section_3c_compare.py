# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""
# Plotting why this occurs (so saturation overview with grouping)

# %% Import
from src.processing.reading import import_processed_data
from src.methods.magnetosheath_saturation.plotting import plot_compare_sources, plot_compare_sources_with_lags
from src.methods.magnetosheath_saturation.merge_region_sc import update_omni
from src.methods.magnetosheath_saturation.sc_delay_time import shift_sc_to_bs

sample_interval = '1min'
data_pop = 'with_plasma'

region = 'sw'

df_omni = import_processed_data('omni', resolution=sample_interval)
update_omni(df_omni)

spacecraft = 'combined'

df_sc = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_{spacecraft}')

# Consider shifting, resampling inc. circular avg, and saving to a dataframe
df_sc = shift_sc_to_bs(df_sc, sample_interval)

df_pc = import_processed_data('indices', file_name=f'combined_{sample_interval}')


# %% Comparison_Grid
import itertools as it
for ind, dep in it.product(('E_R','E_y_GSM'),('PCC','PCN','AE')):

    for fit in ('saturation','linear_flat','straight'):
        plot_compare_sources_with_lags(df_omni, df_sc, df_pc, ind, dep, fit_type=fit)
        plot_compare_sources(df_omni, df_sc, df_pc, ind, dep, fit_type=fit)


## odd behaviour with the 5-min data and lags that divide perfectly - interpolation introducing odd behaviour?

