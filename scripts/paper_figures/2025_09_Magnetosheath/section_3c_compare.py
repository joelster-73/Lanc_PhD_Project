# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""
# Plotting why this occurs (so saturation overview with grouping)

# %% Import
from src.processing.reading import import_processed_data
from src.methods.magnetosheath_saturation.plotting import plot_compare_sources
from src.methods.magnetosheath_saturation.plotting_lags import plot_compare_sources_with_lags
from src.methods.magnetosheath_saturation.merge_region_sc import update_omni

sample_interval = '5min'
data_pop        = 'with_plasma'
region          = 'sw'
spacecraft      = 'shifted'

df_omni = import_processed_data('omni', resolution=sample_interval)
update_omni(df_omni)

df_sc   = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_{spacecraft}')
df_pc   = import_processed_data('indices', file_name=f'combined_{sample_interval}')



# %% Comparison_Grid
import itertools as it
for ind, dep in it.product(('E_R','E_y_GSM'),('PCNC','PCN','PCC','AE')):

    for fit in ('saturation','linear_flat','straight'):
        plot_compare_sources(df_omni, df_sc, df_pc, ind, dep, fit_type=fit)
        plot_compare_sources_with_lags(df_omni, df_sc, df_pc, ind, dep, fit_type=fit)


# %%

plot_compare_sources(df_omni, df_sc, df_pc, 'E_R', 'PCC', fit_type='saturation')
plot_compare_sources(df_omni, df_sc, df_pc, 'E_R', 'PCNC', fit_type='saturation')
plot_compare_sources(df_omni, df_sc, df_pc, 'E_y_GSM', 'PCN', fit_type='saturation')



# %%
from src.plotting.comparing.parameter import compare_columns

compare_columns(df_omni, 'E_R', 'E_y_GSM', display='heat', reference_line='x')
compare_columns(df_omni, 'E_y_GSM', 'E_R', display='heat', reference_line='x')
