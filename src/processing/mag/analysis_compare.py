# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:42:09 2025

@author: richarj2
"""


from src.processing.reading import import_processed_data
from src.processing.mag.plotting import plot_compare_magnetometers
from src.methods.magnetosheath_saturation.merge_region_sc import update_omni

sample_interval = '5min'
data_pop        = 'with_plasma'
region          = 'sw'
spacecraft      = 'shifted'

df_omni = import_processed_data('omni', resolution=sample_interval)
update_omni(df_omni)

#df_sc   = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_{spacecraft}')
df_pc   = import_processed_data('indices', file_name=f'combined_{sample_interval}')

# %% plot

plot_compare_magnetometers(df_omni, df_pc, 'E_R', 'mag', lag=17)
plot_compare_magnetometers(df_omni, df_pc, 'E_y_GSM', 'gsm', lag=17)

