# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""

from src.methods.saturation.plotting import plot_driver_multi_responses

param_names  = {'E_y_GSM': 'E_y',
                'V_flow' : 'V',
                'B_z_GSM': 'B_z',
                'N_tot'  : 'N'}

responses = ('THL','SME','SMR')
param     = 'E_R'

for bottom_axis in ('heat','scatter','hist'):
    plot_driver_multi_responses(param, *responses, lags=(17,53,60), restrict=True, bottom_axis=bottom_axis, data1_name=param_names.get(param,param), data_name_map=param_names)


# %% Columns
from src.processing.reading import import_processed_data

df_sc = import_processed_data('sw', dtype='plasma', resolution='5min', file_name='sw_times_combined')


df = import_processed_data('omni', resolution='15min')



# %% tester
import numpy as np

from src.processing.mms.analysis import mms_region_intervals
from src.methods.saturation.filter import data_populations, filter_region
#from src.methods.saturation.filter import filter_sc_region
from src.processing.reading import import_processed_spacecraft, import_processed_data

data_pop = 'plasma'
resolution = '1min'
sc = 'mms1'
region = 'msh'

param = 'N_tot' if region=='msh' else 'V_flow'

df_omni = import_processed_data('omni', resolution=resolution)

toy_eden_times = mms_region_intervals(sc, region)

#mms = filter_sc_region(sc, region, data_pop=data_pop, resolution=resolution)

total_duration = (toy_eden_times.right - toy_eden_times.left).sum()
total_mins = total_duration.total_seconds() / 60

populations = data_populations(sc, data_pop, region)



df_sc = import_processed_spacecraft(sc, populations, resolution)

# %% num

df_merged, _ = filter_region(df_sc, df_omni, sc, region)

times_with_plasma  = toy_eden_times.get_indexer(df_sc.loc[~df_sc[param].isna()].index) != -1

inside_model  = toy_eden_times.get_indexer(df_merged.index) != -1
outside_model = ~inside_model


print('UNIQUE TIMES')
print('------------')
print(f'Mins of model times:                      {int(total_mins):,}')
print(f'Mins of model time when plasma data:      {np.sum(times_with_plasma):,}')
print(f'Mins of simple method data:               {len(df_merged):,}  | {100*len(df_merged)/np.sum(times_with_plasma):.2f}%')
print(f'Mins of simple method in model times:     {np.sum(inside_model):,}  | {100*np.sum(inside_model)/len(df_merged):.2f}%')
print(f'Mins of simple method not in model times: {np.sum(outside_model):,}   | {100*np.sum(outside_model)/len(df_merged):.2f}%')
