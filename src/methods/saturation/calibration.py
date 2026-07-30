# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:10:20 2026

@author: richarj2
"""

# %% tester
import numpy as np

from src.processing.mms.analysis import mms_region_intervals
from src.methods.saturation.filter import data_populations, filter_region
#from src.methods.saturation.filter import filter_sc_region
from src.processing.reading import import_processed_spacecraft, import_processed_data

data_pop = 'plasma'
resolution = '1min'
sc = 'mms1'
region = 'sw'

param = 'N_tot' if region=='msh' else 'V_flow'

df_omni = import_processed_data('omni', resolution=resolution)

toy_eden_times = mms_region_intervals(sc, region)

earliest = toy_eden_times.left[0]
latest = toy_eden_times.right[-1]

#mms = filter_sc_region(sc, region, data_pop=data_pop, resolution=resolution)

total_duration = (toy_eden_times.right - toy_eden_times.left).sum()
total_mins = total_duration.total_seconds() / 60

populations = data_populations(sc, data_pop, region)

df_sc = import_processed_spacecraft(sc, populations, resolution)

times_with_plasma  = toy_eden_times.get_indexer(df_sc.loc[~df_sc[param].isna()].index) != -1

# %% num

## create loop over densities or msh depth params and plot to see how sensitive
# save this in MMS folder somewhere so can refer back to when writing

df_merged, _ = filter_region(df_sc, df_omni, sc, region)

length0 = len(df_merged)

df_merged = df_merged.loc[(df_merged.index>=earliest)&(df_merged.index<=latest)] # so data outside range doesn't conflate stats
length = len(df_merged)

inside_model  = toy_eden_times.get_indexer(df_merged.index) != -1
outside_model = ~inside_model


print(f'UNIQUE TIMES in {region} [mins]')
print('------------')
print(f'Toy-Edens database in region:                  {int(total_mins):,}')
print(f'Toy-Edens database when plasma data:           {np.sum(times_with_plasma):,}')
print(f'Simple method data in region:                  {length0:,}')
print(f'Simple method data in Toy-Edens time range:    {length:,}  | {100*length/np.sum(times_with_plasma):.2f}%')
print(f'Simple method data in Toy-Edens (good):        {np.sum(inside_model):,}  | {100*np.sum(inside_model)/length:.2f}%')
print(f'Simple method data not in Toy-Edens (contam?): {np.sum(outside_model):,}    | {100*np.sum(outside_model)/length:.2f}%')