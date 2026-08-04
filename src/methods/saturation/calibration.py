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
region = 'msh'

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

densities    = np.array([8,8.5,9,9.5,10,10.5,11,11.5,12])
velocities   = np.array([250,275,300])
rFmins       = np.array([-0.1, -0.05, 0, 0.05, 0.1, 0.15])
rFmaxs       = np.array([1.1, 1.05, 1, 0.95, 0.9, 0.85])
independents = {'N_tot': densities, 'V_sw': velocities, 'r_F_min': rFmins, 'r_F_max': rFmaxs}

ind_param = 'N_tot'

independent = independents.get(ind_param)

percents  = np.zeros(len(independent))
totals    = np.zeros(len(independent))

for i, ind in enumerate(independent):

    df_merged, _ = filter_region(df_sc, df_omni, sc, region, params={region: {ind_param: ind}})

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
    print(f'Simple method data in Toy-Edens (good?):       {np.sum(inside_model):,}  | {100*np.sum(inside_model)/length:.2f}%')
    print(f'Simple method data not in Toy-Edens (contam?): {np.sum(outside_model):,}    | {100*np.sum(outside_model)/length:.2f}%')

    percents[i] = 100*np.sum(inside_model)/length
    totals[i]   = length

# %% plotting

import pandas as pd
from src.methods.saturation.plotting.quality import plot_calibration

df = pd.DataFrame({ind_param: independent, 'perc': percents, 'count': totals})
df.attrs = {'units': {'N_tot': 'n/cc', 'V_sw': 'km/s'}}

plot_calibration(df, ind_param, 'perc', 'count')
