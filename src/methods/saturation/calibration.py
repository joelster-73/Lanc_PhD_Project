# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:10:20 2026

@author: richarj2
"""

import numpy as np

from src.processing.mms.analysis import mms_region_intervals
from src.config import get_data_populations
from src.methods.saturation.filter import spacecraft_distance
from src.processing.reading import import_processed_spacecraft, import_processed_data
from src.processing.dataframes import merge_dataframes
from src.coordinates.boundaries import calc_msh_dist

pos_cols  = ['r_MP','r_BS','r_phi','r_F']


def LOCAL_filter_region(df_merged, sc, region, params={}):
    """
    Filter region of magnetosphere based on Blüthner et al. (2026) for solar wind and adapted for magnetosheath.

    NOTE: This doesn't explicitly remove ARTEMIS Data. This is written to the file, but is removed when the files are merged into a complete file.
    """
    default_params = {'sw':  {'r_x': 7, 'r_theta': np.pi/6, 'V_sw':  275},
                      'msh': {'r_theta': np.pi/4, 'N_tot': 6, 'r_F_min': 0.05, 'r_F_max': 0.85, 'buffer': 'relative'}}

    for reg, vals in params.items():
        default_params[reg].update(vals)


    ####  BEHVAIOUR TAKEN OUT OF LOOP

    # ###-----Location-----###

    # df = df.loc[df['r_x_GSE']>0].copy()

    # spacecraft_distance(df)

    # df_merged_attrs = df.attrs # stores
    # df_merged = merge_dataframes(omni, df, suffices=('sw', sc))

    # ###-----Parameter-----###

    # df_positions = calc_msh_dist(df_merged, position_key=sc, data_key='sw')
    # df_merged    = pd.concat([df_merged, df_positions[pos_cols]], axis=1)

    rcone = default_params[region]['r_theta']

    if region=='sw':

        # Region Filtering

        rx    = default_params[region]['r_x']
        vflow = default_params['sw']['V_sw']

        mask =  (df_merged[f'r_x_GSE_{sc}']>=rx) & (df_merged[f'r_cone_{sc}']<=rcone)
        mask &= (df_merged[f'V_flow_{sc}'] >= vflow)

    elif region=='msh':

        # Region Filtering

        ntot   = default_params['msh']['N_tot']
        rFmin  = default_params['msh']['r_F_min']
        rFmax  = default_params['msh']['r_F_max']
        buffer = default_params['msh']['buffer']

        mask =  (df_merged[f'r_cone_{sc}']<=rcone)

        if buffer=='relative':
            mask &= (df_merged['r_F'] > rFmin) & (df_merged['r_F'] < rFmax) # tweak based on comparison with Toy-Edens
        # elif buffer=='absolute':
        #     mask &= (df_merged[f'r_mag_{sc}'] > df_merged['r_MP'] + rFmin) & (df_merged[f'r_mag_{sc}'] < df_merged['r_BS'] - rFmax)

        mask &= (df_merged[f'N_tot_{sc}'] >= ntot)

    # Post-processing Cleaning

    mask &= (df_merged[f'S_x_GSE_{sc}'] > -1000)
    mask &= (df_merged[f'V_x_GSE_{sc}'] < 0)

    return df_merged.loc[mask]

# %% import

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

populations = get_data_populations(sc, data_pop, region)

df_sc = import_processed_spacecraft(sc, populations, resolution)

times_with_plasma = toy_eden_times.get_indexer(df_sc.loc[~df_sc[param].isna()].index) != -1

# %% merge

import pandas as pd

# Done inside proper function call, done outside for efficiency here

df_merged_attrs = df_sc.attrs # stores
df_sc = df_sc.loc[df_sc['r_x_GSE']>0]

spacecraft_distance(df_sc)

df_merged = merge_dataframes(df_omni, df_sc, suffices=('sw', sc))

###-----Parameter-----###

df_positions = calc_msh_dist(df_merged, position_key=sc, data_key='sw')
df_merged    = pd.concat([df_merged, df_positions[pos_cols]], axis=1)

# %% defaults

default_params = {'msh': {'r_theta': np.pi/4, 'N_tot': 0, 'r_F_min': 0, 'r_F_max': 0.8, 'buffer': 'relative'}}

###-----NOTES-----###
# Maximising overlap not necessarily good - data not in Toy Edens not necessarily bad data.
# Absolute r_F buffers are worse than relative ones - lower overlap and less data.
# r_F_min percentage overlap peaks at 0.17, but only an increase of 1% from 0, and a large decrease in amount of data. Set to 0 as minimal gain.
# r_F_max percentage overlap peaks at 0.65. Get a large improvement going from 1.0 to 0.8 (2%) after which gain is minimal and data lossis large. Set to 0.8.


df_filtered = LOCAL_filter_region(df_merged, sc, region, default_params)

length0 = len(df_filtered)

df_filtered = df_filtered.loc[(df_filtered.index>=earliest)&(df_filtered.index<=latest)] # so data outside range doesn't conflate stats
length = len(df_filtered)

inside_model  = toy_eden_times.get_indexer(df_filtered.index) != -1
outside_model = ~inside_model

print(f'\nUNIQUE TIMES in {region} [mins]')
print('------------')
print(f'Toy-Edens database in region:                  {int(total_mins):,}')
print(f'Toy-Edens database when plasma data:           {np.sum(times_with_plasma):,}')
print(f'Simple method data in region:                  {length0:,}')
print(f'Simple method data in Toy-Edens time range:    {length:,}\t|\t{100*length/np.sum(times_with_plasma):.2f}%')
print(f'Simple method data in Toy-Edens (good?):       {np.sum(inside_model):,}\t|\t{100*np.sum(inside_model)/length:.2f}%')
print(f'Simple method data not in Toy-Edens (contam?): {np.sum(outside_model):,}\t|\t{100*np.sum(outside_model)/length:.2f}%')


# %% loops

# (1) Set density: 6
# (2) Set rFmin: 0.05
# (3) Set rFmax: 0.85

densities  = np.array([0, 1, 2, 3, 4, 5, 5.5, 6, 6.5, 7, 7.5, 8, 9, 10, 11, 12])
velocities = np.array([250, 275, 300])

rFmins = {'relative': np.array([0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20]),
          'absolute': np.array([0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])}
rFmaxs = {'relative': np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]),
          'absolute': np.array([1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20, 1.25])}

independents  = {'N_tot': {'absolute': densities},
                 'V_sw': {'absolute': velocities},
                 'r_F_min': rFmins, 'r_F_max': rFmaxs}

ind_param = 'r_F_max'
buffer    = 'relative'

independent = independents.get(ind_param).get(buffer)

percents  = np.zeros(len(independent))
totals    = np.zeros(len(independent))

###-----Loop-----###

for i, ind in enumerate(independent):

    df_filtered = LOCAL_filter_region(df_merged, sc, region, params={region: {ind_param: ind, 'buffer': buffer}})

    length0 = len(df_filtered)

    df_filtered = df_filtered.loc[(df_filtered.index>=earliest)&(df_filtered.index<=latest)] # so data outside range doesn't conflate stats
    length = len(df_filtered)

    inside_model  = toy_eden_times.get_indexer(df_filtered.index) != -1
    outside_model = ~inside_model

    print(f'\nUNIQUE TIMES in {region} [mins]')
    print(f'{ind_param} = {ind}')
    print('------------')
    print(f'Simple method data in region:                  {length0:,}')
    print(f'Simple method data in Toy-Edens time range:    {length:,}\t|\t{100*length/np.sum(times_with_plasma):.2f}%')
    print(f'Simple method data in Toy-Edens (good?):       {np.sum(inside_model):,}\t|\t{100*np.sum(inside_model)/length:.2f}%')
    print(f'Simple method data not in Toy-Edens (contam?): {np.sum(outside_model):,}\t|\t{100*np.sum(outside_model)/length:.2f}%')

    percents[i] = 100*np.sum(inside_model)/length
    totals[i]   = length

print(f'\nHighest overlap: {ind_param} = {independent[percents.argmax()]} ({percents.max():.2f}% | {int(totals[percents.argmax()]):,})')

# %% plotting

from src.methods.saturation.plotting.quality import plot_calibration

df = pd.DataFrame({ind_param: independent, 'perc': percents, 'count': totals})
if buffer == 'absolute':
    df.attrs = {'units': {'N_tot': 'n/cc', 'V_sw': 'km/s', 'r_F_min': 'Re', 'r_F_max': 'Re'}}

plot_calibration(df, ind_param, 'perc', 'count', buffer)

# %%