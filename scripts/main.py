# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""


#

# %% Importing
from src.config import PROC_CFA_DIR
from src.processing.reading import import_processed_data
from src.processing.cfa.donki import combine_cfa_donki

cfa_shocks = import_processed_data(PROC_CFA_DIR)
shocks = combine_cfa_donki(cfa_shocks)

# %% Importing
from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data
from src.analysing.shocks.discontinuities import find_closest

shocks_intercepts = import_processed_data(PROC_SHOCKS_DIR)
find_closest(shocks_intercepts)



# %%
from src.plotting.shocks import plot_time_differences, plot_time_histogram

plot_time_differences(shocks_intercepts, coeff_lim=0.8, selection='earth', x_axis='x_comp', colouring='spacecraft')
plot_time_differences(shocks_intercepts, coeff_lim=0.8, selection='earth', x_axis='x_comp', colouring='coeff')

plot_time_histogram(shocks_intercepts, coeff_lim=0.8, selection='earth', show_best_fit=False, show_errors=True, colouring='none')


# %%
import numpy as np
import pandas as pd

coeff_lim = 0.7
sc_labels = [col.split('_')[0] for col in shocks_intercepts if '_coeff' in col]

only_Earth = True

indices = []
for index, shock in shocks_intercepts.iterrows():

    detector = shock['spacecraft']

    BS_time     = shock['OMNI_time']
    if pd.isnull(BS_time):
        continue
    BS_coeff = shock['OMNI_coeff']
    if np.isnan(BS_coeff) or BS_coeff<coeff_lim or BS_coeff>1:
        #1.1 indicates exact matches
        continue

    add_shock = False
    for sc in sc_labels:
        if sc in ('OMNI',detector):
            continue

        elif (only_Earth) and sc in ('WIND','ACE','DSC'):
            continue

        corr_coeff = shock[f'{sc}_coeff']
        if isinstance(corr_coeff, (pd.Series, pd.DataFrame)) and len(corr_coeff) > 1:
            corr_coeff = corr_coeff.iloc[0]  # Get the first value
        else:
            corr_coeff = corr_coeff

        if np.isnan(corr_coeff) or corr_coeff<coeff_lim or corr_coeff>1:
            #1.1 indicates exact matches
            continue

        sc_time = shock[f'{sc}_time']
        if pd.isnull(sc_time):
            continue
        time_diff     = (shock[f'{sc}_time'] - BS_time).total_seconds()
        if np.abs(time_diff)>=(45*60): # positive or engative
            add_shock = True

    if add_shock:
        indices.append(index)

# %%Test

from src.plotting.shocks import plot_shock_times
from src.analysing.shocks.intercepts import find_all_shocks
from datetime import timedelta, datetime

## CHANGE TO PRINT VEC COMPONENTS


for shock_index in indices:
    shock = shocks_intercepts.loc[shock_index]

    plot_shock_times(shock, 'B_mag', time_window=30)

    test_shock = find_all_shocks(shocks, 'field', time=shock_index-timedelta(seconds=1))
    plot_shock_times(test_shock, 'B_mag', time_window=30)

    #plot_shock_positions(shock, 'B_mag')

