# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 10:26:21 2025

@author: richarj2
"""

# %% Imports
from src.config import HELSINKI_DIR
from src.processing.reading import import_processed_data
from src.processing.shocks.helsinki import get_list_of_events_simple

helsinki_shocks = import_processed_data(HELSINKI_DIR)
event_list = get_list_of_events_simple(helsinki_shocks)


# %%
import pandas as pd
import numpy as np

from src.processing.speasy.config import sw_monitors

unique_spacecraft = list({key for d in event_list for key in d})
all_spacecraft = [sc for sc in sw_monitors if sc in unique_spacecraft]

helsinki_events = pd.DataFrame()
new_columns = {}

new_columns['detectors'] = ''
for sc in (sw_monitors + ('OMNI',)):

    new_columns[f'{sc}_time'] = pd.NaT
    new_columns[f'{sc}_time_unc_s'] = np.nan

    for comp in ('x','y','z'):
        new_columns[f'{sc}_r_{comp}_GSE'] = np.nan
        new_columns[f'{sc}_r_{comp}_GSE_unc'] = np.nan

    if sc=='OMNI':
        new_columns['OMNI_sc'] = ''

eventIDs = range(1,len(event_list)+1)
helsinki_events = pd.concat([helsinki_events, pd.DataFrame(new_columns, index=eventIDs)], axis=1)

# %%
from src.processing.speasy.retrieval import retrieve_omni_value
from src.processing.speasy.config import speasy_variables

for ind, shock_dict in enumerate(event_list):

    event_num = ind+1

    detectors = [sc for sc in (sw_monitors + ('OMNI',)) if sc in list(shock_dict.keys())]
    helsinki_events.at[event_num,'detectors'] = ','.join(detectors)

    for sc, sc_info in shock_dict.items():
        helsinki_events.at[event_num, f'{sc}_time'] = sc_info[0]
        helsinki_events.at[event_num, f'{sc}_time_unc_s'] = sc_info[1]
        position = helsinki_shocks.loc[sc_info[0],['r_x_GSE','r_y_GSE','r_z_GSE']]
        if isinstance(position, pd.DataFrame):
            position = position.iloc[0].to_numpy()
        else:
            position = position.to_numpy()
        if np.isnan(position[0]):
            continue
        helsinki_events.loc[event_num,[f'{sc}_r_x_GSE',f'{sc}_r_y_GSE',f'{sc}_r_z_GSE']] = position
        if sc=='OMNI':
            helsinki_events.at[event_num,'OMNI_sc'] = retrieve_omni_value(speasy_variables, sc_info[0], omni_var='OMNI_sc')

# %%
from src.plotting.shocks import plot_time_differences

plot_time_differences(helsinki_events, selection='all', x_axis='x_comp', colouring='spacecraft', histograms=True, histogram_fits=False)
plot_time_differences(helsinki_events, selection='earth', x_axis='x_comp', colouring='spacecraft', histograms=True)