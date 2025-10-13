# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 09:33:56 2025

@author: richarj2
"""


from src.config import CROSSINGS_DIR
from src.processing.reading import import_processed_data
from src.plotting.distributions import plot_counts
from src.plotting.utils import change_series_name


# %% Grison_MSH_Times

crossings = import_processed_data(CROSSINGS_DIR)
cross_labels = crossings.attrs['crossings']
for ind, label in cross_labels.items():
    if label=='N/A':
        continue
    elif label=='UNKNOWN':
        cross_labels[ind] = 'UKN'
        continue
    cross_labels[ind] = label.replace('/','\n')

total_durations = crossings.groupby('loc_num')['region_duration'].sum()/60
total_durations.attrs['units']['region_duration'] = 'Mins'
change_series_name(total_durations,'Region')

plot_counts(total_durations, labels=cross_labels, add_percs=True)
