# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:42:09 2025

@author: richarj2
"""


# %% import

from src.processing.reading import import_processed_data
from src.processing.mag.plotting import plot_magnetometer_overview

from datetime import datetime

omni = import_processed_data('omni', resolution='1min', year=2005)
omni = omni.shift(freq='17min')

THL      = import_processed_data('supermag', dtype='THL', resolution='gsm')
indices  = import_processed_data('indices', file_name='combined_1min')

# %% plot

times = [datetime(2006, 1, 1),  # Quiet day
         datetime(2008, 5, 10), # Quiet day
         datetime(2015, 3, 17), # Max PCN
         datetime(2024, 5, 11)  # Max PCN
]

for start_time in times:
    plot_magnetometer_overview(THL, omni, indices, start_time)


plot_magnetometer_overview(THL, omni, indices, datetime(2024, 5, 10), datetime(2024, 5, 12))
