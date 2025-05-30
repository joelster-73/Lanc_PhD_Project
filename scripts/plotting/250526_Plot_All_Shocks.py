# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""
from src.config import PROC_CFA_DIR
from src.processing.reading import import_processed_data


shocks = import_processed_data(PROC_CFA_DIR)


# %%
from datetime import datetime
from src.plotting.shocks import plot_all_shocks

plot_all_shocks(shocks,'B_mag')

# %%

from src.plotting.shocks import plot_shock_times

parameter = 'B_mag'

time_choice = datetime(2013,3,17)
nearest_idx = shocks.index.searchsorted(time_choice, side='right')
nearest_time = shocks.index[nearest_idx]
shock = shocks.loc[nearest_time]


parameter = 'B_mag'
plot_shock_times(shock, parameter)


# %%

#mimport numpy as np



# def plot_window(ax, interval):
#     ax.axvline(interval[0],c='k',ls=':')
#     ax.axvline(interval[1],c='k',ls=':')
#     ax.axvspan(interval[0],interval[1],color='b',alpha=0.05)


# def shock_b_mag(shock, loc):

#     # "where" = "up" or "dw"
#     # only approximate

#     return np.sqrt(shock[f'B_x_GSE_{loc}']**2+shock[f'B_z_GSE_{loc}']**2+shock[f'B_y_GSE_{loc}']**2)


# for interval in (shock_window_up, shock_window_dw):
#     plot_window(ax, interval)
#     data_slice = df_detect[(df_detect.index>=interval[0]) & (df_detect.index <interval[1])].to_numpy()
#     average = ufloat(np.mean(data_slice),np.std(data_slice))
#     ax.plot([interval[0],interval[1]],[average.n,average.n],c='k')
#     ax.text(interval[0],average.n*1.1,average)

#def find_shock(df_sc, detection_zone, min_width=timedelta(minutes=5), min_gap=timedelta(minutes=0), resolution=timedelta(seconds=60)):

    #all_windows





