# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:42:01 2025

@author: richarj2
"""
import os
import pandas as pd


# %% Region_intervals

def mms_region_intervals(mms_dir, region='sw'):

    if region=='msh':
        direction = 1 # inbound from BS
    elif region=='sw':
        direction = -1 # outbound from BS

    file_path = os.path.join(mms_dir,'Lalti_2022_BS_crossings.csv')

    crossings = pd.read_csv(file_path, skiprows=53)
    crossings = crossings[['#time','direction']]
    crossings['time'] = pd.to_datetime(crossings['#time'],unit='s')
    crossings.set_index('time',inplace=True)
    crossings.drop(columns='#time',inplace=True)

    starts = crossings.index[(crossings['direction'] == direction) & (crossings['direction'].shift(1) != direction)]  # previous direction != 1
    ends   = crossings.index[(crossings['direction'] == direction) & (crossings['direction'].shift(-1) != direction)] # next direction != 1
    m1_intervals = list(zip(starts, ends))


    return m1_intervals