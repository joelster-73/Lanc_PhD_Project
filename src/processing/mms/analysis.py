# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 16:42:01 2025

@author: richarj2
"""
import os
import pandas as pd
from ...config import get_proc_directory

# %% Region_intervals

def mms_region_intervals(region='sw', max_gap=pd.Timedelta('30h')):

    mms_dir = get_proc_directory('mms1') # crossings in mms1 folder for all sc

    if region=='msh':
        direction = 1 # inbound from BS
    elif region=='sw':
        direction = -1 # outbound from BS

    file_path = os.path.join(mms_dir,'Lalti_2022_BS_crossings.csv')

    crossings = pd.read_csv(file_path, skiprows=53)
    crossings = crossings[['#time','direction']]
    crossings['time'] = pd.to_datetime(crossings['#time'],unit='s')

    crossings.set_index('time',inplace=True)
    crossings.sort_index(inplace=True)
    crossings.drop(columns='#time',inplace=True)

    starts = crossings.index[(crossings['direction'] == direction) & (crossings['direction'].shift(-1) == -direction)]  # next direction is opposite
    valid = crossings.index.get_indexer(starts) + 1
    valid = valid[valid < len(crossings)]  # avoid IndexError
    ends = crossings.index[valid]

    intervals = pd.DataFrame({'start': starts.values, 'end': ends.values})
    intervals = intervals.dropna()
    intervals = intervals[(intervals['end'] - intervals['start']) <= max_gap]

    return list(intervals.itertuples(index=False, name=None))