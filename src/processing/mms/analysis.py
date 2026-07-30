# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 10:49:57 2026

@author: richarj2
"""
import os
import pandas as pd
from ...config import get_proc_directory

def mms_region_intervals(spacecraft, region='sw'):
    """
    Returns intervals when the mms spacecraft is in a particular magnetosphere region according to Toy-Edens model
    """

    region_map = {'sw': 'solar_wind', 'msh': 'magnetosheath', 'ms': 'magnetosphere', 'ion': 'ion_foreshock'}

    directory = get_proc_directory(spacecraft, dtype='crossings')
    file_name=f'{region_map.get(region)}_region_list.csv'

    file_path = os.path.join(directory, file_name)

    crossings = pd.read_csv(file_path)
    crossings['start'] = pd.to_datetime(crossings['start'])
    crossings['stop'] = pd.to_datetime(crossings['stop'])

    crossings = crossings.dropna(subset=['start', 'stop'])
    crossings = crossings.loc[crossings['probe']==spacecraft]

    times = list(zip(crossings['start'], crossings['stop']))
    mms_intervals = pd.IntervalIndex.from_tuples(times, closed='both')

    total_duration = (mms_intervals.right - mms_intervals.left).sum()
    mins = total_duration.total_seconds() / 60
    print(f'Total duration of {spacecraft} in {region}: {total_duration} | {int(mins):,}')

    return mms_intervals