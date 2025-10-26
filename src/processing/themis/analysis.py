# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 13:34:11 2025

@author: richarj2
"""

import os
import numpy as np
import pandas as pd
from uncertainties import ufloat

from .config import PROC_THEMIS_DIRECTORIES
from ..reading import import_processed_data

def obtain_mp_boundaries(themis_dir, mp_file='Grimmich_2023_MP_Crossings.txt', resolution='5min'):


    thresholds = {'1min': 60, '5min': 300}
    threshold_seconds = thresholds.get(resolution)

    if mp_file=='Staples_2024_MP_Crossings.txt':
        file_path = os.path.join(themis_dir,mp_file)
        boundaries = pd.read_csv(file_path, skiprows=42, sep='\t')
        boundaries.sort_values(by='TIMESTAMP',inplace=True)
        boundaries['time'] = pd.to_datetime(boundaries['TIMESTAMP'])
        boundaries.set_index('time',inplace=True)
        boundaries.index = boundaries.index.tz_localize(None)
        boundaries.drop(columns='TIMESTAMP',inplace=True)
        boundaries['Probe'] = 'th' + boundaries['PROBE'].astype(str)

    elif mp_file=='Grimmich_2023_MP_Crossings.txt':
        file_path = os.path.join(themis_dir,mp_file)
        crossings = pd.read_csv(file_path, skiprows=27, sep='\t')
        crossings.sort_values(by='Timestamp',inplace=True)
        crossings['time'] = pd.to_datetime(crossings['Timestamp'])
        crossings.set_index('time',inplace=True)
        crossings.index = crossings.index.tz_localize(None)
        crossings.drop(columns='Timestamp',inplace=True)

        # Group crossings within a minute of another into one cluster
        # Keep first and last time of a crossing

        def _filter_group(probe, g):
            dt = g.index.to_series().diff().dt.total_seconds()
            cluster_id = (dt >= threshold_seconds).cumsum()
            out = g.groupby(cluster_id, group_keys=False).apply(
                lambda x: x.iloc[[0, -1]] if len(x) > 1 else x
            )
            out['Probe'] = probe
            return out

        out = []
        for probe, g in crossings.sort_index().groupby('Probe', sort=False):
            out.append(_filter_group(probe, g))
        boundaries = pd.concat(out).sort_index().copy()

        return boundaries

# %% Determine_direction

def themis_region_intervals(spacecraft, themis_dir, region='msh', data_pop='with_plasma', resolution='5min'):

    """
    Resolution is the data point before and after the boundary used to determine the conditinos
    Data population is the type of data used to establish the direction of themis
    """

    boundaries = obtain_mp_boundaries(themis_dir, resolution=resolution)

    sc_boundaries = boundaries[boundaries['Probe']==spacecraft]

    sc_dir = PROC_THEMIS_DIRECTORIES[spacecraft]
    if data_pop=='field_only':
        sc_dir = os.path.join(sc_dir, 'FGM')
    else:
        sc_dir = os.path.join(sc_dir, 'msh')

    sc_dir = os.path.join(sc_dir, resolution)
    df_msh = import_processed_data(sc_dir)

    directions = {}

    for time in sc_boundaries.index:
        lower = time.floor(resolution) - pd.Timedelta(resolution)
        upper = time.ceil(resolution)

        if lower not in df_msh.index or upper not in df_msh.index:
            continue

        # inc: parameter increases going into magnetosphere
        # dec: parameter decreases going into magnetosphere
        if data_pop=='field_only':
            bound_params = {'B_avg': 'inc'}
        else:
            bound_params = {'B_avg': 'inc', 'N_tot': 'dec'}

        bound_directions = []

        for boundary, change in bound_params.items():

            before_val = df_msh.at[lower,boundary]
            after_val  = df_msh.at[upper,boundary]
            if np.isnan(before_val) or np.isnan(after_val):
                continue

            before_unc = df_msh.at[lower,f'{boundary}_unc']
            if np.isnan(before_unc):
                before_unc = 0

            after_unc = df_msh.at[upper,f'{boundary}_unc']
            if np.isnan(after_unc):
                after_unc = 0

            diff_avg = ufloat(after_val,after_unc) - ufloat(before_val,before_unc)

            # Bz increases going from MSH to MS
            # So diff > 0 means going into magnetosphere (for magnetic field, plasma decreases)
            if change=='dec':
                diff_avg *= -1

            if diff_avg.s > np.abs(diff_avg.n): # Uncertainty of difference greater than it itself, so not certain
                direction = 0
            else:
                direction = np.sign(diff_avg.n)
                # Inbound = 1, Outbound = -1

            bound_directions.append(direction)

        if set(bound_directions)=={1}:
            flag = 1
        elif set(bound_directions)=={-1}:
            flag = -1
        else:
            flag = 0

        directions[time] = flag

    sc_df = pd.DataFrame(list(directions.items()), columns=['time','direction'])
    sc_df.set_index('time',inplace=True)

    starts = sc_df.index[(sc_df['direction'] == -1) & (sc_df['direction'].shift(1) != -1)]  # previous direction != -1
    ends   = sc_df.index[(sc_df['direction'] == -1) & (sc_df['direction'].shift(-1) != -1)] # next direction != -1
    intervals = list(zip(starts, ends))

    return intervals
