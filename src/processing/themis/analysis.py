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


    thresholds = {'1min': 120, '5min': 360}
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

        # Paper suggests only using crossings with >0.75 probability
        crossings = crossings.loc[crossings['Probability']>0.75]

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


def determine_mp_direction(mp_boundaries, df_sc, df_resolution, mp_params):

    directions = {}

    for time in mp_boundaries.index:
        lower = time.floor(df_resolution) - pd.Timedelta(df_resolution)
        upper = time.ceil(df_resolution)

        if lower not in df_sc.index or upper not in df_sc.index:
            continue

        bound_directions = []

        for boundary, change in mp_params.items():

            before_val = df_sc.at[lower,boundary]
            after_val  = df_sc.at[upper,boundary]
            if np.isnan(before_val) or np.isnan(after_val):
                continue

            before_unc = df_sc.at[lower,f'{boundary}_unc']
            if np.isnan(before_unc):
                before_unc = 0

            after_unc = df_sc.at[upper,f'{boundary}_unc']
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

    return directions

def obtain_bs_boundaries(themis_dir, bs_file='Pallocchia_2024_BS_Crossings.txt'):

    if bs_file=='Pallocchia_2024_BS_Crossings.txt':

        sc_dict = {1: 'imp-8', 2: 'geotail', 3: 'magion', 4: 'c1', 5: 'thb', 6: 'thc'}

        file_path = os.path.join(themis_dir,bs_file)
        boundaries = pd.read_csv(file_path, skiprows=61, sep='\t')

        boundaries['time'] = pd.to_datetime(boundaries['YYYY'].astype(str) + boundaries['DOY'].astype(str) + boundaries['HH:MM:SS(UT)'],format='%Y%j%H:%M:%S')

        boundaries.set_index('time',inplace=True)
        boundaries['Probe'] = boundaries['S/C'].map(sc_dict)
        boundaries.drop(columns=['YYYY','DOY','HH:MM:SS(UT)','S/C'])

    return boundaries


def determine_bs_direction(bs_boundaries, df_sc, df_resolution, bs_params):

    directions = {}

    if 'r_mag' not in df_sc:
        cols           = [f'r_{comp}_GSE' for comp in ('x','y','z')]
        r        = np.linalg.norm(df_sc[cols].values, axis=1)
        unc_cols = [f'r_{comp}_GSE_unc' for comp in ('x','y','z')]
        try:
            sigma_r = np.sqrt(((df_sc[cols].values / r[:, None])**2 * df_sc[unc_cols].values**2).sum(axis=1))
        except:
            sigma_r = np.nan

        df_sc.insert(0, 'r_mag', r)
        df_sc.insert(1, 'r_mag_unc', sigma_r)

    for time in bs_boundaries.index:
        lower = time.floor(df_resolution) - pd.Timedelta(df_resolution)
        upper = time.ceil(df_resolution)

        if lower not in df_sc.index or upper not in df_sc.index:
            continue

        bound_directions = []

        for boundary, change in bs_params.items():

            before_val = df_sc.at[lower,boundary]
            after_val  = df_sc.at[upper,boundary]
            if np.isnan(before_val) or np.isnan(after_val):
                continue

            before_unc = df_sc.at[lower,f'{boundary}_unc']
            if np.isnan(before_unc):
                before_unc = 0

            after_unc = df_sc.at[upper,f'{boundary}_unc']
            if np.isnan(after_unc):
                after_unc = 0

            diff_avg = ufloat(after_val,after_unc) - ufloat(before_val,before_unc)

            # Bz decreases going from MSH to BS
            # So diff < 0 means going into bow shock (for magnetic field, plasma decreases)
            if change=='inc':
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

    return directions

# %% Determine_direction

def themis_region_intervals(spacecraft, themis_dir, region='msh', data_pop='with_plasma', resolution='5min', max_gap=None):

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
        sc_dir = os.path.join(sc_dir, region)

    sc_dir = os.path.join(sc_dir, resolution)
    df_sc  = import_processed_data(sc_dir)

    if region=='msh':
        boundaries = obtain_mp_boundaries(themis_dir, resolution=resolution)
        sc_boundaries = boundaries[boundaries['Probe']==spacecraft]

        # inc: parameter increases going into magnetosphere
        # dec: parameter decreases going into magnetosphere
        if data_pop=='field_only':
            bound_params = {'B_avg': 'inc'}
        else:
            bound_params = {'B_avg': 'inc', 'N_tot': 'dec', 'T_tot': 'inc'}

        directions = determine_mp_direction(sc_boundaries, df_sc, resolution, bound_params)

        if max_gap is None:
            max_gap = pd.Timedelta('15h')

    elif region=='sw':
        boundaries = obtain_bs_boundaries(themis_dir)
        sc_boundaries = boundaries[boundaries['Probe']==spacecraft]

        # inc: parameter increases going into solar wind
        # dec: parameter decreases going into solar wind
        if data_pop=='field_only':
            bound_params = {'B_avg': 'dec'}
        else:
            bound_params = {'B_avg': 'dec', 'N_tot': 'dec', 'r_mag': 'inc'}

        directions = determine_bs_direction(sc_boundaries, df_sc, resolution, bound_params)

        if max_gap is None:
            max_gap = pd.Timedelta('40h')

    # Want to be outbound from BS for sw or MP for MSH
    outbound = -1

    sc_df = pd.DataFrame(list(directions.items()), columns=['time','direction'])
    sc_df.set_index('time',inplace=True)
    sc_df.sort_index(inplace=True)

    # Want outbound times (i.e. crossing MP into MSH or BS into SW)
    starts = sc_df.index[(sc_df['direction'] == outbound) & (sc_df['direction'].shift(-1) == -outbound)]  # next direction is opposite
    valid = sc_df.index.get_indexer(starts) + 1
    valid = valid[valid < len(sc_df)]  # avoid IndexError
    ends = sc_df.index[valid]

    # align start and end
    intervals = pd.DataFrame({'start': starts.values, 'end': ends.values})
    intervals = intervals.dropna()
    intervals = intervals[(intervals['end'] - intervals['start']) <= max_gap]

    return list(intervals.itertuples(index=False, name=None))