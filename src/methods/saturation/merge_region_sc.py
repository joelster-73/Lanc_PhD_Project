# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:21:25 2025

@author: richarj2
"""

import numpy as np
import pandas as pd

from scipy.constants import physical_constants
m_a = physical_constants['alpha particle mass'][0]

from .sc_delay_time import calc_flat_delay

from ...config import get_proc_directory, CLUSTER_SPACECRAFT as cluster, THEMIS_SPACECRAFT as themis, MMS_SPACECRAFT as mms

from ...processing.mms.analysis import mms_region_intervals
from ...processing.cluster.analysis import cluster_region_intervals
from ...processing.themis.analysis import themis_region_intervals

from ...processing.reading import import_processed_spacecraft, import_updated_omni
from ...processing.dataframes import merge_dataframes
from ...processing.writing import write_to_cdf

from ...coordinates.boundaries import calc_msh_r_diff, vector_component_surface
from ...analysing.comparing import difference_series

column_names = {
    'r_x_name'  : 'r_x_GSE',
    'r_y_name'  : 'r_y_GSE',
    'r_z_name'  : 'r_z_GSE',
    'r_name'    : 'r_mag',
    'r_ax_name' : 'r_x_aGSE',
    'r_ay_name' : 'r_y_aGSE',
    'r_az_name' : 'r_z_aGSE',
    'v_x_name'  : 'V_x_GSE',
    'v_y_name'  : 'V_y_GSE',
    'v_z_name'  : 'V_z_GSE',
    'p_name'    : 'P_flow',
    'bz_name'   : 'B_z_GSM'
}

# MP & BS
pos_cols  = ['r_MP','r_BS','r_phi','r_F']
norm_vecs = {'field': ('B',), 'plasma': ('B','E','V','S')}

all_spacecraft = ('c1','mms1','tha','thb','thc','thd','the')

spacecraft     = {'sw': {'field': all_spacecraft, 'plasma': ('c1','mms1','thb')},
                  'msh': {'field': all_spacecraft, 'plasma': ('c1','mms1','the')}}


# %%% Merge_sc

def data_populations(sc, data, region):

    populations = ['state']

    if sc in cluster:

        if data in ('field','plasma'):
            populations.append('fgm')
        if data in ('plasma',):
            populations.append(region)

    elif sc in themis:

        populations[0] = 'STATE'

        if data in ('field','plasma'):
            populations.append('FGM')
        if data in ('plasma',):
            populations.append(region)

    elif sc in mms:

        if data in ('field','plasma'):
            populations.append('fgm')
        if data in ('plasma',):
            populations.append('fpi')

    return populations

def merge_sc_in_region(region, data_pop='plasma', sample_interval='5min', sc_keys=None, nose=False):

    """
    data_pop = 'field' means field only
    data_pop = 'plasma' means including plasma (so also field)
    """
    print(f'Processing {sample_interval} {region} data.')

    DIR = get_proc_directory(region, dtype=data_pop, resolution=sample_interval, create=True) # output directory

    dfs_combined = []
    if sc_keys is None:
        sc_keys = spacecraft[region][data_pop]

    ###----------IMPORTS----------###
    print('Importing OMNI.\n')
    df_omni = import_updated_omni(resolution=sample_interval)

    print_data(df_omni, 'OMNI')

    for sc in sc_keys:

        print(f'Importing -{sc}-.')

        populations = data_populations(sc, data_pop, region)

        df_sc = import_processed_spacecraft(sc, populations, sample_interval)

        ###----------FILTERING----------###

        print('Initial')
        print_data(df_sc, sc)

        if sc in cluster:
            intervals = cluster_region_intervals(sc, region)

        elif sc in themis: # requires themis data to determine crossings
            intervals = themis_region_intervals(sc, region, data_pop, sample_interval, df_sc=df_sc)

        elif sc in mms:
            intervals = mms_region_intervals(region)

        df_merged, df_merged_attrs = filter_spacecraft_region(df_sc, df_omni, sc, intervals, region, nose=nose)
        if df_merged.empty:
            print(f'No {sc} data in {region}.')
            continue

        suffix = f'_{sc}'
        df_merged.rename(columns={col: f'{col}{suffix}' for col in pos_cols}, inplace=True) # adds _sc suffix

        ###----------EXTRA PARAMETERS----------###

        print_median_params(df_merged, sc)

        # Components parallel/perp to bs/mp surface
        vector_component_surface(df_merged, sc, region, data_pop, surface_params=column_names)

        # Correction to time lag based on spacecraft position in solar wind
        calc_flat_delay(df_merged, region=region, pos_col=f'r_x_GSE_{sc}', pres_col=f'P_flow_{sc}', vel_col=f'V_x_GSE_{sc}', lag_col=f'prop_time_s_{sc}')

        update_parameters(df_merged, sc, region)

        ###----------WRITING----------###

        # Add to combined
        df_merged.dropna(how='all',inplace=True)
        df_merged.drop(columns=[col for col in df_merged if (col.endswith('_sw') or col.endswith('_pc'))], inplace=True)
        dfs_combined.append(df_merged)

        # Need suffix when combining but removing for individual file
        df_merged = df_merged.rename(columns={col: col[:-len(suffix)] for col in df_merged.columns if col.endswith(suffix)})

        print('Complete mask')
        print_data(df_merged, sc)

        # Writes individual to file with omni
        update_attributes(df_merged, df_merged_attrs, region, suffix)

        print(f'Writing {sc} to file...')
        write_to_cdf(df_merged, directory=DIR, file_name=f'{region}_times_{sc}', reset_index=True)

    ###----------COMBINING----------###
    print('Combining spacecraft')
    df_wide = pd.concat(dfs_combined, axis=1)

    mask = pd.DataFrame({sc: df_wide[f'B_avg_{sc}'].notna() for sc in sc_keys})
    first_valid = mask.idxmax(axis=1)

    result = []
    for sc in sc_keys:
        suffix = f'_{sc}'
        sc_cols = [col for col in df_wide.columns if col.endswith(suffix)]
        renamed = {col: col[:-len(suffix)] for col in sc_cols}
        subset = (df_wide.loc[first_valid == sc, sc_cols].rename(columns=renamed))
        subset[f'sc_{region}'] = sc
        result.append(subset)

    df_combined = pd.concat(result).sort_index()
    df_combined.index.name = 'epoch'
    update_attributes(df_combined, df_merged_attrs, region, '')

    print('Merging')
    df_combined.attrs['units'][f'sc_{region}'] = 'STRING'

    df_combined.drop(columns=['no_np_ratio', 'no_np_ratio_unc', 'no_np_ratio_count', 'nhe_np_ratio', 'nhe_np_ratio_unc', 'nhe_np_ratio_count'], inplace=True) # columns don't care about when combined

    # Write
    print('Writing combined to file...')
    write_to_cdf(df_combined, directory=DIR, file_name=f'{region}_times_combined', reset_index=True)


def print_data(data, name):
    print(f'Amount of {name}:\n|B|: {data["B_avg"].count():,}\n|V|: {data["V_flow"].count():,}\n')

def print_median_params(df, sc):

    print('omni')
    for (param,col) in (('pressure','P_flow_sw'),('density','N_tot_sw'),('velocity','V_x_GSE_sw')):
        vals = df[col].to_numpy()

        print(f'mean {param}:   {np.mean(vals[~np.isnan(vals)]):.3g}')
        print(f'median {param}: {np.median(vals[~np.isnan(vals)]):.3g}')

    print(sc)

    for (param,col) in (('pressure',f'P_flow_{sc}'),('density',f'N_tot_{sc}'),('velocity',f'V_x_GSE_{sc}')):
        vals = df[col].to_numpy()

        print(f'mean {param}:   {np.mean(vals[~np.isnan(vals)]):.3g}')
        print(f'median {param}: {np.median(vals[~np.isnan(vals)]):.3g}')

def filter_spacecraft_region(df, omni, sc, intervals, region, nose=False):

    ###-----Location-----###
    spacecraft_distance(df)

    location_mask = (df['r_x_GSE']>0)
    if nose:
        location_mask &= (df['r_z_GSE'].abs()<5)
    location_mask &= (df['r_mag'] < 35) # exclude ARTEMIS

    df = df.loc[location_mask]

    print('Upstream of Earth')
    print_data(df, sc)

    # Combine OMNI and spacecraft

    df_merged_attrs = df.attrs # stores
    df_merged = merge_dataframes(omni, df, suffices=('sw',sc))

    ###-----Region-----###

    df_positions = calc_msh_r_diff(df_merged, 'BOTH', position_key=sc, data_key='sw', column_names=column_names)
    df_merged    = pd.concat([df_merged, df_positions[pos_cols]], axis=1)

    mask = np.zeros(len(df_merged),dtype=bool)
    interval_index = pd.IntervalIndex.from_tuples(intervals, closed='both')

    # Times when have crossings
    inside_interval  = interval_index.get_indexer(df_merged.index) != -1
    outside_interval = ~inside_interval

    if region=='msh':
        condition       = (df_merged['r_F'] > 0.1) & (df_merged['r_F'] < 1)
        loose_condition = (df_merged['r_F'] > 0.05) & (df_merged['r_F'] < 1.2)

    elif region=='sw':
        condition       = (df_merged['r_F'] > 1.5) # position further than empirical BS
        loose_condition = (df_merged['r_F'] > 1.0 )  # looser than the 1.5 used outside intervals

    mask |= outside_interval & condition
    mask |= inside_interval  & loose_condition # prevent crossings not being concurrent contaminating the dataset

    df_merged = df_merged.loc[mask]

    return df_merged, df_merged_attrs



# %% updates

def spacecraft_distance(df):
    if 'r_mag' not in df:
        cols     = [f'r_{comp}_GSE' for comp in ('x','y','z')]
        r        = (df[cols]**2).sum(axis=1)**0.5
        try:
            unc_cols = [f'r_{comp}_GSE_unc' for comp in ('x','y','z')]
            sigma_r = np.sqrt(((df[cols].to_numpy() / r[:, None])**2 * df[unc_cols].to_numpy()**2).sum(axis=1))
        except:
            sigma_r = np.nan

        df.insert(0, 'r_mag', r)
        df.insert(1, 'r_mag_unc', sigma_r)

    if 'r_mag_count' in df:
        df.drop(columns=['r_mag_count'],inplace=True)

def update_parameters(df, sc, region):
    """
    Removes erroneous values
    """

    suffix = f'_{sc}'

    for temp in (f'T_tot{suffix}',f'T_tot_unc{suffix}'):
        df[temp] /= 1e6 #convert to MK

    if region=='sw':
        remove_extremes(df, {f'beta{suffix}': 100, f'P_flow{suffix}': 30, f'E_mag{suffix}': 20, f'E_y_GSM{suffix}': 20})

    elif region=='msh':
        remove_extremes(df, {f'B_avg{suffix}': 100}, {f'B_z_GSM{suffix}': 250, f'beta{suffix}': 100})

        # Rotation of clock angle, ignoring sw uncertainty for time being
        clock_change = difference_series(df['B_clock_sw'],df[f'B_clock{suffix}'],unit='rad').abs()
        idx = df.columns.get_loc('B_clock_sw')
        df.insert(idx+2, f'Delta B_theta{suffix}', clock_change)

        not_nan = ~df[f'Delta B_theta{suffix}'].isna()
        if f'B_clock_unc{suffix}' in df: # Not implemented for Cluster currently
            df.loc[not_nan,f'Delta B_theta_unc{suffix}'] = df.loc[not_nan,f'B_clock_unc{suffix}']

        # Reversal of Bz; not interested in error
        bz_change = df[f'B_z_GSM{suffix}']/df['B_z_GSM_sw'] - 1
        bz_change[~np.isfinite(bz_change)] = np.nan

        idx = df.columns.get_loc('B_z_GSM_sw')
        df.insert(idx+2, f'Delta B_z{suffix}', bz_change)

        remove_extremes(df, {f'Delta B_z{suffix}': 1000})

def remove_extremes(df, mapping={}, abs_mapping={}):

    for col, limit in mapping.items():
        df.loc[df[col]>limit,col] = np.nan

    for col, limit in abs_mapping.items():
        df.loc[df[col].abs()>limit,col] = np.nan


def update_attributes(df, attrs_dict, region='sw', suffix=''):

    df.attrs = attrs_dict
    for temp in (f'T_tot{suffix}',f'T_tot_unc{suffix}'):
        df.attrs['units'][temp] = 'MK'

    df.attrs['units']['prop_time_s'] = 's'
    if region=='msh':
        df.attrs['units']['Delta B_theta'] = 'rad'
        df.attrs['units']['Delta B_z'] = '1'
