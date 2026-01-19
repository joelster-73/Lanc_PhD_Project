# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:21:25 2025

@author: richarj2
"""

import numpy as np
import pandas as pd

from scipy.constants import m_p, physical_constants
m_a = physical_constants['alpha particle mass'][0]

from .sc_delay_time import calc_bs_sc_delay

from ...config import get_proc_directory, CLUSTER_SPACECRAFT as cluster, THEMIS_SPACECRAFT as themis, MMS_SPACECRAFT as mms

from ...processing.mms.analysis import mms_region_intervals
from ...processing.cluster.analysis import cluster_region_intervals
from ...processing.themis.analysis import themis_region_intervals
from ...processing.omni.config import indices_columns

from ...processing.reading import import_processed_data, import_processed_spacecraft
from ...processing.dataframes import merge_dataframes
from ...processing.writing import write_to_cdf

from ...coordinates.boundaries import calc_msh_r_diff, vector_component_surface
from ...coordinates.magnetic import calc_GSE_to_GSM_angles
from ...analysing.comparing import difference_series
from ...analysing.calculations import calc_angle_between_vecs

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


# %%% Merge_OMNI_sc

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
    df_omni = import_processed_data('omni', resolution=sample_interval)
    update_omni(df_omni, indices_columns)

    for sc in sc_keys:

        print(f'Importing -{sc}-.')

        populations = data_populations(sc, data_pop, region)

        df_sc = import_processed_spacecraft(sc, populations, sample_interval)

        if sc in cluster:
            intervals = cluster_region_intervals(sc, region)

        elif sc in themis: # requires themis data to determine crossings
            intervals = themis_region_intervals(sc, region, data_pop, sample_interval, df_sc=df_sc)

        elif sc in mms:
            intervals = mms_region_intervals(region)

        ###---------------FILTERING------------###

        location_mask = (df_sc['r_x_GSE']>0)
        if nose:
            location_mask &= (df_sc['r_z_GSE'].abs()<5)

        df_sc = df_sc.loc[location_mask]
        spacecraft_distance(df_sc)

        # Combine OMNI and spacecraft

        df_merged_attrs = df_sc.attrs # stores
        df_merged = merge_dataframes(df_omni, df_sc, suffices=('sw',sc))
        suffix = f'_{sc}'

        ###----------FILTER FOR REGION----------###

        df_positions = calc_msh_r_diff(df_merged, 'BOTH', position_key=sc, data_key='sw', column_names=column_names)
        df_merged    = pd.concat([df_merged, df_positions[pos_cols]], axis=1)

        mask = np.zeros(len(df_merged),dtype=bool)
        interval_index = pd.IntervalIndex.from_tuples(intervals, closed='both')

        # Times when have crossings
        inside_interval  = interval_index.get_indexer(df_merged.index) != -1
        outside_interval = ~inside_interval

        if region=='msh':
            condition = (df_merged['r_F'] > 0.1) & (df_merged['r_F'] < 1)

        elif region=='sw':
            condition = (df_merged['r_F'] > 1.5) # position further than empirial BS
            #condition &= (df_merged[f'r_mag{suffix}']<35) # exclude ARTEMIS

        mask |= outside_interval & condition
        mask |= inside_interval

        df_merged = df_merged.loc[mask]
        df_merged.rename(columns={col: f'{col}{suffix}' for col in pos_cols}, inplace=True) # adds _sc suffix

        if df_merged.empty:
            print(f'No {sc} data in {region}.')
            continue

        ###----------EXTRA PARAMETERS----------###

        vector_component_surface(df_merged, sc, region, data_pop, surface_params=column_names)

        # Correction to time lag based on spacecraft position in solar wind
        calc_bs_sc_delay(df_merged, omni_key='sw', sc_key=sc, region=region)

        update_parameters(df_merged, sc, region)

        ###----------WRITING----------###

        # Add to combined
        df_merged.dropna(how='all',inplace=True)
        df_merged.drop(columns=[col for col in df_merged if (col.endswith('_sw') or col.endswith('_pc'))], inplace=True)
        dfs_combined.append(df_merged)

        # Need suffix when combining but removing for individual file
        df_merged = df_merged.rename(columns={col: col[:-len(suffix)] for col in df_merged.columns if col.endswith(suffix)})

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

    # Write
    print('Writing combined to file...')
    write_to_cdf(df_combined, directory=DIR, file_name=f'{region}_times_combined', reset_index=True)



# %% updates

def update_omni(df, drop_cols=indices_columns):

    # Rename temperature for consistency, assuming isothermal
    df.rename(columns={'P_flow': 'P_p', 'T_p': 'T_tot'}, inplace=True)
    df.attrs['units']['P_p'] = 'nPa'

    df.loc[df['T_tot']>=9999999, 'T_tot'] = np.nan # replace fills
    df['T_tot'] /= 1e6 #convert to MK
    df.attrs['units']['T_tot'] = 'MK'

    # OMNI defines pressure as rhoV^2 for just the protons, so halving for consistency
    df['P_flow'] = 0.5 * (df['n_p']*m_p + (1+df['na_np_ratio'])*m_a) * df['V_flow']**2 * 1e21

    df.loc[df['M_A']>100,'M_A'] = np.nan

    df['N_tot'] = df['n_p'] * (1+df['na_np_ratio'])
    df.attrs['units']['N_tot'] = 'n/cc'

    # Theta Bn angle - quasi-perp/quasi-para
    df['theta_Bn'] = calc_angle_between_vecs(df, 'B_GSE', 'R_BSN')

    # restrict to be between 0 and 90 degrees
    df.loc[df['theta_Bn']>np.pi/2,'theta_Bn'] = np.pi - df.loc[df['theta_Bn']>np.pi/2,'theta_Bn']
    df.attrs['units']['theta_Bn'] = 'rad'

    df['gse_to_gsm_angle'] = calc_GSE_to_GSM_angles(df, ref='B')

    # Drop index columns
    df.drop(columns=drop_cols,inplace=True,errors='ignore')

def spacecraft_distance(df):
    if 'r_mag' not in df:
        cols     = [f'r_{comp}_GSE' for comp in ('x','y','z')]
        r        = (df[cols]**2).sum(axis=1)**0.5
        try:
            unc_cols = [f'r_{comp}_GSE_unc' for comp in ('x','y','z')]
            sigma_r = np.sqrt(((df[cols].values / r[:, None])**2 * df[unc_cols].values**2).sum(axis=1))
        except:
            sigma_r = np.nan

        df.insert(0, 'r_mag', r)
        df.insert(1, 'r_mag_unc', sigma_r)

    if 'r_mag_count' in df:
        df.drop(columns=['r_mag_count'],inplace=True)

def update_parameters(df, sc, region):

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
