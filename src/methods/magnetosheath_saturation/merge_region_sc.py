# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:21:25 2025

@author: richarj2
"""

import numpy as np
import pandas as pd

from scipy.constants import m_p, physical_constants
m_a = physical_constants['alpha particle mass'][0]
k_B = physical_constants['Boltzmann constant in eV/K'][0]

from .sc_delay_time import calc_bs_sc_delay

from ...config import get_proc_directory

from ...processing.mms.analysis import mms_region_intervals
from ...processing.cluster.analysis import cluster_region_intervals
from ...processing.themis.analysis import themis_region_intervals
from ...processing.omni.config import indices_columns

from ...processing.reading import import_processed_data
from ...processing.dataframes import merge_dataframes
from ...processing.writing import write_to_cdf

from ...coordinates.boundaries import calc_msh_r_diff, calc_normal_for_sc
from ...coordinates.magnetic import convert_GSE_to_GSM_with_angles, calc_GSE_to_GSM_angles
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

# Map to make consistent with other dataframes
cluster_column_map = {'V_mag': 'V_flow', 'T_thm': 'T_tot', 'N_ion': 'N_tot'}

# MP & BS
pos_cols  = ['r_MP','r_BS','r_phi','r_F']
norm_vecs = {'field': ('B',), 'plasma': ('B','E','V','S')}
surfaces  = {'sw': 'BS', 'msh': 'MP'}

all_spacecraft = ('c1','mms1','tha','thb','thc','thd','the')
cluster        = ('c1','c2','c3','c4')
themis         = ('tha','thb','thc','thd','the')
mms            = ('mms1','mms2','mms3','mms4')

spacecraft     = {'sw': {'field': all_spacecraft, 'plasma': ('c1','mms1','thb')},
                  'msh': {'field': all_spacecraft, 'plasma': ('c1','mms1','the')}}


# %%% Merge_OMNI_sc

def merge_sc_in_region(region, data_pop='plasma', sample_interval='5min', sc_keys=None, nose=False):

    """
    Data_pop = 'field' means field only
    data_pop = 'plasma' means including plasma (so also field)
    """

    DIR = get_proc_directory(region, dtype=data_pop, resolution=sample_interval, create=True) # output directory

    dfs_combined = []
    field_col = 'B_avg'
    if sc_keys is None:
        sc_keys = spacecraft[region][data_pop]

    ###----------IMPORTS----------###
    print('Importing OMNI.\n')
    df_omni = import_processed_data('omni', resolution=sample_interval)
    update_omni(df_omni, indices_columns)

    for sc in sc_keys:

        print(f'Importing {sc}')
        if sc in mms:

            df_field  = import_processed_data(sc, dtype='fgm', resolution=sample_interval)

            if data_pop=='field':
                df_sc = df_field
            else:
                df_plasma = import_processed_data(sc, dtype='fpi', resolution=sample_interval)
                df_sc = merge_dataframes(df_field, df_plasma)

            intervals = mms_region_intervals('msh')

        elif sc in themis:

            continue ##################################

            df_sc = import_processed_data(sc, dtype=region, resolution=sample_interval)
            intervals = themis_region_intervals(sc, region, data_pop, sample_interval)

        elif sc in cluster:

            continue ##################################

            df_sc = import_processed_data(sc, dtype=region, resolution=sample_interval)

            rename_map = {col: cluster_column_map[key] + col[len(key):] for col in df_sc.columns for key in cluster_column_map if col.startswith(key)}
            df_sc.rename(columns=rename_map, inplace=True)
            if 'units' in df_sc.attrs and isinstance(df_sc.attrs['units'], dict):
                df_sc.attrs['units'] = {rename_map.get(col, col): unit for col, unit in df_sc.attrs['units'].items()}

            intervals = cluster_region_intervals(sc, region)

        else:
            raise Exception(f'"{sc}" not implemented.')


        for column in ('quality_esa','flag','quality_fgm'):
            # Erroneous columns left in thb before resampling
            if column in df_sc:
                df_sc.drop(columns=[column],inplace=True)

        ###---------------FILTERING------------###

        location_mask = (df_sc['r_x_GSE']>0)
        if nose:
            location_mask &= (df_sc['r_z_GSE'].abs()<5)
        location_mask &= ~np.isnan(df_sc[field_col])

        df_sc = df_sc.loc[location_mask]

        if 'r_mag' not in df_sc:
            cols     = [f'r_{comp}_GSE' for comp in ('x','y','z')]
            r        = np.linalg.norm(df_sc[cols].values, axis=1)
            unc_cols = [f'r_{comp}_GSE_unc' for comp in ('x','y','z')]
            try:
                sigma_r = np.sqrt(((df_sc[cols].values / r[:, None])**2 * df_sc[unc_cols].values**2).sum(axis=1))
            except:
                sigma_r = np.nan

            df_sc.insert(0, 'r_mag', r)
            df_sc.insert(1, 'r_mag_unc', sigma_r)

        if 'r_mag_count' in df_sc:
            df_sc.drop(columns=['r_mag_count'],inplace=True)

        # Combine OMNI and spacecraft

        df_merged_attrs = df_sc.attrs # stores
        df_merged = merge_dataframes(df_omni, df_sc, suffix_1='sw', suffix_2=sc)
        suffix = f'_{sc}'

        ###----------FILTER FOR REGION----------###

        df_positions = calc_msh_r_diff(df_merged, 'BOTH', position_key=sc, data_key='sw', column_names=column_names)
        df_merged = pd.concat([df_merged, df_positions[pos_cols]], axis=1)

        mask = np.zeros(len(df_merged),dtype=bool)
        interval_index = pd.IntervalIndex.from_tuples(intervals, closed='both')

        if region=='msh':

            if data_pop=='plasma':
                condition = df_merged[f'T_tot{suffix}'] >= (1e3 / k_B) # ion temp > 1 keV
            else:
                condition = (df_merged['r_F']>0.1) & (df_merged['r_F']<1) # position between empirial MP and BS

            if sc in themis:
                mask |= ((interval_index.get_indexer(df_merged.index) != -1) & condition)
                mask |= condition

            elif sc in cluster: # MSH interval
                mask |= (interval_index.get_indexer(df_merged.index) != -1)
                mask |= condition

            elif sc in mms: # Within BS (BS crossings)
                mask |= ((interval_index.get_indexer(df_merged.index) != -1) & condition)
                mask |= condition

        elif region=='sw':

            if data_pop=='plasma':
                condition = df_merged[f'T_tot{suffix}'] < (1e3 / k_B) # ion temp < 1 keV
            else:
                condition = df_merged['r_F'] < 1 # position further than empirial BS

            # Second condition is to prevent incorrect boundaries
            mask |= ((interval_index.get_indexer(df_merged.index) != -1) & (condition))
            mask |= condition

            mask &= (df_merged[f'r_mag{suffix}']<35) # To exclude ARTEMIS

        df_merged = df_merged.loc[mask]
        df_merged.rename(columns={col: f'{col}{suffix}' for col in pos_cols}, inplace=True) # adds _sc suffix

        if df_merged.empty:
            print(f'No {sc} data in {region}.')
            continue

        ###----------NORMAL TO SURFACE----------###

        surface = surfaces[region]
        couple_vecs = norm_vecs[data_pop]

        if surface=='MP': # BS not currently implemented
            normals = calc_normal_for_sc(df_merged, surface, position_key=sc, data_key='sw', column_names=column_names)

            normals_gsm = convert_GSE_to_GSM_with_angles(normals, (list(normals.columns),), df_coords=df_merged, coords_suffix='sw')
            df_merged = pd.concat([df_merged,normals_gsm],axis=1)

            norm_cols = [f'N{comp}_GSM_{surface}' for comp in ('x','y','z')]
            N = df_merged[norm_cols].to_numpy()
            for vec in couple_vecs:

                A_cols = [f'{vec}_{comp}_GSM{suffix}' for comp in ('x','y','z')]
                if A_cols[0] not in df_merged:
                    A_cols[0] = A_cols[0].replace('GSM','GSE')

                A = df_merged[A_cols].to_numpy()
                A_dot_N   = np.einsum('ij,ij->i', A, N)
                A_norm_sq = np.einsum('ij,ij->i', A, A)

                with np.errstate(divide='ignore', invalid='ignore'):
                    tangential_mag = np.sqrt(A_norm_sq - (A_dot_N ** 2))
                    tangential_mag = np.nan_to_num(tangential_mag)  # Replace NaNs with 0

                df_merged[f'{vec}_perp{suffix}']     = A_dot_N
                df_merged[f'{vec}_parallel{suffix}'] = tangential_mag

            df_merged.rename(columns={col: f'{col}{suffix}' for col in norm_cols}, inplace=True) # adds _sc suffix

        ###----------EXTRA PARAMETERS----------###

        # Correction to time lag based on spacecraft position in solar wind
        calc_bs_sc_delay(df_merged, omni_key='sw', sc_key=sc, region=region)

        if region=='sw':
            remove_extremes(df_merged, {f'beta{suffix}': 1000, f'P_flow{suffix}': 30})

        elif region=='msh':
            remove_extremes(df_merged, {f'B_avg{suffix}': 100}, {f'B_z_GSM{suffix}': 250, f'beta{suffix}': 100})

            # Rotation of clock angle, ignoring sw uncertainty for time being
            df_merged[f'Delta B_theta{suffix}'] = np.abs(difference_series(df_merged['B_clock_sw'],df_merged[f'B_clock{suffix}'],unit='rad'))
            not_nan = ~np.isnan(df_merged[f'Delta B_theta{suffix}'])
            if f'B_clock_unc{suffix}' in df_merged: # Not implemented for Cluster currently
                df_merged.loc[not_nan,f'Delta B_theta_unc{suffix}'] = df_merged.loc[not_nan,f'B_clock_unc{suffix}']

            # Reversal of Bz; not interested in error
            df_merged[f'Delta B_z{suffix}'] = df_merged[f'B_z_GSM{suffix}']/df_merged['B_z_GSM_sw'] - 1
            df_merged[f'Delta B_z{suffix}'] = df_merged[f'Delta B_z{suffix}'].replace([np.inf, -np.inf], np.nan)

            remove_extremes(df_merged, {f'Delta B_z{suffix}': 1000})

        ###----------WRITING----------###

        # Add to combined
        df_merged.dropna(how='all',inplace=True)
        df_merged.drop(columns=[col for col in df_merged if (col.endswith('_sw') or col.endswith('_pc'))], inplace=True)
        dfs_combined.append(df_merged)


        # Need suffix when combining but removing for individual file
        df_merged = df_merged.rename(columns={col: col[:-len(suffix)] for col in df_merged.columns if col.endswith(suffix)})

        # Writes individual to file with omni
        df_merged.attrs = df_merged_attrs
        df_merged.attrs['units']['prop_time_s'] = 's'
        if region=='msh':
            df_merged.attrs['units']['Delta B_theta'] = 'rad'
            df_merged.attrs['units']['Delta B_z'] = '1'

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
    df_combined.attrs = df_merged_attrs

    df_combined.attrs['units']['prop_time_s'] = 's'
    if region=='msh':
        df_combined.attrs['units']['Delta B_theta'] = 'rad'
        df_combined.attrs['units']['Delta B_z'] = '1'

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

def remove_extremes(df, mapping={}, abs_mapping={}):

    for col, limit in mapping.items():
        df.loc[df[col]>limit,col] = np.nan

    for col, limit in abs_mapping.items():
        df.loc[df[col].abs()>limit,col] = np.nan