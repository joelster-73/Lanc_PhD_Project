# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:21:25 2025

@author: richarj2
"""
import os

import re
import numpy as np
import pandas as pd
from src.config import CROSSINGS_DIR, PROC_CLUS_DIR_SPIN, PROC_CLUS_DIR_5VPS, SW_DIR, OMNI_DIR, MMS_DIR, THEMIS_DIR

from src.processing.mms.handling import mms_region_intervals
from src.processing.cluster.handling import cluster_region_intervals
from src.processing.themis.config import PROC_THEMIS_DIRECTORIES
from src.processing.themis.analysis import themis_region_intervals
from src.processing.reading import import_processed_data
from src.processing.dataframes import merge_dataframes
from src.processing.writing import write_to_cdf

from src.coordinates.boundaries import calc_msh_r_diff, calc_normal_for_sc
from src.coordinates.magnetic import GSE_to_GSM_with_angles
from src.analysing.comparing import difference_series

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
cluster_directories = {'field_only': PROC_CLUS_DIR_5VPS, 'with_plasma': PROC_CLUS_DIR_SPIN}

# MP & BS
pos_cols  = ['r_MP','r_BS','r_phi','r_F']
norm_vecs = {'field_only': ('B',), 'with_plasma': ('B','E','V','S')}
surfaces  = {'sw': 'BS', 'msh': 'MP'}

all_spacecraft = ('c1','m1','tha','thb','thc','thd','the')
cluster        = ('c1','c2','c3','c4')
themis         = ('tha','thb','thc','thd','the')
mms            = ('m1','m2','m3','m4')

spacecraft     = {'sw': {'field_only': all_spacecraft, 'msh': ('c1','m1')},
                  'msh': {'field_only': all_spacecraft, 'with_plasma': ('c1','m1','the')}}
# consider adding/removing some themis spacecraft, e.g. thd for sw with plasma

param_map_pc = {k: k.replace('_sw', '_pc') for k in ['AE_sw','AL_sw','AU_sw', 'AE_17m_sw','PCN_sw','PCN_17m_sw','SYM_D_sw','SYM_H_sw','ASY_D_sw','ASY_H_sw','PSI_P_10_sw','PSI_P_30_sw','PSI_P_60_sw']}

# %%%
def merge_sc_in_region(region, data_pop='with_plasma', sample_interval='5min', sc_keys=None):

    # Solar wind data
    df_sw    = import_processed_data(os.path.join(OMNI_DIR, 'with_lag'), f'omni_{sample_interval}.cdf')

    dfs_combined = []

    field_col = 'B_avg'

    for sc in sc_keys:

        if sc in mms:
            intervals = mms_region_intervals(MMS_DIR, 'msh')

            field_dir  = os.path.join(MMS_DIR, 'field', sample_interval)
            df_field  = import_processed_data(field_dir)

            if data_pop=='field_only':
                df_sc = df_field
            else:
                plasma_dir = os.path.join(MMS_DIR, 'plasma', sample_interval)
                df_plasma = import_processed_data(plasma_dir)
                df_sc = merge_dataframes(df_field, df_plasma)

        elif sc in cluster:
            intervals = cluster_region_intervals(CROSSINGS_DIR, region)

            sc_dir = cluster_directories[data_pop]

            if data_pop=='with_plasma':
                sc_dir = os.path.join(sc_dir, region)

            sc_dir = os.path.join(sc_dir, sample_interval)
            df_sc = import_processed_data(sc_dir)


        elif sc in themis:
            intervals = themis_region_intervals(sc, THEMIS_DIR, region, data_pop)
            sc_dir = PROC_THEMIS_DIRECTORIES[sc]
            if data_pop=='field_only':
                sc_dir = os.path.join(sc_dir, 'FGM')
            else:
                sc_dir = os.path.join(sc_dir, 'msh')

            sc_dir = os.path.join(sc_dir, sample_interval)
            df_sc = import_processed_data(sc_dir)

        else:
            raise Exception(f'"{sc}" not implemented.')


        ###---------------FILTERING------------###
        rename_map = {
            col: cluster_column_map[key] + col[len(key):]
            for col in df_sc.columns
            for key in cluster_column_map
            if col.startswith(key)
        }
        df_sc.rename(columns=rename_map, inplace=True)

        # Update attrs['units'] if it exists
        if 'units' in df_sc.attrs and isinstance(df_sc.attrs['units'], dict):
            df_sc.attrs['units'] = {
                rename_map.get(col, col): unit
                for col, unit in df_sc.attrs['units'].items()
            }

        location_mask   = (df_sc['r_x_GSE']>0)
        #location_mask &= (np.abs(df_sc['r_z_GSE'])<5)
        location_mask  &= ~np.isnan(df_sc[field_col])

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

        if 'r_mag_count_sc' in df_sc:
            df_sc.drop(columns=['r_mag_count_sc'],inplace=True)

        # Combine
        df_merged = merge_dataframes(df_sw, df_sc, suffix_1='sw', suffix_2=sc)
        df_merged.columns = [param_map_pc.get(col,col) for col in df_merged.columns] # changes sw to pc for some omni
        df_merged.attrs['units'] = {param_map_pc.get(col,col): df_merged.attrs['units'].get(col,col) for col in df_merged.attrs['units']}
        df_merged_attrs = df_merged.attrs # stores

        ###----------FILTER FOR REGION----------###

        # Filter for MSH
        df_positions = calc_msh_r_diff(df_merged, 'BOTH', position_key=sc, data_key='sw', column_names=column_names)
        df_merged = pd.concat([df_merged,df_positions[pos_cols]],axis=1)

        mask = np.zeros(len(df_merged),dtype=bool)
        interval_index = pd.IntervalIndex.from_tuples(intervals, closed='both')

        if sc in cluster:
            mask |= (interval_index.get_indexer(df_merged.index) != -1)

        elif sc in mms:
            mask |= ((interval_index.get_indexer(df_merged.index) != -1) & (df_merged['r_F']>0))

        elif sc in themis:
            mask |= ((interval_index.get_indexer(df_merged.index) != -1) & (df_merged['r_F']<1))

        mask |= (df_merged['r_F']>0) & (df_merged['r_F']<1)

        df_merged = df_merged.loc[mask]
        df_merged.rename(columns={col: f'{col}_{sc}' for col in pos_cols}, inplace=True) # adds _sc suffix

        ###----------NORMAL TO SURFACE----------###

        surface = surfaces[region]
        couple_vecs = norm_vecs[data_pop]

        normals = calc_normal_for_sc(df_merged, surface, position_key=sc, data_key='sw', column_names=column_names)
        normals_gsm = GSE_to_GSM_with_angles(normals, (list(normals.columns),), df_coords=df_merged, coords_suffix='sw')
        df_merged = pd.concat([df_merged,normals_gsm],axis=1)

        norm_cols = [f'N{comp}_GSM_{surface}' for comp in ('x','y','z')]
        N = df_merged[norm_cols].to_numpy()
        for vec in couple_vecs:

            A_cols = [f'{vec}_{comp}_GSM_{sc}' for comp in ('x','y','z')]
            if A_cols[0] not in df_merged:
                A_cols[0] = A_cols[0].replace('GSM','GSE')

            A = df_merged[A_cols].to_numpy()
            A_dot_N   = np.einsum('ij,ij->i', A, N)
            A_norm_sq = np.einsum('ij,ij->i', A, A)

            with np.errstate(divide='ignore', invalid='ignore'):
                tangential_mag = np.sqrt(A_norm_sq - (A_dot_N ** 2))
                tangential_mag = np.nan_to_num(tangential_mag)  # Replace NaNs with 0

            df_merged[f'{vec}_perp_{sc}'] = A_dot_N
            df_merged[f'{vec}_parallel_{sc}'] = tangential_mag

        df_merged.rename(columns={col: f'{col}_{sc}' for col in norm_cols}, inplace=True) # adds _sc suffix

        ###----------EXTRA PARAMETERS----------###
        if region=='sw':
            print('Change erroneous boundaries')
            # df_merged.loc[df_merged[f'B_avg_{sc}']>100,f'B_avg_{sc}'] = np.nan
            # df_merged.loc[df_merged[f'B_z_GSM_{sc}'].abs()>250,f'B_z_GSM_{sc}'] = np.nan
            # df_merged.loc[df_merged[f'beta_{sc}'].abs()>100,f'beta_{sc}'] = np.nan

        elif region=='msh':
            df_merged.loc[df_merged[f'B_avg_{sc}']>100,f'B_avg_{sc}'] = np.nan
            df_merged.loc[df_merged[f'B_z_GSM_{sc}'].abs()>250,f'B_z_GSM_{sc}'] = np.nan
            df_merged.loc[df_merged[f'beta_{sc}'].abs()>100,f'beta_{sc}'] = np.nan

            # Rotation of clock angle
            # Ignoring sw uncertainty for time being
            df_merged[f'Delta B_theta_{sc}'] = np.abs(difference_series(df_merged['B_clock_sw'],df_merged[f'B_clock_{sc}'],unit='rad'))
            not_nan = ~np.isnan(df_merged[f'Delta B_theta_{sc}'])
            if f'B_clock_unc_{sc}' in df_merged: # Not implemented for Cluster currently
                df_merged.loc[not_nan,f'Delta B_theta_unc_{sc}'] = df_merged.loc[not_nan,f'B_clock_unc_{sc}']

            # Reversal of Bz
            # Not interested in error
            df_merged[f'Delta B_z_{sc}'] = df_merged[f'B_z_GSM_{sc}']/df_merged['B_z_GSM_sw'] - 1
            df_merged[f'Delta B_z_{sc}'] = df_merged[f'Delta B_z_{sc}'].replace([np.inf, -np.inf], np.nan)
            df_merged.loc[np.abs(df_merged[f'Delta B_z_{sc}'])>1000,f'Delta B_z_{sc}'] = np.nan

        # Add to combined
        df_merged.dropna(how='all',inplace=True)
        dfs_combined.append(df_merged[[col for col in df_merged if f'_{sc}' in col]])

        # Writes individual to file with omni
        df_merged.attrs = df_merged_attrs
        df_merged.attrs['units'][f'Delta B_theta_{sc}'] = 'rad'
        df_merged.attrs['units'][f'Delta B_z_{sc}'] = '1'

        output_file = os.path.join(SW_DIR, data_pop, sample_interval, f'{region}_times_{sc}.cdf')
        print(f'Writing {sc} to file...')
        write_to_cdf(df_merged, output_file, reset_index=True)


    # Combining
    print('Combining spacecraft')
    df_wide = pd.concat(dfs_combined, axis=1)
    rows = []

    print('Selecting spacecraft')
    for idx, row in df_wide.iterrows():
        chosen_sc = None
        sc_data = {}

        for sc in sc_keys:
            key_col = f'B_avg_{sc}'
            if pd.notna(row.get(key_col, None)):
                chosen_sc = sc
                sc_data = {re.sub(f'_{sc}$', '_sc', col): row[col]
                           for col in df_wide.columns if col.endswith(f'_{sc}')}
                sc_data[f'sc_{region}'] = sc
                break

        if chosen_sc is not None:
            rows.append(pd.Series(sc_data, name=idx))

    df_combined = pd.DataFrame(rows)
    df_combined.index.name = 'epoch'
    df_combined.attrs = df_merged_attrs

    if region=='msh':
        df_combined.attrs['units']['Delta B_theta_msh'] = 'rad'
        df_combined.attrs['units']['Delta B_z_msh'] = '1'

    print('Merging')
    df_final = merge_dataframes(df_sw, df_combined, suffix_1='sw', suffix_2=None)
    df_final.columns = [param_map_pc.get(col,col) for col in df_final.columns]
    df_final.attrs['units'] = {param_map_pc.get(col,col): df_final.attrs['units'].get(col,col) for col in df_final.attrs['units']}
    df_final.attrs['units'][f'sc_{region}'] = 'STRING'

    # Write
    output_file = os.path.join(SW_DIR, data_pop, sample_interval, f'{region}_times_combined.cdf')
    print('Writing combined to file...')
    write_to_cdf(df_final, output_file, reset_index=True)



# %%