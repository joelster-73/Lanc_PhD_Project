# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:21:25 2025

@author: richarj2
"""





### MOVE FUNCTIONS FROM OTHER MERGE FILE INTO THE HANDLING OF THOSE SPACECRAFT
### TO THEN USE CROSSING DATABASE(S) TO FILTER FOR WHEN THE SPACECRAFT IS IN A PARTICULAR REGION
### ALSO INCLUDE USING R_F WITH THE OMNI DATA
### AND ACCOUNT FOR TIME RESOLUTION
### THEN CLEAN UP OTHER FILE AND THEN IN THIS ONE CREATE A DATASET OF SOLAR WIND OBSERVATIONS FROM CLUSTER AND MMS
### IF DATA IS LACKING, THEN TRY INCLUDE ONE THEMIS SPACECRAFT, PROBABLY D
### MAY NEED TO FIND THEMIS BS CROSSINGS
### CONSIDER ADDING BS NORMAL VECTORS



import os
import glob
import re
import numpy as np
import pandas as pd
from uncertainties import ufloat

from spacepy import pycdf

from src.config import CROSSINGS_DIR, PROC_CLUS_DIR_SW, PROC_CLUS_DIR_5VPS, MSH_DIR, OMNI_DIR, MMS_DIR, THEMIS_DIR, PCN_DIR
from src.processing.themis.config import PROC_THEMIS_DIRECTORIES
from src.processing.reading import import_processed_data
from src.processing.dataframes import merge_dataframes
from src.processing.writing import write_to_cdf
from src.coordinates.boundaries import calc_msh_r_diff, calc_normal_for_sc
from src.coordinates.magnetic import calc_GSE_to_GSM_angles, GSE_to_GSM_with_angles
from src.analysing.calculations import average_of_averages, calc_angle_between_vecs
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

if False:
    # Field only
    data_pop = 'field_only'
    couple_vecs = ('B',)
    cluster_directory = PROC_CLUS_DIR_5VPS
    themis_directory  = PROC_THEMIS_DIRECTORIES
    mms_directory     = os.path.join(MMS_DIR,'field')
    sw_keys          = ('c1','m1','tha','thb','thc','thd','the')
else:
    # With Plasma
    data_pop = 'with_plasma'
    couple_vecs = ('B','E','V','S')
    cluster_directory  = PROC_CLUS_DIR_SW
    themis_directory   = PROC_THEMIS_DIRECTORIES
    mms_directory      = os.path.join(MMS_DIR,'field')
    mms_directory_hpca = os.path.join(MMS_DIR,'plasma')
    sw_keys           = ('c1','m1')


# Map to make consistent with other dataframes
cluster_column_map = {'V_mag': 'V_flow', 'T_thm': 'T_tot', 'N_ion': 'N_tot'}


sample_intervals = ('1min','5min')
pos_cols  = ['r_MP','r_BS','r_phi','r_F']

param_map_pc = {k: k.replace('_sw', '_pc') for k in ['AE_sw','AL_sw','AU_sw', 'AE_17m_sw','PCN_sw','PCN_17m_sw','SYM_D_sw','SYM_H_sw','ASY_D_sw','ASY_H_sw','PSI_P_10_sw','PSI_P_30_sw','PSI_P_60_sw']}



# %% Combined
sc_sw_intervals = {'c1': c1_intervals, 'm1': m1_intervals}

for sample_interval in sample_intervals:

    # Solar wind data
    df_sw    = import_processed_data(os.path.join(OMNI_DIR, 'with_lag'), f'omni_{sample_interval}.cdf')

    dfs_combined = []

    for sc in sw_keys:

        field_col = 'B_avg'
        if sc in ('m1',):
            field_dir  = os.path.join(mms_directory, sample_interval)
            df_field  = import_processed_data(field_dir)

            if data_pop=='field_only':
                df_sc = df_field
            else:
                plasma_dir = os.path.join(mms_directory_hpca, sample_interval)
                df_plasma = import_processed_data(plasma_dir)
                df_sc = merge_dataframes(df_field, df_plasma)

        else:
            if sc in ('c1',):
                sc_dir = cluster_directory
            elif sc in ('tha','thb','thc','thd','the'):
                sc_dir = themis_directory[sc]
                if data_pop=='field_only':
                    sc_dir = os.path.join(sc_dir, 'FGM')
                else:
                    sc_dir = os.path.join(sc_dir, 'sw')
            else:
                raise Exception(f'"{sc}" not implemented.')
            sc_dir = os.path.join(sc_dir, sample_interval)

            df_sc = import_processed_data(sc_dir)

        # Renaming
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

        ###----------FILTER FOR SOLAR WIND----------###

        # Filter for MSH
        df_positions = calc_msh_r_diff(df_merged, 'BOTH', position_key=sc, data_key='sw', column_names=column_names)
        df_merged = pd.concat([df_merged,df_positions[pos_cols]],axis=1)

        mask = np.zeros(len(df_merged),dtype=bool)


        sc_intervals = sc_sw_intervals.get(sc,None)
        if sc_intervals is not None:
            interval_index = pd.IntervalIndex.from_tuples(sc_intervals, closed='both')
            mask |= (interval_index.get_indexer(df_merged.index) != -1)
        mask |= (df_merged['r_F']>1)


        df_merged = df_merged.loc[mask]
        df_merged.rename(columns={col: f'{col}_{sc}' for col in pos_cols}, inplace=True) # adds _sc suffix


        ###----------EXTRA PARAMETERS----------###
        df_merged.loc[df_merged[f'B_avg_{sc}']>100,f'B_avg_{sc}'] = np.nan
        df_merged.loc[df_merged[f'B_z_GSM_{sc}'].abs()>250,f'B_z_GSM_{sc}'] = np.nan
        df_merged.loc[df_merged[f'beta_{sc}'].abs()>100,f'beta_{sc}'] = np.nan

        # Add to combined
        df_merged.dropna(how='all',inplace=True)
        dfs_combined.append(df_merged[[col for col in df_merged if f'_{sc}' in col]])

        # Writes individual to file with omni
        df_merged.attrs = df_merged_attrs

        output_file = os.path.join(MSH_DIR, data_pop, sample_interval, f'sw_times_{sc}.cdf')
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

        for sc in sw_keys:
            key_col = f'B_avg_{sc}'
            if pd.notna(row.get(key_col, None)):
                chosen_sc = sc
                sc_data = {re.sub(f'_{sc}$', '_sc', col): row[col]
                           for col in df_wide.columns if col.endswith(f'_{sc}')}
                sc_data['sc_sw'] = sc
                break

        if chosen_sc is not None:
            rows.append(pd.Series(sc_data, name=idx))

    df_combined = pd.DataFrame(rows)
    df_combined.index.name = 'epoch'
    df_combined.attrs = df_merged_attrs

    print('Merging')
    df_final = merge_dataframes(df_sw, df_combined, suffix_1='sw', suffix_2=None)
    df_final.columns = [param_map_pc.get(col,col) for col in df_final.columns]
    df_final.attrs['units'] = {param_map_pc.get(col,col): df_final.attrs['units'].get(col,col) for col in df_final.attrs['units']}
    df_final.attrs['units']['sc_sw'] = 'STRING'

    # Write
    output_file = os.path.join(MSH_DIR, data_pop, sample_interval, 'sw_times_combined.cdf')
    print('Writing combined to file...')
    write_to_cdf(df_final, output_file, reset_index=True)



# %%