# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:52:33 2026

@author: richarj2
"""

import os
import numpy as np
import pandas as pd

from scipy.constants import physical_constants
m_a = physical_constants['alpha particle mass'][0]

from .delay import calc_flat_delay

from ...config import get_proc_directory, get_data_populations
from ...processing.reading import import_processed_spacecraft, import_processed_omni
from ...processing.dataframes import merge_dataframes
from ...processing.writing import write_to_cdf

from ...coordinates.boundaries import calc_msh_dist, vector_component_surface

all_spacecraft = ('c1','c3','mms1','tha','thb','thc','thd','the')
sw_spacecraft  = ('c1','c3','mms1','thb','thc')
msh_spacecraft = all_spacecraft

filter_spacecraft = {'sw': sw_spacecraft, 'msh': all_spacecraft, 'msh20': all_spacecraft}

pos_cols  = ['r_MP','r_BS','r_phi','r_F']
norm_vecs = {'field': ('B',), 'plasma': ('B','E','V','S')}

# %% procedure

def filter_sc_region(sc, region, data_pop='plasma', resolution='5min', df_omni=None, test_return=True):

    """
    data_pop = 'field' means field only
    data_pop = 'plasma' means including plasma (so also field)
    """

    ###----------IMPORTS----------###
    print(f'Importing {sc.upper()} {region} {resolution}.\n')

    restricted = False
    if region=='msh20':
        restricted = True
        region = 'msh'

    populations = get_data_populations(sc, data_pop, region)

    df_sc = import_processed_spacecraft(sc, populations, resolution)

    length0 = len(df_sc)

    if df_omni is None:
        print('Importing OMNI.\n')
        # contains updated flow pressure calculation
        df_omni = import_processed_omni(resolution=resolution)

    df_merged_attrs = df_sc.attrs # stores
    df_merged = merge_dataframes(df_omni, df_sc, suffices=('sw', sc))

    length1 = len(df_merged)

    ###----------FILTERING----------###

    params = {'msh': {'r_F_max': 0.2}} if restricted else {}

    times = filter_region(df_merged, sc, region, params)
    if len(times)==0:
        print(f'No {sc} data in {region} after filtering.\n')
        return

    df_filtered = df_merged.loc[df_merged.index.isin(times)]
    suffix = f'_{sc}'

    length2 = len(df_filtered)
    del df_merged

    ###----------EXTRA PARAMETERS----------###

    if region in ('msh', 'msh20'):

        calc_msh_dist(df_filtered, position_key=sc, data_key='sw', inplace=True)

        # Components parallel/perp to bs/mp surface
        vector_component_surface(df_filtered, sc, region, data_pop)

    # Correction to time lag based on spacecraft position in solar wind
    calc_flat_delay(df_filtered, region=region, pos_col=f'r_x_GSE_{sc}', vel_col=f'V_x_GSE_{sc}', omni_vel_col='V_x_GSE_sw', omni_pres_col='P_p_sw', lag_col=f'prop_time_s_{sc}')

    df_filtered.dropna(subset=f'prop_time_s_{sc}', inplace=True)

    update_parameters(df_filtered, sc, region)

    length3 = len(df_filtered)

    ###----------WRITING----------###

    # Add to combined
    df_filtered.drop(columns=[col for col in df_filtered if (col.endswith('_sw') or col.endswith('_pc'))], inplace=True)

    # These columns are naturally in mms data so removing for consistency
    df_filtered.drop(columns=[col for col in df_filtered.columns if col.startswith('r_avg')], inplace=True)

    # Need suffix when combining but removing for individual file
    df_filtered.rename(columns={col: col[:-len(suffix)] for col in df_filtered.columns if col.endswith(suffix)}, inplace=True)

    df_filtered.dropna(how='all',inplace=True)
    update_attributes(df_filtered, df_merged_attrs, region)

    if test_return:
        return df_filtered

    print(f'Writing {sc} to file...')
    region = 'msh20' if restricted else region
    out_dir = get_proc_directory(region, dtype=data_pop, resolution=resolution, create=True) # output directory
    write_to_cdf(df_filtered, directory=out_dir, file_name=f'{region}_times_{sc}', reset_index=True)

    ###----------WRITING----------###

    info  = f'\nFiltering {sc} in {region} using {resolution} data:\n'
    info += f'Imported data (yearly region files): {length0:,}\n'
    info += f'After merging with OMNI            : {length1:,}\n'
    info += f'After applying region filtering    : {length2:,}\n'
    info += f'After dropping NaN lag times       : {length3:,}\n'
    with open(os.path.join(out_dir, 'Filtering_Information.txt'), 'a') as log_file:
        log_file.write(f'{info}\n')



def filter_region(df, sc, region, params={}):
    """
    Filter region of magnetosphere based on Blüthner et al. (2026) for solar wind and adapted for magnetosheath.
    df contains both spacecraft and OMNI data for filtering conditions.
    Returns the times when the spacecraft is in the region.

    NOTE: This doesn't explicitly remove ARTEMIS Data. This is written to the file, but is removed when the files are merged into a complete file.
    """
    default_params = {'sw':  {'r_x': 7, 'r_theta': np.pi/6, 'V_sw':  275},
                      'msh': {'r_theta': np.pi/4, 'N_tot': 6, 'r_F_min': 0.05, 'r_F_max': 0.85, 'buffer': 'relative'}}

    for reg, vals in params.items():
        default_params[reg].update(vals)

    df = df.loc[df[f'r_x_GSE_{sc}']>0].copy()

    spacecraft_distance(df, f'_{sc}')

    ###-----Parameter-----###

    rcone = default_params[region]['r_theta']

    if region=='sw':

        # Region Filtering

        rx    = default_params['sw']['r_x']
        vflow = default_params['sw']['V_sw']

        mask =  (df[f'r_x_GSE_{sc}']>=rx) & (df[f'r_cone_{sc}']<=rcone)
        mask &= (df[f'V_flow_{sc}'] >= vflow)

    elif region=='msh':

        df_positions = calc_msh_dist(df, position_key=sc, data_key='sw')
        df           = pd.concat([df, df_positions[pos_cols]], axis=1)

        # Region Filtering

        ntot   = default_params['msh']['N_tot']
        rFmin  = default_params['msh']['r_F_min']
        rFmax  = default_params['msh']['r_F_max']
        buffer = default_params['msh']['buffer']

        mask =  (df[f'r_cone_{sc}'] <= rcone)
        mask &= (df[f'N_tot_{sc}'] >= ntot)

        if buffer=='relative':
            mask &= (df['r_F'] > rFmin) & (df['r_F'] < rFmax) # tweak based on comparison with Toy-Edens
        # elif buffer=='absolute':
        #     mask &= (df_merged[f'r_mag_{sc}'] > df_merged['r_MP'] + rFmin) & (df_merged[f'r_mag_{sc}'] < df_merged['r_BS'] - rFmax)


    # Post-processing Cleaning
    mask &= (df[f'S_x_GSE_{sc}'] > -1000)
    mask &= (df[f'V_x_GSE_{sc}'] < 0)

    return df.index[mask]


# %% updates

def spacecraft_distance(df, suffix=''):

    if f'r_mag{suffix}' not in df:
        cols     = [f'r_{comp}_GSE{suffix}' for comp in ('x','y','z')]
        r        = (df[cols].to_numpy()**2).sum(axis=1)**0.5

        try:
            unc_cols = [f'r_{comp}_GSE_unc{suffix}' for comp in ('x','y','z')]
            sigma_r = (((df[cols].to_numpy() / r)**2 * df[unc_cols].to_numpy()**2).sum(axis=1))**0.5
        except:
            sigma_r = np.nan

        df.insert(0, f'r_mag{suffix}', r)
        df.insert(1, f'r_mag_unc{suffix}', sigma_r)

        df.attrs['units']['r_mag'] = 'Re'
        df.attrs['units']['r_mag_unc'] = 'Re'

    if f'r_mag_count{suffix}' in df:
        df.drop(columns=[f'r_mag_count{suffix}'],inplace=True)

    if f'r_cone{suffix}' not in df:

        x = df[f'r_x_GSE{suffix}'].to_numpy()
        y = df[f'r_y_GSE{suffix}'].to_numpy()
        z = df[f'r_z_GSE{suffix}'].to_numpy()
        r = df[f'r_mag{suffix}'].to_numpy()

        cone = np.arccos(x / r)

        df.insert(2, f'r_cone{suffix}', cone)

        try:
            sx = df[f'r_x_GSE_unc{suffix}'].to_numpy()
            sy = df[f'r_y_GSE_unc{suffix}'].to_numpy()
            sz = df[f'r_z_GSE_unc{suffix}'].to_numpy()

            yz = np.sqrt(y**2 + z**2)
            yz = np.where(yz == 0, np.nan, yz) # if lies on Earth-Sun line

            dtheta_dx = -yz / r**2
            dtheta_dy = x * y / (r**2 * yz)
            dtheta_dz = x * z / (r**2 * yz)

            sigma_cone = np.sqrt((dtheta_dx * sx)**2 + (dtheta_dy * sy)**2 + (dtheta_dz * sz)**2)

        except:
            sigma_cone = np.full(len(df), np.nan)

        df.insert(3, f'r_cone_unc{suffix}', sigma_cone)


        df.attrs['units']['r_cone'] = 'rad'
        df.attrs['units']['r_cone_unc'] = 'rad'

def update_parameters(df, sc, region):
    """
    Removes erroneous values
    """

    suffix = f'_{sc}'

    for temp in (f'T_tot{suffix}',f'T_tot_unc{suffix}'):
        df[temp] /= 1e6 #convert to MK

    # if region=='sw':
    #     remove_extremes(df, {f'beta{suffix}': 100, f'N_tot{suffix}': 300})

    # elif region=='msh':
    #     remove_extremes(df, {}, {})

def remove_extremes(df, mapping={}, abs_mapping={}):

    mask = np.ones_like(len(df))

    for col, limit in mapping.items():
        mask &= df[col] < limit

    for col, limit in abs_mapping.items():
        mask &= df[col].abs() < limit

    df = df.loc[mask]

def update_attributes(df, attrs_dict, region='sw', suffix=''):

    df.attrs = attrs_dict
    for temp in (f'T_tot{suffix}',f'T_tot_unc{suffix}'):
        df.attrs['units'][temp] = 'MK'

    df.attrs['units']['prop_time_s{suffix}'] = 's'

