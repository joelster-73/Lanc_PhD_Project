# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 15:52:33 2026

@author: richarj2
"""

import numpy as np
import pandas as pd

from scipy.constants import physical_constants
m_a = physical_constants['alpha particle mass'][0]

from .delay import calc_flat_delay

from ...config import get_proc_directory, get_data_populations
from ...processing.reading import import_processed_spacecraft, import_processed_data
from ...processing.dataframes import merge_dataframes
from ...processing.writing import write_to_cdf

from ...coordinates.boundaries import calc_msh_dist, vector_component_surface

all_spacecraft = ('c1','c3','mms1','tha','thb','thc','thd','the')
sw_spacecraft  = ('c1','c3','mms1','thb','thc')
msh_spacecraft = ('c1','c3','mms1','tha','thd','the')

filter_spacecraft = {'sw': sw_spacecraft, 'msh': all_spacecraft}

pos_cols  = ['r_MP','r_BS','r_phi','r_F']
norm_vecs = {'field': ('B',), 'plasma': ('B','E','V','S')}

# %% procedure

def filter_sc_region(sc, region, data_pop='plasma', resolution='5min', df_omni=None, test_return=True):

    """
    data_pop = 'field' means field only
    data_pop = 'plasma' means including plasma (so also field)
    """

    print(f'Processing {resolution} {region} data.\n')

    ###----------IMPORTS----------###
    if df_omni is None:
        print('Importing OMNI.\n')
        df_omni = import_processed_data('omni', resolution=resolution)

    print(f'Importing {sc.upper()}.')

    populations = get_data_populations(sc, data_pop, region)

    df_sc = import_processed_spacecraft(sc, populations, resolution)

    ###----------FILTERING----------###

    df_merged, df_merged_attrs = filter_region(df_sc, df_omni, sc, region)
    if df_merged.empty:
        print(f'No {sc} data in {region}.')
        return

    suffix = f'_{sc}'
    df_merged.rename(columns={col: f'{col}{suffix}' for col in pos_cols}, inplace=True) # adds _sc suffix

    ###----------EXTRA PARAMETERS----------###

    # Components parallel/perp to bs/mp surface
    vector_component_surface(df_merged, sc, region, data_pop)

    # Correction to time lag based on spacecraft position in solar wind
    calc_flat_delay(df_merged, region=region, pos_col=f'r_x_GSE_{sc}', pres_col=f'P_flow_{sc}', vel_col=f'V_x_GSE_{sc}', lag_col=f'prop_time_s_{sc}')

    update_parameters(df_merged, sc, region)

    ###----------WRITING----------###

    # Add to combined
    df_merged.dropna(how='all',inplace=True)
    df_merged.drop(columns=[col for col in df_merged if (col.endswith('_sw') or col.endswith('_pc'))], inplace=True)

    # Need suffix when combining but removing for individual file
    df_merged = df_merged.rename(columns={col: col[:-len(suffix)] for col in df_merged.columns if col.endswith(suffix)})
    update_attributes(df_merged, df_merged_attrs, region, suffix)

    if test_return:
        return df_merged

    print(f'Writing {sc} to file...')
    out_dir = get_proc_directory(region, dtype=data_pop, resolution=resolution, create=True) # output directory
    write_to_cdf(df_merged, directory=out_dir, file_name=f'{region}_times_{sc}', reset_index=True)



def filter_region(df, omni, sc, region, params={}):
    """
    Filter region of magnetosphere based on Blüthner et al. (2026) for solar wind and adapted for magnetosheath
    """
    default_params = {'sw':  {'r_x': 7, 'r_theta': np.pi/6, 'V_sw': 275},
                      'msh': {'r_x': 5, 'r_theta': np.pi/4, 'N_tot': 10, 'r_F_min': 0, 'r_F_max': 1}}

    for reg, vals in params.items():
        default_params[reg].update(vals)

    ###-----Location-----###

    df = df.loc[df['r_x_GSE']>0].copy()

    spacecraft_distance(df)

    df_merged_attrs = df.attrs # stores
    df_merged = merge_dataframes(omni, df, suffices=('sw', sc))

    ###-----Parameter-----###

    df_positions = calc_msh_dist(df_merged, position_key=sc, data_key='sw')
    df_merged    = pd.concat([df_merged, df_positions[pos_cols]], axis=1)

    rx    = default_params[region]['r_x']
    rcone = default_params[region]['r_theta']

    if region=='sw':
        vflow = default_params['sw']['V_sw']

        mask =  (df_merged[f'r_x_GSE_{sc}']>=rx) & (df_merged[f'r_cone_{sc}']<=rcone)
        mask &= (df_merged[f'V_flow_{sc}'] >= vflow)

    elif region=='msh':
        ntot  = default_params['msh']['N_tot']
        rFmin = default_params['msh']['r_F_min']
        rFmax = default_params['msh']['r_F_max']

        mask =  (df_merged[f'r_x_GSE_{sc}']>=rx) & (df_merged[f'r_cone_{sc}']<=rcone)
        mask &= (df_merged['r_F'] > rFmin) & (df_merged['r_F'] < rFmax) # tweak based on comparison with Toy-Edens
        mask &= (df_merged[f'N_tot_{sc}'] >= ntot)  # test how sensitive results are to this


    df_merged = df_merged.loc[mask]

    return df_merged, df_merged_attrs


# %% updates

def spacecraft_distance(df):

    if 'r_mag' not in df:
        cols     = [f'r_{comp}_GSE' for comp in ('x','y','z')]
        r        = (df[cols].to_numpy()**2).sum(axis=1)**0.5

        try:
            unc_cols = [f'r_{comp}_GSE_unc' for comp in ('x','y','z')]
            sigma_r = (((df[cols].to_numpy() / r)**2 * df[unc_cols].to_numpy()**2).sum(axis=1))**0.5
        except:
            sigma_r = np.nan

        df.insert(0, 'r_mag', r)
        df.insert(1, 'r_mag_unc', sigma_r)

        df.attrs['units']['r_mag'] = 'Re'
        df.attrs['units']['r_mag_unc'] = 'Re'

    if 'r_mag_count' in df:
        df.drop(columns=['r_mag_count'],inplace=True)

    if 'r_cone' not in df:

        x = df['r_x_GSE'].to_numpy()
        y = df['r_y_GSE'].to_numpy()
        z = df['r_z_GSE'].to_numpy()
        r = df['r_mag'].to_numpy()

        cone = np.arccos(x / r)

        df.insert(2, 'r_cone', cone)

        try:
            sx = df['r_x_GSE_unc'].to_numpy()
            sy = df['r_y_GSE_unc'].to_numpy()
            sz = df['r_z_GSE_unc'].to_numpy()

            yz = np.sqrt(y**2 + z**2)
            yz = np.where(yz == 0, np.nan, yz) # if lies on Earth-Sun line

            dtheta_dx = -yz / r**2
            dtheta_dy = x * y / (r**2 * yz)
            dtheta_dz = x * z / (r**2 * yz)

            sigma_cone = np.sqrt((dtheta_dx * sx)**2 + (dtheta_dy * sy)**2 + (dtheta_dz * sz)**2)

        except:
            sigma_cone = np.full(len(df), np.nan)

        df.insert(3, 'r_cone_unc', sigma_cone)


        df.attrs['units']['r_cone'] = 'rad'
        df.attrs['units']['r_cone_unc'] = 'rad'

def update_parameters(df, sc, region):
    """
    Removes erroneous values
    """

    suffix = f'_{sc}'

    for temp in (f'T_tot{suffix}',f'T_tot_unc{suffix}'):
        df[temp] /= 1e6 #convert to MK

    if region=='sw':
        remove_extremes(df, {f'beta{suffix}': 100, f'P_flow{suffix}': 15, f'E_mag{suffix}': 20, f'E_y_GSM{suffix}': 20, f'V_flow{suffix}' : 1400, f'N_tot{suffix}': 300})

    elif region=='msh':
        remove_extremes(df, {f'B_avg{suffix}': 100}, {f'B_z_GSM{suffix}': 250, f'beta{suffix}': 100})

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

