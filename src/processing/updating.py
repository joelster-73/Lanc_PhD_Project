# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 11:18:29 2025

@author: richarj2
"""

import os
import glob

import numpy as np
import pandas as pd
from uncertainties import unumpy as unp
from scipy.constants import mu_0, m_p
from scipy.constants import physical_constants
m_a = physical_constants['alpha particle mass'][0]

from .dataframes import add_df_units, resample_data
from .writing import write_to_cdf
from .reading import import_processed_data
from .utils import create_directory

from ..coordinates.magnetic import convert_GSE_to_GSM_with_angles
from ..config import R_E, get_proc_directory, CLUSTER_SPACECRAFT, THEMIS_SPACECRAFT, MMS_SPACECRAFT

from .cluster.handling import update_hia_data, filter_hia_data
from .themis.handling import update_esa_data, filter_esa_data
from .mms.handling import update_fpi_data
from .mms.config import ION_MASS_DICT, ION_SPECIES

INIT_FUNCTIONS = {}
FILT_FUNCTIONS = {}
for sc in CLUSTER_SPACECRAFT:
    INIT_FUNCTIONS[sc] = update_hia_data
    FILT_FUNCTIONS[sc] = filter_hia_data
for sc in THEMIS_SPACECRAFT:
    INIT_FUNCTIONS[sc] = update_esa_data
    FILT_FUNCTIONS[sc] = filter_esa_data
for sc in MMS_SPACECRAFT:
    INIT_FUNCTIONS[sc] = update_fpi_data


# %% Helpers

def vec_cols(field, vec_coords='GSE'):
    return [f'{field}_x_{vec_coords}',f'{field}_y_{vec_coords}',f'{field}_z_{vec_coords}']

def insert_magnitude(df, field, vec_coords, suffix='mag'):
    new_column = f'{field}_{suffix}'
    x_column   = df.columns.get_loc(f'{field}_x_{vec_coords}')# insert before x column

    if new_column not in df:
        df.insert(x_column, new_column, (df[[f'{field}_{c}_{vec_coords}' for c in ('x','y','z')]].pow(2).sum(axis=1))**0.5)


# %% Resample

def rename_files(directory, old_string, new_string):
    for filename in os.listdir(directory):
        if old_string in filename:
            old_path = os.path.join(directory, filename)
            new_filename = filename.replace(old_string, new_string)
            new_path = os.path.join(directory, new_filename)
            os.rename(old_path, new_path)

def get_file_keys(spacecraft, data, raw_res='raw'):
    file_keys = {}
    data_dir = get_proc_directory(spacecraft, dtype=data, resolution=raw_res)
    pattern = os.path.join(data_dir, '*.cdf')
    for cdf_file in sorted(glob.glob(pattern)):
        file_name = os.path.splitext(os.path.basename(cdf_file))[0]
        key = file_name.split('_')[-1] # splitext removes the extension
        file_keys[key] = file_name
    return file_keys


def resample_monthly_files(spacecraft, data, raw_res='spin', sample_intervals=('1min',), time_col='epoch', overwrite=True, start_year=None):
    """
    Resample monthly files into yearly files.
    """
    ###----------SET UP----------###
    print('Resampling.')

    save_directories = {}

    for sample_interval in sample_intervals:

        save_directory = get_proc_directory(spacecraft, dtype=data, resolution=sample_interval, create=True)
        create_directory(save_directory)
        save_directories[sample_interval] = save_directory

    raw_dir = get_proc_directory(spacecraft, dtype=data, resolution=raw_res)
    pattern = os.path.join(raw_dir, '*.cdf')
    files_by_year = {}

    for cdf_file in sorted(glob.glob(pattern)):
        file_name = os.path.basename(cdf_file)
        dir_name = '_'.join(file_name.split('_')[:-1])
        year = file_name.split('_')[-1][:4]
        files_by_year.setdefault(year, []).append(file_name)

    ###----------PROCESS----------###
    for year, files in files_by_year.items():
        print(f'Updating {year}.')
        yearly_list = []
        if start_year and int(year)<start_year:
            continue

        for file in files:
            df = import_processed_data(spacecraft, dtype=data, resolution=raw_res, file_name=file)
            yearly_list.append(df)

        yearly_df = pd.concat(yearly_list) # don't set ignore_index to True
        yearly_df.drop(columns=[c for c in yearly_df.columns if c.endswith('_unc')],inplace=True) # measurement error << statistical uncertainty
        add_df_units(yearly_df)

        for sample_interval, samp_dir in save_directories.items():

            sampled_df = resample_data(yearly_df, time_col='index', sample_interval=sample_interval)

            attributes = {'sample_interval': sample_interval, 'time_col': time_col, 'R_E': R_E}
            print(f'{sample_interval} reprocessed.')
            write_to_cdf(sampled_df, directory=samp_dir, file_name=f'{dir_name}_{year}', attributes=attributes, overwrite=overwrite, reset_index=True)


# %% Plasma

def update_plasma_data(spacecraft, field='fgm', plasma='mom', ion_source='omni', regions=('sw',), **kwargs):
    """
    Takes in the (monthly) plasma files and uses (monthly) fgm files (and OMNI)
    Then will save files in a 'spin' directory to be re-sampled after
    init_func is to initilise the dataframe (including quality filters)
    filter_func filters the data for the sw/ms region
    """
    save_directory = get_proc_directory(spacecraft, dtype=plasma, resolution='spin', create=True)

    field_keys  = get_file_keys(spacecraft, field)
    plasma_keys = get_file_keys(spacecraft, plasma)

    field_res   = kwargs.get('field_res','raw')
    plasma_res  = kwargs.get('plasma_res','raw')

    if ion_source == 'omni':
        heavy_ions = import_processed_data('omni', resolution='5min')
        heavy_ions = heavy_ions[['na_np_ratio']]
    elif ion_source == 'hpca':
        ion_species = kwargs.get('ion_species',ION_SPECIES)
        heavy_ions = import_processed_data(spacecraft, dtype='hpca', resolution='raw')
        heavy_ions = heavy_ions[[f'N_{ion}' for ion in ion_species]] # Just need ion density columns
    elif ion_source == 'none': # e.g. HPCA raw data
        heavy_ions = None

    init_func = INIT_FUNCTIONS.get(spacecraft,None)
    filt_func = FILT_FUNCTIONS.get(spacecraft,None)

    for key, field_file in field_keys.items():
        plasma_file = plasma_keys.get(key,None)
        if not plasma_file:
            continue

        print(f'Updating {key}...')

        ###----------PROCESSING----------###

        field_df  = import_processed_data(spacecraft, field, resolution=field_res, file_name=field_file)
        plasma_df = import_processed_data(spacecraft, plasma, resolution=plasma_res, file_name=plasma_file)

        merged_df = init_func(field_df, plasma_df) # initial processing

        if spacecraft in CLUSTER_SPACECRAFT or spacecraft in THEMIS_SPACECRAFT:
            uncertainties = False
        elif spacecraft in MMS_SPACECRAFT:
            uncertainties = True

        if heavy_ions is None:
            updated_df = merged_df
        else:
            if '-' in key:
                year, month = key.split('-')
                mask = (heavy_ions.index.year==int(year))&(heavy_ions.index.month==int(month))
            else:
                mask = heavy_ions.index.year==int(key)

            ion_df = heavy_ions.loc[mask]
            updated_df = process_plasma_data(merged_df, ion_df, ion_source, with_unc=uncertainties, **kwargs)

            cols_to_drop = [col for col in field_df if col in updated_df]
            updated_df.drop(columns=cols_to_drop, inplace=True)

        ###----------WRITE TO FILE----------###

        print(f'{key} processed.')
        write_to_cdf(updated_df, directory=save_directory, file_name=f'{spacecraft.upper()}_SPIN_{key}', attributes={'R_E': R_E}, overwrite=True, reset_index=True)

        if not filt_func:
            print('No region filter function.')
            continue

        for region in regions:
            filtered_df = filt_func(updated_df, region)
            if filtered_df.empty:
                print(f'No {region} data in correct mode.')
                continue

            print(f'-{region}- data...')
            region_save_directory = get_proc_directory(spacecraft, dtype=region, resolution='spin', create=True)
            write_to_cdf(filtered_df, directory=region_save_directory, file_name=f'{spacecraft.upper()}_{region.upper()}_{key}', attributes={'R_E': R_E}, overwrite=True, reset_index=True)


def process_plasma_data(merged_df, ion_df, ion_source, with_unc=False, convert_fields=None, **kwargs):
    """
    Once the plasma data has been extracted from the raw moments files
    This function will combine into total flow pressure and velocity etc
    """
    def assign_values(df, column, uarr, col_before=None):

        vals = unp.nominal_values(uarr) if with_unc else np.asarray(uarr, dtype=float)

        if not isinstance(column, str):  # list of columns
            df.loc[:, column] = vals
            if with_unc:
                df.loc[:, [f'{c}_unc' for c in column]] = unp.std_devs(uarr)

        elif col_before:
            idx = df.columns.get_loc(col_before)
            df.insert(idx, column, vals)
            if with_unc:
                df.insert(idx + 1, f'{column}_unc', unp.std_devs(uarr))

        else:
            df[column] = vals
            if with_unc:
                df[f'{column}_unc'] = unp.std_devs(uarr)

    def build_uarr(df, columns):

        values = df[columns].to_numpy()
        if not with_unc:
            return values

        values = np.nan_to_num(values, nan=0.0)

        if isinstance(columns, str):
            # single column
            try:
                uncs = np.nan_to_num(df[f'{columns}_unc'].to_numpy(), nan=0.0)
            except:
                uncs = np.zeros(len(values))
        else:
            # list of columns
            try:
                uncs = np.nan_to_num(df[[f'{c}_unc' for c in columns]].to_numpy(), nan=0.0)
            except:
                uncs = np.zeros(len(values))

        return unp.uarray(values, uncs)

    def cross_u(a, b):
        if with_unc:
            # Handles uncertainties
            return np.stack([
                a[:,1]*b[:,2] - a[:,2]*b[:,1],
                a[:,2]*b[:,0] - a[:,0]*b[:,2],
                a[:,0]*b[:,1] - a[:,1]*b[:,0]
            ], axis=1)

        return np.cross(a,b)

    ###----------CALCULATIONS----------###
    insert_magnitude(merged_df, 'V', 'GSE', 'flow')

    if with_unc and 'V_flow_unc' not in merged_df:
        idx = merged_df.columns.get_loc('V_flow')
        merged_df.insert(idx+1, 'V_flow_unc', (merged_df['V_x_GSE']**2 * merged_df['V_x_GSE_unc']**2 + merged_df['V_y_GSE']**2 * merged_df['V_y_GSE_unc']**2 + merged_df['V_z_GSE']**2 * merged_df['V_z_GSE_unc']**2) ** 0.5 / merged_df['V_flow'])

    v_flow = build_uarr(merged_df, 'V_flow')

    # Dynamic pressure
    merged_df = calc_avg_ion_mass(merged_df, ion_df, ion_source, **kwargs)

    n_tot   = build_uarr(merged_df, 'N_tot')
    rho_tot = (merged_df['m_avg_ratio']*m_p) * n_tot # kg/cc

    if with_unc:
        mask = np.array(unp.nominal_values(rho_tot)) <= 0
        rho_tot[mask] = unp.uarray(np.nan, 0)
    else:
        rho_tot[rho_tot <= 0] = np.nan

    # P = 0.5 * rho * V^2
    # N *= 1e6, V *= 1e6, P *= 1e9, so P_flow *= 1e21
    p_flow = 0.5 * rho_tot * v_flow**2 * 1e21
    assign_values(merged_df, 'P_flow', p_flow)

    # Beta = p_th / p_mag, p_mag = B^2/2mu_0
    # p_dyn *= 1e-9, 1/B_avg^2 *= 1e18, so beta *= 1e9
    P_th = build_uarr(merged_df, 'P_th')
    beta = P_th / (merged_df['B_avg']**2) * (2*mu_0) * 1e9
    assign_values(merged_df, 'beta', beta)

    # Alfven Speed = B / sqrt(mu_0 * rho)
    # B_avg *= 1e-9, 1/sqrt(rho) *= 1e-3, vA *= 1e-3, so speed *= 1e-15
    V_A = merged_df['B_avg'] / unp.sqrt(mu_0 * rho_tot) * 1e-15
    assign_values(merged_df, 'V_A', V_A)

    ###----------GSE to GSM----------###

    print('Rotating...')

    if convert_fields is not None:
        convert_columns = [vec_cols(field,'GSE') for field in convert_fields]

    gsm_vectors = convert_GSE_to_GSM_with_angles(merged_df, convert_columns, ref='B', interp=True, include_unc=with_unc)
    merged_df   = pd.concat([merged_df,gsm_vectors], axis=1)
    vec_coords = 'GSM'

    # Clock Angle: theta = atan2(By,Bz)
    if 'B_clock' not in merged_df:
        B_clock = np.arctan2(merged_df[f'B_y_{vec_coords}'], merged_df[f'B_z_{vec_coords}'])
    else:
        B_clock = merged_df['B_clock']

    # Kan and Lee Electric Field: E_R = |V| * B_T * sin^2 (clock/2)
    # V *= 1e3, B *= 1e-9, E *= 1e3, so E_R *= 1e-3
    E_R = (v_flow * np.sqrt(merged_df[f'B_y_{vec_coords}']**2+merged_df[f'B_z_{vec_coords}']**2) * (np.sin(B_clock/2))**2) * 1e-3
    assign_values(merged_df, 'E_R', E_R)

    ###----------CROSS PRODUCTS----------###

    # Build uarray for V
    V = build_uarr(merged_df, vec_cols('V',vec_coords))
    B = merged_df[vec_cols('B',vec_coords)].to_numpy()

    # E = -V x B = B x V
    # V *= 1e3, B *= 1e-9, and E *= 1e3 so E_gse *= 1e-3
    E = cross_u(B, V) * 1e-3
    assign_values(merged_df, vec_cols('E',vec_coords), E)

    E_mag = unp.sqrt(np.sum(E**2, axis=1))
    assign_values(merged_df, 'E_mag', E_mag, col_before=f'E_x_{vec_coords}')

    # S = E x H = E x B / mu_0
    # E *= 1e-3, B *= 1e-9, and S *= 1e6 so S_gse *= 1e-6
    S = cross_u(E, B) * 1e-6 / mu_0
    assign_values(merged_df, vec_cols('S',vec_coords), S)

    S_mag = unp.sqrt(np.sum(S**2, axis=1))
    assign_values(merged_df, 'S_mag', S_mag, col_before=f'S_x_{vec_coords}')

    return merged_df

def calc_avg_ion_mass(merged_df, ion_df, ion_source, **kwargs):
    """
    This method assumes the relative amount of ions measured by the HPCA instrument (so the ratios) are correct.
    This ratio is then scaled by the total denisty measured by FPI.
    """

    try: # find ratio of alpha and proton densities

        if ion_source=='hpca':
            print('Using HPCA data.')
            ion_mass_dict = kwargs.get('ion_mass_dict', ION_MASS_DICT)

            # avoid extrapolation
            ion_interp = ion_df.reindex(merged_df.index).interpolate(method='time')
            ion_interp = ion_interp.loc[(ion_interp.index >= ion_df.index.min()) & (ion_interp.index <= ion_df.index.max())]

            # hplus density saturates in HPCA
            ion_ratio_map = {'heplus': 'nhe_np_ratio', 'oplus': 'no_np_ratio', 'heplusplus': 'na_np_ratio'}
            default_ratios = {'heplus': 0.0005, 'oplus': 0.01, 'heplusplus': 0.05}  # default values

            num = m_p
            den = 1.0

            idx = merged_df.columns.get_loc('N_tot')
            for ion, ratio_col in ion_ratio_map.items():

                r = ion_interp[f'N_{ion}'] / ion_interp['N_hplus']
                r = r.fillna(default_ratios[ion]).clip(lower=0)

                try:
                    merged_df.insert(idx+2, ratio_col, r)
                except:
                    merged_df[ratio_col] = r

                num += r * ion_mass_dict[ion]
                den += r

            m_avg = num / den # kg
            merged_df.insert(idx+2, 'm_avg_ratio', m_avg/m_p)

        elif ion_source=='omni':
            print('Using OMNI alpha ratio.')

            merged_df = pd.merge_asof(merged_df.sort_index(), ion_df[['na_np_ratio']].sort_index(), left_index=True, right_index=True, direction='backward')   # take the closest value on or before the timestamp
            m_avg   = (m_p + merged_df['na_np_ratio'] * m_a) / (merged_df['na_np_ratio'] + 1) # kg
            idx = merged_df.columns.get_loc('N_tot')
            merged_df.insert(idx+2, 'm_avg_ratio', m_avg/m_p)

    except:
        print('Using default alpha ratio')
        default_ratio = kwargs.get('default_ratio', 0.05)

        idx = merged_df.columns.get_loc('N_tot')
        if 'na_np_ratio' in merged_df:
            merged_df['na_np_ratio'] = default_ratio
        else:
            merged_df.insert(idx+2, 'na_np_ratio', default_ratio)
        m_avg   = (m_p + merged_df['na_np_ratio'] * m_a) / (merged_df['na_np_ratio'] + 1) # kg
        merged_df.insert(idx+2, 'm_avg_ratio', m_avg/m_p)

    return merged_df
