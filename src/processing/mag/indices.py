# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 12:41:52 2025

@author: richarj2
"""


import os
import glob
import numpy as np
import pandas as pd

import warnings

from spacepy import pycdf
from datetime import datetime, timedelta

from .config import PC_STATIONS

from ..reading import import_processed_data
from ..writing import write_to_cdf
from ..dataframes import resample_data, rename_columns

from ...config import get_luna_directory, get_proc_directory

# %% PCN

def process_PCN_data():

    pcn_dir = get_proc_directory('pcn')

    pattern = os.path.join(pcn_dir, '*.cdf')
    files = sorted(glob.glob(pattern))

    yearly_pcn = []

    for file in files:
        with pycdf.CDF(file) as cdf:
            df_year = pd.DataFrame({'time': cdf['Time'][...],'PCN': cdf['PCN'][...]})
            yearly_pcn.append(df_year)

    df_pcn = pd.concat(yearly_pcn, ignore_index=True)
    mjd2000_epoch = pd.Timestamp('2000-01-01')

    df_pcn['epoch'] = mjd2000_epoch + pd.to_timedelta(df_pcn['time'], unit='D')
    df_pcn['epoch'] = df_pcn['epoch'].dt.round('min')
    df_pcn.drop(columns=['time'],inplace=True)
    df_pcn.set_index('epoch',inplace=True)

    df_pcn.loc[df_pcn['PCN']>=999] = np.nan

    df_pcn.attrs = {'units': {'PCN': 'mV/m'}, 'time_col': 'epoch'}

    return df_pcn

def process_PCC_data(include_prelim=True):
    '''
    Include prelim is the 'preliminary' sub directory
    Extra data is on my local drive and not on LUNA
    '''
    pcn_pcs_dir = get_luna_directory('pc_index')


    def process_files(files_list):

        df = pd.concat([pd.read_csv(f, sep=r'\s+') for f in files_list], ignore_index=True)

        mask = df['h:m'].str.len() > 5

        # fix corrupted rows
        # e.g. 553857      2021-01-19  14:5799999.00    1.47   NaN
        for idx in df.index[mask]:
            val = df.at[idx, 'h:m']
            time_part = val[:5]          # first 5 chars -> HH:MM
            pcn_part  = val[5:]           # rest -> PCN value

            df.at[idx, 'h:m'] = time_part
            df.at[idx, 'PCS'] = df.at[idx, 'PCN']  # backup old PCN to PCS
            df.at[idx, 'PCN'] = float(pcn_part)    # set new PCN

        df['epoch'] = pd.to_datetime(df['#year-month-day'] + ' ' + df['h:m'])
        df.drop(columns=['#year-month-day', 'h:m'], inplace=True)
        df.set_index('epoch', inplace=True)

        df.loc[df['PCN'] >= 999, 'PCN'] = np.nan
        df.loc[df['PCS'] >= 999, 'PCS'] = np.nan

        return df

    def merge_prelim_like(df_def, files):

        if not files:
            return df_def

        df_temp = process_files(files)
        df_temp.rename(columns={'PCN': 'PCN_prelim', 'PCS': 'PCS_prelim'}, inplace=True)

        # align indexes so new rows get added
        df_merged = df_def.join(df_temp, how='outer')

        for col_main, col_prelim in [('PCN', 'PCN_prelim'), ('PCS', 'PCS_prelim')]:
            df_merged[col_main] = df_merged[col_main].combine_first(df_merged[col_prelim])

        return df_merged

    main_files = sorted(f for f in glob.glob(os.path.join(pcn_pcs_dir, '*.txt'))
                        if 'readme' not in os.path.basename(f).lower())

    df_main = process_files(main_files)

    if include_prelim:
        prelim_files = sorted(glob.glob(os.path.join(pcn_pcs_dir, 'preliminary', '*.txt')))
        df_main = merge_prelim_like(df_main, prelim_files)

    df_main = df_main.loc[df_main.index.year > 2000]
    df_main['PCC'] = df_main[['PCN', 'PCS']].clip(lower=0).mean(axis=1, skipna=True)
    df_main.drop(columns=['PCN','PCS','PCN_prelim','PCS_prelim'],inplace=True,errors='ignore')
    df_main.sort_index(inplace=True)

    return df_main

# %% Ring Current

def process_Dst_data():

    dst_dir = get_proc_directory('dst')

    records = []

    for file_name in sorted(os.listdir(dst_dir)):

        file_path = os.path.join(dst_dir, file_name)
        if not os.path.isfile(file_path) or file_name.lower() == 'readme.txt':
            continue

        with open(file_path, 'r') as file:
            days = file.readlines()

        for line in days:

            if len(line.strip()) < 120:
                continue

            yr      = line[3:5]
            century = line[14:16].strip()
            century = int(century) if century else 19  # blank = 19XX

            year    = century * 100 + int(yr)
            month   = int(line[5:7])
            day     = int(line[8:10])

            date    = datetime(year, month, day)

            version = line[13] # 0 - quicklook, 1 - provisional, 2 - final

            base_value = int(line[16:20])

            # 24 hourly values, 4 digits each
            hourly_str = line[20:116]
            hourly_values = [int(hourly_str[i:i+4]) for i in range(0, 96, 4)]

            for hour, value in enumerate(hourly_values):
                records.append({
                    'epoch': date + timedelta(hours=hour),
                    'Dst': value + base_value,
                    'version': version
                })

    df_dst = pd.DataFrame(records)
    df_dst.set_index('epoch',inplace=True)

    df_dst['Dst'] = df_dst['Dst'].replace(9999, np.nan)

    df_dst.attrs = {'units': {'Dst': 'nT'}, 'time_col': 'epoch'}
    return df_dst


# %% SuperMAG
def process_SMR_data():

    file_path = os.path.join(get_proc_directory('smr'), 'SMR_indices.txt')

    df_smr = pd.read_csv(file_path, skiprows=104, sep='\t')
    df_smr = df_smr.loc[df_smr['<year>'] > 1995]

    df_smr.rename(columns={'<year>': 'year', '<month>': 'month', '<day>': 'day', '<hour>': 'hour', '<min>': 'minute', '<SMR (nT)>': 'SMR'},inplace=True)

    df_smr['epoch'] = pd.to_datetime(df_smr[['year','month','day','hour','minute']])

    df_smr.set_index('epoch',inplace=True)
    df_smr.drop(columns=['year','month','day','hour','minute','<sec>'],inplace=True)

    df_smr.loc[df_smr['SMR']>=3000] = np.nan

    df_smr.attrs = {'units': {'SMR': 'nT'}, 'time_col': 'epoch'}

    return df_smr


def process_SME_data():

    file_path = os.path.join(get_proc_directory('sme'), 'SME_indices.txt')

    df_sme = pd.read_csv(file_path, skiprows=104, sep='\t')
    df_sme = df_sme.loc[df_sme['<year>']>1995]

    df_sme.rename(columns={'<year>': 'year', '<month>': 'month', '<day>': 'day', '<hour>': 'hour', '<min>': 'minute', '<SME (nT)>': 'SME'},inplace=True)

    df_sme['epoch'] = pd.to_datetime(df_sme[['year','month','day','hour','minute']])

    df_sme.set_index('epoch',inplace=True)
    df_sme.drop(columns=['year','month','day','hour','minute','<sec>'],inplace=True)

    df_sme.loc[df_sme['SME']>=3000] = np.nan

    df_sme.attrs = {'units': {'SME': 'nT'}, 'time_col': 'epoch'}

    return df_sme


# %% Misc

def process_AA_data():

    aa_dir = get_proc_directory('aa')
    file_path = os.path.join(aa_dir, 'AA_indices.dat')

    df_aa = pd.read_csv(file_path, skiprows=34, sep=r'\s+', comment='|')
    df_aa['epoch'] = pd.to_datetime(df_aa['DATE'] + ' ' + df_aa['TIME'])

    df_aa.set_index('epoch',inplace=True)
    df_aa.drop(columns=['DATE','TIME','DOY','Kpa','Aa','CK24','CK48'],inplace=True)

    df_aa.loc[df_aa['aa']>=999] = np.nan

    df_aa.attrs = {'units': {'aa': 'nT'}, 'time_col': 'epoch'}

    return df_aa


# %% averaged_indices

resolutions = ['1min','5min','15min','1hour','3hour']
native_resolutions = {'AA': '3hour', 'Dst': '1hour'}

def build_averaged_index(index, sample_intervals=('1min','5min','15min','1hour')):

    """
    Procedure will take in one specific index, process the raw data files, then average and save to a file.
    Unlike the other functions, this will not do any time lagging, as interpolation isn't going to be allowed in the future studies.
    Ensure that the time stamps are continuous.
    """

    print(f'\nImporting {index}...')


    index_cols = {'AE': ['AE','AL','AU'], 'SYM': ['SYM_D','SYM_H'], 'ASY': ['ASY_D','ASY_H']}

    procedures = {'PCN':  process_PCN_data,'PCC': process_PCC_data, 'AA': process_AA_data, 'SME': process_SME_data, 'Dst': process_Dst_data, 'SMR': process_SMR_data}

    if index in index_cols.keys():
        # indices_columns = ['AE', 'AL', 'AU', 'SYM_D', 'SYM_H', 'ASY_D', 'ASY_H']
        # for consistency, averaging 1min omni data rather than 5min
        df_index = import_processed_data('omni', resolution='1min')[index_cols.get(index)]

        df_index.attrs = {'units': {col: 'nT' for col in index_cols.get(index)}, 'time_col': 'epoch'}

    elif index=='PC':
        df_pcc = procedures['PCC']()
        df_pcn = procedures['PCN']()
        df_index = pd.concat([df_pcn, df_pcc], axis=1)
        df_index.attrs = {'units': {'PCN': 'mV/m', 'PCC': 'mV/m'}, 'time_col': 'epoch'}

    elif index in PC_STATIONS:

        # SuperMAG PolarCap - magnitude of horizontal field and y-component as "index"
        # need X and Z components for averaging
        df_index = import_processed_data('supermag', dtype=index, resolution='gsm')[['H_mag', 'H_x_GSE', 'H_y_GSE', 'H_z_GSE', 'H_y_GSM', 'H_z_GSM']]

    else:
        df_index = procedures[index]()
        df_index.attrs['units'][index] = 'nT'

    for sample_interval in sample_intervals:

        print(f'Writing {sample_interval} data.')

        output_file = os.path.join(get_proc_directory('indices', resolution=sample_interval, create=True), index)
        native_res  = native_resolutions.get(index,'1min')

        if native_res == sample_interval:

            if index in PC_STATIONS:
                df_clean = df_index.copy()
                df_clean.attrs = df_index.attrs
                clean_mag_data(df_clean, index)

                write_to_cdf(df_clean, output_file, reset_index=True)

            else:
                write_to_cdf(df_index, output_file, reset_index=True)

        elif resolutions.index(sample_interval) > resolutions.index(native_res):

            # resample to a lower resolution (i.e. average)

            df_sampled = resample_data(df_index, 'index', sample_interval.replace('hour','h'))

            # need to drop after resampling
            if index in PC_STATIONS:
                clean_mag_data(df_sampled, index)

            write_to_cdf(df_sampled, output_file, reset_index=True)

    print(f'{index} processed.')

def clean_mag_data(df, index):

    columns_to_drop = [col for col in df if col.startswith(('H_x','H_z'))]
    for col in ('H_GSE_count','H_GSM_count'):
        if col in df:
            columns_to_drop.append(col)

    df.drop(columns=columns_to_drop, inplace=True)
    df.attrs['units'] = {col: df.attrs.get('units',{}).get(col,'') for col in df} # removes extra columns

    columns_to_rename = {'H_mag': index} | {col: col.replace('H_',f'{index}_') for col in df if col!='H_mag'}
    rename_columns(df, columns_to_rename)


def import_processed_index(index, resolution='1min'):
    """
    Wrapper function to import a processed index as the structure depends on the index
    """
    native_res  = native_resolutions.get(index,'1min')
    if resolutions.index(resolution) < resolutions.index(native_res):
        warnings.warn(f'Resolution {resolution} is not available as the native resolution is {native_res}.')
        resolution = native_res

    index_parents = {'SYM_D': 'SYM', 'SYM_H': 'SYM', 'ASY_D': 'ASY', 'ASY_H': 'ASY', 'PCC': 'PC', 'PCN': 'PC', 'AL': 'AE', 'AU': 'AE'}
    index_parents |= {f'{station}_mag': station for station in PC_STATIONS} | {f'{station}_y_GSE': station for station in PC_STATIONS} | {f'{station}_y_GSM': station for station in PC_STATIONS}

    df = import_processed_data('indices', resolution=resolution, file_name=index_parents.get(index,index))

    return df.loc[:,index]

