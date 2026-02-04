# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 12:41:52 2025

@author: richarj2
"""


import os
import glob
import numpy as np
import pandas as pd
from spacepy import pycdf

from .config import lagged_indices

from ..reading import import_processed_data
from ..writing import write_to_cdf
from ..omni.config import indices_columns

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


def process_SME_data():

    sme_dir = get_proc_directory('sme')
    file_path = os.path.join(sme_dir, 'SME_indices.txt')

    df_sme = pd.read_csv(file_path, skiprows=104, sep='\t')
    df_sme = df_sme.loc[df_sme['<year>']>2000]

    df_sme.rename(columns={'<year>': 'year', '<month>': 'month', '<day>': 'day', '<hour>': 'hour', '<min>': 'minute', '<SME (nT)>': 'SME'},inplace=True)

    df_sme['epoch'] = pd.to_datetime(df_sme[['year','month','day','hour','minute']])

    df_sme.set_index('epoch',inplace=True)
    df_sme.drop(columns=['year','month','day','hour','minute','<sec>'],inplace=True)

    df_sme.loc[df_sme['SME']>=3000] = np.nan

    return df_sme

def process_AA_data():

    aa_dir = get_proc_directory('aa')
    file_path = os.path.join(aa_dir, 'AA_indices.dat')

    df_aa = pd.read_csv(file_path, skiprows=34, sep=r'\s+', comment='|')
    df_aa['epoch'] = pd.to_datetime(df_aa['DATE'] + ' ' + df_aa['TIME'])

    df_aa.set_index('epoch',inplace=True)
    df_aa.drop(columns=['DATE','TIME','DOY','Kpa','Aa','CK24','CK48'],inplace=True)

    df_aa.loc[df_aa['aa']>=999] = np.nan

    return df_aa

def correction_AE(df):

    """
    Weimer et al. (1990) use a correction to AE data to make comparison over a year consistent

    AE_c = AE_m (1 + 0.5.sin^2((d-172).pi/365))
    AE_c = AE_m (1 + 0.5.sin^2((f-0.5).pi))
    where f is year fraction

    AE indices near the summer solstice are about 1.5 times the AE indices ohstained near the winder solstice (Northern hemisphere)
    """

    doy = df.index.dayofyear
    frac_day = (df.index.hour + df.index.minute/60) / 24.0

    year_frac = (doy + frac_day) / np.where(df.index.is_leap_year, 366, 365)

    return df['AE'] * (1 + 0.5*np.sin((year_frac-0.5)*np.pi)**2)

# %% OMNI_with_lag


def build_lagged_indices(sample_interval, indices=lagged_indices):

    # lagged_indices = ('AE','PCN',...)
    # omni_lags = (10,17,20,30,...)

    print('Importing...')
    df_pcn = process_PCN_data()
    df_pcc = process_PCC_data()
    df_aa  = process_AA_data()
    df_sme = process_SME_data()

    # SuperMAG PolarCap
    # Currently using magnitude of horizontal field and y-component as "index"
    # Will change eventually to field proected onto optimum direction
    df_THL = import_processed_data('supermag', dtype='THL', resolution='gsm')
    THL_columns = ['H_mag','H_y_GSE','H_y_GSM']
    THL_columns_new = ['SMC','SMC_y_GSE','SMC_y_GSM']

    # Extracts indices contained in OMNI - doesn't use any of the SW data
    df_pc  = import_processed_data('omni', resolution=sample_interval)
    df_pc = df_pc[indices_columns] # drops other columns

    #df_sw['AEc'] = correction_AE(df_sw) # using Weimer (1990) correction
    df_pc['SME'] = df_sme['SME'].reindex(df_pc.index)
    df_pc['PCN'] = df_pcn['PCN'].reindex(df_pc.index)
    df_pc['PCC'] = df_pcc['PCC'].reindex(df_pc.index)
    df_pc['AA']  = df_aa['aa'].reindex(df_pc.index, method='ffill') # 3 hourly
    df_pc[THL_columns_new] = df_THL[THL_columns].reindex(df_pc.index)

    print(df_pc.columns)

    print('Lagging...')

    for ind in df_pc.columns:
        lags = indices.get(ind,None)
        if lags is None:
            print(f'{ind} has no lag times.')
            continue
        print(ind,lags)

        for lag in lags:
            dt_lag = pd.Timedelta(minutes=lag)

            # Lagged index (estimated response from BSN to PC/AE)
            if (dt_lag % pd.Timedelta(sample_interval)) == pd.Timedelta(0):
                new_data = df_pc[ind].shift(freq=dt_lag)
            else:

                # Perhaps instead don't include?


                print('Interpolating lag.')
                target_index = df_pc.index + dt_lag
                full_index = df_pc.index.union(target_index)
                temp = df_pc[ind].reindex(full_index).interpolate(method='time')
                new_data = temp.loc[target_index].values

            df_pc.insert(df_pc.columns.get_loc(ind) + 1, f'{ind}_{lag}m', new_data)

    # Writes OMNI with lag to file
    output_file = os.path.join(get_proc_directory('indices'), f'combined_{sample_interval}')
    write_to_cdf(df_pc, output_file, reset_index=True)