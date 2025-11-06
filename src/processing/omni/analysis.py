# -*- coding: utf-8 -*-
"""
Created on Mon May 19 23:02:24 2025

@author: joels
"""
import os
import glob
import numpy as np
import pandas as pd
from spacepy import pycdf

from .config import lagged_indices
from ..reading import import_processed_data
from ..writing import write_to_cdf
from ...config import PCN_DIR, AA_DIR, SME_DIR
from ...analysing.calculations import calc_angle_between_vecs
from ...coordinates.magnetic import calc_GSE_to_GSM_angles

# %% PCN

def process_PCN_data(pcn_dir):

    mjd2000_epoch = pd.Timestamp('2000-01-01')

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


def process_SME_data(sme_dir):

    file_path = os.path.join(sme_dir, 'SME_indices.txt')
    df_sme = pd.read_csv(file_path, skiprows=104, sep='\t')
    df_sme = df_sme.loc[df_sme['<year>']>2000]

    df_sme.rename(columns={'<year>': 'year', '<month>': 'month', '<day>': 'day', '<hour>': 'hour', '<min>': 'minute', '<SME (nT)>': 'SME'},inplace=True)

    df_sme['epoch'] = pd.to_datetime(df_sme[['year','month','day','hour','minute']])

    df_sme.set_index('epoch',inplace=True)
    df_sme.drop(columns=['year','month','day','hour','minute','<sec>'],inplace=True)

    df_sme.loc[df_sme['SME']>=3000] = np.nan

    return df_sme

def process_AA_data(aa_dir):

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


def add_index_lag(omni_dir, sample_interval, indices=lagged_indices):

    # lagged_indices = ('AE','PCN',...)
    # omni_lags = (10,17,20,30,...)

    df_pcn = process_PCN_data(PCN_DIR)
    df_aa  = process_AA_data(AA_DIR)
    df_sme = process_SME_data(SME_DIR)

    # Solar wind data
    df_sw    = import_processed_data(os.path.join(omni_dir, sample_interval))

    df_sw['AEc'] = correction_AE(df_sw) # using Weimer (1990) correction
    df_sw['SME'] = df_sme['SME'].reindex(df_sw.index)
    df_sw['PCN'] = df_pcn['PCN'].reindex(df_sw.index)
    df_sw['AA']  = df_aa['aa'].reindex(df_sw.index, method='ffill') # 3 hourly

    for ind, lags in indices.items():
        if ind not in df_sw:
            continue
        print(ind)

        for lag in lags:
            dt_lag = pd.Timedelta(minutes=lag)

            # Lagged index
            if (dt_lag % pd.Timedelta(sample_interval)) == pd.Timedelta(0):
                df_sw.insert(df_sw.columns.get_loc(ind) + 1, f'{ind}_{lag}m', df_sw[ind].shift(freq=dt_lag))
            else:
                print('Interpolating lag.')
                target_index = df_sw.index + dt_lag
                full_index = df_sw.index.union(target_index)
                temp = df_sw[ind].reindex(full_index).interpolate(method='time')
                df_sw.insert(df_sw.columns.get_loc(ind) + 1, f'{ind}_{lag}m', temp.loc[target_index].values)

    df_sw.loc[df_sw['M_A']>100,'M_A'] = np.nan

    df_sw['N_tot'] = df_sw['n_p'] * (1+df_sw['na_np_ratio'])
    df_sw.attrs['units']['N_tot'] = 'n/cc'

    # Theta Bn angle - quasi-perp/quasi-para
    df_sw['theta_Bn'] = calc_angle_between_vecs(df_sw, 'B_GSE', 'R_BSN')
    # restrict to be between 0 and 90 degrees
    df_sw.loc[df_sw['theta_Bn']>np.pi/2,'theta_Bn'] = np.pi - df_sw.loc[df_sw['theta_Bn']>np.pi/2,'theta_Bn']
    df_sw.attrs['units']['theta_Bn'] = 'rad'

    # GSE to GSM
    df_sw['gse_to_gsm_angle'] = calc_GSE_to_GSM_angles(df_sw, ref='B')

    # Writes OMNI with lag to file
    output_file = os.path.join(omni_dir, 'with_lag', f'omni_{sample_interval}.cdf')
    write_to_cdf(df_sw, output_file, reset_index=True)

