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

from ..reading import import_processed_data
from ..writing import write_to_cdf
from ...config import PCN_DIR
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

# %% OMNI_with_lag


def add_index_lag(omni_dir, sample_interval, indices=('AE','PCN'), lags=(10,17,20,30)):

    df_pcn = process_PCN_data(PCN_DIR)

    # Solar wind data
    omni_dir = os.path.join(omni_dir, sample_interval)
    df_sw    = import_processed_data(omni_dir)
    df_sw['PCN'] = df_pcn['PCN'].reindex(df_sw.index)

    for lag in lags:
        dt_lag = pd.Timedelta(minutes=lag)

        # Lagged index
        if (dt_lag % pd.Timedelta(sample_interval)) == pd.Timedelta(0):

            for ind in indices:
                df_sw[f'{ind}_{lag}m'] = df_sw[ind].shift(freq=dt_lag)

        else:
            print('Interpolating lag.')

            target_index = df_sw.index + dt_lag
            full_index = df_sw.index.union(target_index)
            for ind in indices:
                temp = df_sw[ind].reindex(full_index)
                temp = temp.interpolate(method='time')
                df_sw[f'{ind}_{lag}m'] = temp.loc[target_index].values

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

