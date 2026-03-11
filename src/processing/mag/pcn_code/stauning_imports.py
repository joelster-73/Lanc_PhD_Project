# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:44:21 2026

@author: richarj2
"""

import os
import numpy as np
import pandas as pd
import calendar

import glob
from spacepy import pycdf
import scipy.io

from config import DIRECTORIES

def import_data(coeff, var, source='original'):

    units_dict = {'a': 'mV/m/nT', 'b': 'nT', 'phi': 'deg', 'H_x': 'nT', 'H_y': 'nT', 'Hproj': 'nT', 'E_r': 'mV/m'}

    import_functions = {'phi': import_phi, 'ab': import_ab, 'coeff': import_coeff, 'pcn': import_pcn, 'hproj': import_hproj}

    import_function = import_functions.get(coeff)
    data_dict = import_function(var, source)

    for key, val in data_dict.items():
        data_dict[key] = val.ravel()

    if 'time' in data_dict:
        timestamps = data_dict['time'].copy()
        del data_dict['time']
    else:
        timestamps = get_timestamps(data_dict[list(data_dict.keys())[0]], var)

    attrs_dict = {'units': {key: units_dict.get(key) for key in data_dict}}

    df = pd.DataFrame(data_dict, index=timestamps)
    df = df.loc[:, (df != 0).any(axis=0)]
    df.attrs = attrs_dict

    return df

def import_pcn(year, source):

    # output path
    if source == 'original':
        path = DIRECTORIES.get('coeff')
    else:
        path = DIRECTORIES.get(source)

    pcn_dir = os.path.join(path, 'pcn')

    pcn = np.load(os.path.join(pcn_dir, f'pc_{year}.npz'))

    data_dict = {'time': pcn['time']}

    for key in ('pcn', 'pcn_unc', 'pc'):
        if key in pcn:
            data_dict[key] = pcn[key]

    return data_dict

def get_timestamps(arr, var=None):
    n = len(arr.ravel())

    if isinstance(var, int) or var.isdigit():
        num_days = 366 if calendar.isleap(int(var)) else 365
        start = pd.Timestamp(f'{var}-01-01')

        if n % 1440 == 0 and n // 1440 == num_days:
            times = pd.date_range(start=start, periods=n, freq='1min')
        elif n % 288 == 0 and n // 288 == num_days:
            times = pd.date_range(start=start, periods=n, freq='5min')
        else:
            raise ValueError(f"Array length {n} doesn't match year {var} ({num_days} days)")
        return times

    else:
        if n % 1440 == 0 and n // 1440 in (365, 366):
            days = range(1, n // 1440 + 1)
            times = pd.date_range(start='00:00', periods=1440, freq='1min').time
            return pd.MultiIndex.from_product([days, times], names=['doy', 'time'])

        elif n % 288 == 0 and n // 288 in (365, 366):
            days = range(1, n // 288 + 1)
            times = pd.date_range(start='00:00', periods=288, freq='5min').time
            return pd.MultiIndex.from_product([days, times], names=['doy', 'time'])

        elif n == 12 * 1440:
            times = pd.date_range(start='00:00', periods=1440, freq='1min').time
            return pd.MultiIndex.from_product([range(1, 13), times], names=['month', 'time'])

        elif n == 12 * 288:
            times = pd.date_range(start='00:00', periods=288, freq='5min').time
            return pd.MultiIndex.from_product([range(1, 13), times], names=['month', 'time'])

        else:
            raise ValueError(f'Cannot infer timestamps from array length {n}')


def import_true_pcn():

    pcn_dir = 'Y:/Processed_Data/INDICES/PCN'

    pattern = os.path.join(pcn_dir, '*.cdf')
    files = sorted(glob.glob(pattern))

    yearly_pcn = []

    for file in files:
        with pycdf.CDF(file) as cdf:
            df_year = pd.DataFrame({'time': cdf['Time'][...],'pcn': cdf['PCN'][...]})
            yearly_pcn.append(df_year)

    df_pcn = pd.concat(yearly_pcn, ignore_index=True)
    mjd2000_epoch = pd.Timestamp('2000-01-01')

    df_pcn['epoch'] = mjd2000_epoch + pd.to_timedelta(df_pcn['time'], unit='D')
    df_pcn['epoch'] = df_pcn['epoch'].dt.round('min')
    df_pcn.drop(columns=['time'],inplace=True)
    df_pcn.set_index('epoch',inplace=True)

    df_pcn.loc[df_pcn['pcn']>=999] = np.nan

    return df_pcn
# %% Documentation

def import_er(year, source):

    if source in ('staun_proj', 'staun_phi', 'recreated_phi'):

        ekl = scipy.io.loadmat(os.path.join(DIRECTORIES.get('data'), f'ekls_{year}.mat'))[f'ekls_{year}']

        yeartime   = scipy.io.loadmat(os.path.join(DIRECTORIES.get('data'), 'yeartime.mat'))['yeartime']
        timestamps = yeartime.T # transpose to match MATLAB's yeartime'

    elif source=='updated_phi':
        raise ValueError('Not implemented.')

    return {'E_R': ekl, 't': timestamps}

def import_hproj(year, source):


    if source=='staun_proj':
        in_dir = DIRECTORIES.get('data')
        hproj_data = scipy.io.loadmat(os.path.join(in_dir, f'Hproj_{year}.mat'))

    elif source in ('staun_phi', 'recreated_phi'):
        in_dir = DIRECTORIES.get(source)
        hproj_data = np.load(os.path.join(in_dir, 'hprojs', f'Hproj_{year}.npz'))

    elif source=='updated_phi':
        raise ValueError('Not implemented.')

    H_proj = {'hproj': hproj_data[f'Hproj_{year}'].flatten()}
    if f'Hproj_{year}_var' in hproj_data:
        H_proj['var'] = hproj_data[f'Hproj_{year}_var']

    return H_proj

def import_phi(var, source='original'):
    """
    Imports direction angles 'phi'
    If source = original, loads in the matlab file, else laods in the pickle
    Same file to import yearly file or averaged 2d/year file
    """

    if source in ('original','staun_proj'):
        if isinstance(var,int) or var.isdigit():
            path = os.path.join(DIRECTORIES.get('phi'), 'phis', f'F_{var}.mat')
            var = f'F_{var}'
        else:
            path = os.path.join(DIRECTORIES.get('phi'), f'Fi_{var}.mat')
            var = f'Fi_{var}'

        mat = scipy.io.loadmat(path)
        struc = {'phi': mat[var]}

    else:

        in_dir = DIRECTORIES.get(source)

        if isinstance(var,int) or var.isdigit():
            path = os.path.join(in_dir, 'phis', f'Phi_{var}.npz')
        else:
            path = os.path.join(in_dir, f'Phi_{var}.npz')

        data  = np.load(path)
        struc = {'phi': data['Phi']}
        if 'Phi_var' in data:
            struc['phi_var'] = data['Phi_var']

    return struc


def import_ab(var, source='original'):
    """
    Imports regression coefficients a/b
    If source = original, loads in the matlab file, else laods in the pickle
    Same file to import yearly file or averaged 2d/year file
    """

    if source == 'original':
        if isinstance(var,int) or var.isdigit():
            path = os.path.join(DIRECTORIES.get('ab'), 'abs', f'ab_{var}.mat')
        else:
            path = os.path.join(DIRECTORIES.get('ab'), f'ab_{var}.mat')

        mat = scipy.io.loadmat(path)
        var = f'ab_{var}'

        struc = mat[var]
        struc = {'a': struc[0][0][0].squeeze(), 'b': struc[0][0][1].squeeze()}

    else:
        if isinstance(var,int) or var.isdigit():
            path = os.path.join(DIRECTORIES.get(source), 'abs', f'ab_{var}.npz')
        else:
            path = os.path.join(DIRECTORIES.get(source), f'ab_{var}.npz')

        data = np.load(path)

        struc = {'a': data['a'], 'b': data['b']}
        for key in ('a_var','b_var','covar'):
            if key in data:
                struc[key] = data[key]

    return struc

def import_coeff(var=None, source='original'):
    """
    Imports average coefficients phi/a/b
    If source = original, loads in the matlab file, else laods in the pickle
    Same file to import yearly file or averaged year file
    """

    if source == 'original':
        if var is None or var=='':
            file_name = 'coeff.mat'
        elif var=='2d':
            file_name = 'coeff_2d_THL.mat'
        elif var=='year':
            file_name = 'coeff_THL.mat'

        path = os.path.join(DIRECTORIES.get('coeff'), file_name)
        mat = scipy.io.loadmat(path)

        if var=='2d':
            struc = {'phi': mat['f'].squeeze(), 'a': mat['a'].squeeze(), 'b': mat['b'].squeeze()}

        else:
            struc = mat['coeff']
            struc = {'phi': struc[0][0][0].squeeze(), 'a': struc[0][0][1].squeeze(), 'b': struc[0][0][2].squeeze()}

    else:
        path = os.path.join(DIRECTORIES.get(source), 'coeff.npz')
        data = np.load(path)


        struc = {'phi': data['phi'].flatten(), 'a': data['a'], 'b': data['b']}
        for key in ('phi_var','a_var','b_var','covar'):
            if key in data:
                struc[key] = data[key]

    return struc