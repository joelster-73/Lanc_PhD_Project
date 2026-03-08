# -*- coding: utf-8 -*-
'''
Created on Sat Feb 28 14:22:13 2026

@author: richarj2
'''

import os
import numpy as np
import pandas as pd

import scipy.io
import matplotlib.pyplot as plt


from config import DATA_DIR, FI_DIR, AB_DIR, COEFF_DIR, OUT_DIR, PLOT_LABELS_SHORT, PLOT_LABELS_LONG, LIST_OF_MONTHS

def import_data(var, source='mat', year=None):

    if source=='mat':

        return import_mat_data(var, year)

    elif source=='npz':

        return import_npz_data(var, year)

def import_mat_data(var, year=None):

    if var=='dist':
        # Every 5 minutes of the year (366 x 288)
        dist = scipy.io.loadmat(os.path.join(DATA_DIR, f'dist_{year}.mat'))[f'dist_{year}']

        dist_x = dist['x'][0,0].ravel()[:105120]
        dist_y = dist['y'][0,0].ravel()[:105120]

        timestamps = pd.date_range(start=f'{year}-01-01', periods=len(dist_x), freq='5min')

        data_dict = {'H_x': dist_x, 'H_y': dist_y}
        attr_dict = {'units': {'H_x': 'nT', 'H_y': 'nT'}}

    elif var=='ekl':
        # Every 5 minutes of the year (365 x 288)
        ekl = scipy.io.loadmat(os.path.join(DATA_DIR, f'ekls_{year}.mat'))[f'ekls_{year}']

        timestamps = pd.date_range(start=f'{year}-01-01', periods=len(ekl[0]), freq='5min')

        data_dict = {'E_R': ekl[0]}
        attr_dict = {'units': {'E_R': 'mV/m'}}

    elif var=='proj':

        hproj = scipy.io.loadmat(os.path.join(DATA_DIR,f'Hproj_{year}.mat'))[f'Hproj_{year}']

        timestamps = pd.date_range(start=f'{year}-01-01', periods=len(hproj.flatten()), freq='5min')

        data_dict = {'H_phi': hproj.flatten()}
        attr_dict = {'units': {'H_phi': 'nT'}}

    elif var=='coeff':
        # Every 1 minute of a year (366 x 1440)
        coeff = scipy.io.loadmat(os.path.join(COEFF_DIR, 'coeff.mat'))['coeff']

        f, a, b = coeff[0][0]

        if year == 'year':
            days = range(1, len(a.ravel()) // 1440 + 1)
            times = pd.date_range(start='00:00', periods=1440, freq='1min').time
            timestamps = pd.MultiIndex.from_product([days, times], names=['doy', 'time'])
        else:
            periods = len(f[0])
            times = pd.date_range(start='00:00', periods=periods, freq='1min').time
            months = range(1, 13)
            timestamps = pd.MultiIndex.from_product([months, times], names=['month', 'time'])

        data_dict = {'phi': f.ravel(), 'a': a.ravel(), 'b': b.ravel()}
        attr_dict = {'units': {'phi': 'deg', 'a': 'mV/m/nT', 'b': 'nT'}}

    elif var=='ab':
        # Every 5 minutes of a month's day (12 x 288)
        # except 'year' which is repeated for every day (int year is also 2d)
        if year in ('2d','year') or isinstance(year,int) or year.isdigit():
            ab = scipy.io.loadmat(os.path.join(AB_DIR, f'ab_{year}.mat'))[f'ab_{year}']
        else:
            raise ValueError(f'"{var} {year}" not valid.')

        a, b = ab[0][0]

        if year == 'year':
            days = range(1, len(a.ravel()) // 288 + 1)
            times = pd.date_range(start='00:00', periods=288, freq='5min').time
            timestamps = pd.MultiIndex.from_product([days, times], names=['doy', 'time'])
        else:
            periods = len(a[0])
            times = pd.date_range(start='00:00', periods=periods, freq='5min').time
            months = range(1, 13)
            timestamps = pd.MultiIndex.from_product([months, times], names=['month', 'time'])

        data_dict = {'a': a.ravel(), 'b': b.ravel()}
        attr_dict = {'units': {'a': 'mV/m/nT', 'b': 'nT'}}

    elif var=='phi':
        # Every 5 minutes of a month's day (12 x 288)
        # except 'year' which is repeated for every day (int year is also 2d)
        if year in ('2d','year'):
            phi = scipy.io.loadmat(os.path.join(FI_DIR, f'Fi_{year}.mat'))[f'Fi_{year}']
        elif isinstance(year,int) or year.isdigit():
            phi = scipy.io.loadmat(os.path.join(FI_DIR, f'F_{year}.mat'))[f'F_{year}']
        else:
            raise ValueError(f'"{var} {year}" not valid.')

        if year == 'year':
            days = range(1, len(phi.ravel()) // 288 + 1)
            times = pd.date_range(start='00:00', periods=288, freq='5min').time
            timestamps = pd.MultiIndex.from_product([days, times], names=['day_of_year', 'time'])
        else:
            periods = len(phi[0])
            times = pd.date_range(start='00:00', periods=periods, freq='5min').time
            months = range(1, 13)
            timestamps = pd.MultiIndex.from_product([months, times], names=['month', 'time'])

        data_dict = {'phi': phi.ravel()}
        attr_dict = {'units': {'phi': 'deg'}}

    df = pd.DataFrame(data_dict, index=timestamps)
    df.attrs = attr_dict
    return df

def import_npz_data(var, year=None):

    if var=='dist':

        raise ValueError(f'"{var}" not implemented.')

    elif var=='ekl':

        raise ValueError(f'"{var}" not implemented.')

    elif var=='proj':

        raise ValueError(f'"{var}" not implemented.')

    elif var=='coeff':

        raise ValueError(f'"{var}" not implemented.')

    elif var=='ab':
        # Every 5 minutes of a month's day (12 x 288)
        # except 'year' which is repeated for every day (int year is also 2d)
        if year in ('2d','year') or isinstance(year,int) or year.isdigit():
            ab = np.load(os.path.join(OUT_DIR, f'ab_{year}.npz'))[f'ab_{year}']
        else:
            raise ValueError(f'"{var} {year}" not valid.')

        a, b = ab[0][0]

        if year == 'year':
            days = range(1, len(a.ravel()) // 288 + 1)
            times = pd.date_range(start='00:00', periods=288, freq='5min').time
            timestamps = pd.MultiIndex.from_product([days, times], names=['doy', 'time'])
        else:
            periods = len(a[0])
            times = pd.date_range(start='00:00', periods=periods, freq='5min').time
            months = range(1, 13)
            timestamps = pd.MultiIndex.from_product([months, times], names=['month', 'time'])

        data_dict = {'a': a.ravel(), 'b': b.ravel()}
        attr_dict = {'units': {'a': 'mV/m/nT', 'b': 'nT'}}

    elif var=='phi':
        # Every 5 minutes of a month's day (12 x 288)
        # except 'year' which is repeated for every day (int year is also 2d)
        if year in ('2d','year') or isinstance(year,int) or year.isdigit():
            key = 'Phi'
            phi = np.load(os.path.join(OUT_DIR, f'Phi_{year}.npz'))[key]
        else:
            raise ValueError(f'"{var} {year}" not valid.')

        if year == 'year':
            days = range(1, len(phi.ravel()) // 288 + 1)
            times = pd.date_range(start='00:00', periods=288, freq='5min').time
            timestamps = pd.MultiIndex.from_product([days, times], names=['day_of_year', 'time'])
        else:
            periods = len(phi[0])
            times = pd.date_range(start='00:00', periods=periods, freq='5min').time
            months = range(1, 13)
            timestamps = pd.MultiIndex.from_product([months, times], names=['month', 'time'])

        data_dict = {'phi': phi.ravel()}
        attr_dict = {'units': {'phi': 'deg'}}

    df = pd.DataFrame(data_dict, index=timestamps)
    df.attrs = attr_dict
    return df


def convert_array_to_df(struc):

    if struc.shape[1]==1:
        struc = struc.reshape(-1,288)
    if struc.shape[0] in (12,365,366):
        struc = struc.T

    if struc.shape[1]>=365: # column per day
        phi_df = pd.DataFrame(struc, columns=list(range(1,struc.shape[1]+1)))
    elif struc.shape[1]==12:
        phi_df = pd.DataFrame(struc, columns=LIST_OF_MONTHS)

    if len(phi_df)>24:
        phi_df.index = phi_df.index / (len(phi_df)//24)

    phi_df.index = phi_df.index.rename('UT')

    return phi_df


def print_coeffs_monthly_ut(df):

    print(df)

    if len(df)!=24:

        phi_hours = np.zeros((24,12))
        # for i in range(24):
        #     avg = np.average(struc[:,12*i:12*(i+1)], axis=1)
        #     phi_hours[i] = avg

    phi_df = pd.DataFrame(phi_hours, columns=LIST_OF_MONTHS)
    phi_df.index = phi_df.index.rename('UT')

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(phi_df)



def coeff_plot(source='npz', coeff_data=None):
    """
    Load Phi data (.npz or .mat) and generate a contour plot.
    var: 'Phi_2d' or 'Phi_year'
    """
    if coeff_data is None:
        coeff_data = coeff_import(source=source)


    nrows = len(coeff_data.columns)

    fig, axs = plt.subplots(nrows=nrows, figsize=(10,6), sharex=True, sharey=True, dpi=300)

    if nrows>1:

        for r, ax in enumerate(axs):

            coeff  = list(coeff_data.columns)[r]
            data   = coeff_data[coeff].to_numpy()

            n_days = len(data) // 1440
            data = data[:n_days * 1440]
            daily = data.reshape(n_days, 1440)  # (n_days, 1440)

            ncontours = 20
            cb = ax.contourf(np.arange(1440), np.arange(n_days), daily, ncontours, cmap='turbo')
            _ = ax.contour(np.arange(1440), np.arange(n_days), daily, ncontours, colors='black', linewidths=0.5)

            ax.set_xticks(np.arange(0, 1440+1, 60))
            ax.set_xticklabels(np.arange(0, 24+1))

            ax.set_yticks(np.arange(0, n_days+1, 90))
            ax.set_yticklabels(np.arange(0, 12+1, 3))

            cbar = plt.colorbar(cb)
            cbar.ax.set_ylabel(PLOT_LABELS_SHORT[coeff])
            cbar.ax.yaxis.set_ticks_position('left')

            ax.set_ylabel('Month')

        axs[-1].set_xlabel('UT hour')

   # plt.title(f'{source.upper()}')

    plt.tight_layout()
    plt.show()


def coeff_plot_single(source='npz', coeff_data=None, coeff='f'):
    """
    Load Phi data (.npz or .mat) and generate a contour plot.
    var: 'Phi_2d' or 'Phi_year'
    """
    if coeff_data is None:
        coeff_data = coeff_import(source=source)


    fig, ax = plt.subplots(figsize=(10,6), sharex=True, sharey=True, dpi=300)


    data   = coeff_data[coeff].to_numpy()

    n_days = len(data) // 1440
    data = data[:n_days * 1440]
    daily = data.reshape(n_days, 1440)  # (n_days, 1440)

    ncontours = 20
    cb = ax.contourf(np.arange(1440), np.arange(n_days), daily, ncontours, cmap='turbo')
    _ = ax.contour(np.arange(1440), np.arange(n_days), daily, ncontours, colors='black', linewidths=0.5)

    ax.set_xticks(np.arange(0, 1440+1, 60))
    ax.set_xticklabels(np.arange(0, 24+1))

    ax.set_yticks(np.arange(0, n_days+1, 90))
    ax.set_yticklabels(np.arange(0, 12+1, 3))

    _ = plt.colorbar(cb)

    ax.set_ylabel('Month')
    ax.set_xlabel('UT hour')
    ax.set_title(PLOT_LABELS_LONG[coeff])


    plt.tight_layout()
    plt.show()

# %%


test = import_data('phi', source='npz', year='2d')
print(test.attrs)
test
# %%
if __name__ == '__main__':



    ### create general functions for plotting and importing



    # go through and check all files
    # then print all raw matlab files and compare to documentation
    # then reproduce with python code


    coeffs = coeff_import(source='mat')
    a_vals = convert_array_to_df(coeffs[['a']].values)
    b_vals = convert_array_to_df(coeffs[['b']].values)

    coeff_plot('mat', coeffs)

    coeff_plot_single('mat', coeffs, 'f')

    """
    'year' should be used for plotting/calcs
    '2d' should be used for checking values
    """

    phis = coeff_import(source='mat', var='2d')
    phis = coeff_import(source='mat', var='year')

    print_coeffs_monthly_ut(phis)
    print_coeffs_monthly_ut(a_vals)

    _ = coeff_plot_single(var='year', source='mat', phi_data=phis)
    _ = coeff_plot_single(var='2d', source='mat', phi_data=phis)
    #_ = phi_plot(var='2d', source='npz')
