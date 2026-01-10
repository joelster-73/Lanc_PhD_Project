# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:42:09 2025

@author: richarj2
"""


import matplotlib
import numpy as np
import itertools as it
import pandas as pd

import spaceweather as sw

import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt

from ...plotting.utils import save_figure
from ...plotting.formatting import custom_date_formatter, add_legend
from ...plotting.config import blue, dark_mode, black, white
from ...coordinates.spatial import convert_GEO_position

# %% mag

def plot_mag_data(station, sw, pc, param='H', ind='ER', show_sw=True, folder=None):

    # add argument to account for shift but probably will be taken care off when using full data

    if param=='H':
        components = ('n','e')
    elif param=='B':
        components = ('n','e','z')

    station_name = station.attrs.get('id',param)
    station = station.copy()

    if show_sw:
        fig, (ax0,ax,ax2) = plt.subplots(nrows=3, figsize=(10,8), dpi=300, sharex=True)
        ax0.axhline(0,c='k',ls=':')
    else:
        fig, (ax,ax2) = plt.subplots(nrows=2, figsize=(10,6), dpi=300, sharex=True)

    ax2.axhline(0,c='k',ls=':')
    ax.axhline(0,c='k',ls=':')

    plt.subplots_adjust(hspace=0)

    if show_sw:
        if ind=='Ey':
            ind_col = 'E_y_GSM'
            ax0.set_ylabel('$E_y$ [mV/m]')
        elif ind=='ER':
            ind_col = 'E_R'
            ax0.set_ylabel('$E_R$ [mV/m]')

        if ind_col not in sw:
            print(f'"{ind_col}" not in df.')
            ind_col = 'E_y'

        ax0.plot(sw.index,sw[ind_col],lw=0.7,c='orange')

    alpha, colour_map = 0.2, 'tab10'
    if dark_mode:
        alpha, colour_map = 0.4, 'Pastel1'

    cmap = matplotlib.colormaps[colour_map]
    colours = cmap(np.arange(len(components)))

    for i, c in enumerate(components):
        ax.plot(station.index, station[f'B_{c}_NEZ'], lw=0.7, alpha=alpha, color=colours[i], label=f'{c.upper()}')

    ax.plot(station.index, station[f'{param}_x_GSE'], c='blue', lw=0.7, label='X')
    ax.plot(station.index, station[f'{param}_y_GSE'], c='green', lw=0.7, label='Y')

    add_legend(fig, ax, rows=1)

    ax2.plot(pc.index,pc['PCN'],c='r',lw=0.7)
    ax2.set_xlabel(f'{sw.index[0].strftime("%b")} {sw.index.year[0]}')
    ax.set_ylabel(f'{station_name} [nT]')
    ax2.set_ylabel('PCN [mV/m]')

    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(mdates.num2date(x).day)))

    if (station.index[-1] - station.index[0]).total_seconds()<86400:
        ax_MLT = ax0.twiny()
        ax_MLT.set_xlim(ax.get_xlim())
        ax_MLT.set_xlabel('MLT')

        top_times = station.index[::180]
        top_ticks = ax.convert_xunits(top_times)
        top_labels = station.loc[top_times, 'MLT'].round().astype(int)

        ax_MLT.set_xticks(top_ticks)
        ax_MLT.set_xticklabels(top_labels)

    fig.align_ylabels()

    formatter = FuncFormatter(custom_date_formatter)
    ax2.xaxis.set_major_formatter(formatter)

    save_figure(fig, sub_directory=folder)
    plt.show()
    plt.close()

def plot_mag_data_corr(station, sw, pc, param='H', coords='GSE', quantity='mag', ind='ER', show_sw=True, phi=None, folder=None):

    # add argument to account for shift but probably will be taken care off when using full data

    if param=='H':
        components = ('n','e')
    elif param=='B':
        components = ('n','e','z')

    station_name = station.attrs.get('id',param)
    station = station.copy()

    if show_sw:
        fig, (ax0,ax,ax2) = plt.subplots(nrows=3, figsize=(10,8), dpi=300, sharex=True)
        ax0.axhline(0,c='k',ls=':')
    else:
        fig, (ax,ax2) = plt.subplots(nrows=2, figsize=(10,6), dpi=300, sharex=True)

    ax2.axhline(0,c='k',ls=':')
    ax.axhline(0,c='k',ls=':')

    plt.subplots_adjust(hspace=0)

    if show_sw:
        if ind=='Ey':
            ind_col = f'E_y_{coords}'
            ax0.set_ylabel('$E_y$ [mV/m]')
        elif ind=='ER':
            ind_col = 'E_R'
            ax0.set_ylabel('$E_R$ [mV/m]')

        if ind_col not in sw:
            print(f'"{ind_col}" not in df.')
            ind_col = 'E_y'

        ax0.plot(sw.index,sw[ind_col],lw=0.7,c='orange')

    alpha, colour_map = 0.2, 'tab10'
    if dark_mode:
        alpha, colour_map = 0.4, 'Pastel1'

    cmap = matplotlib.colormaps[colour_map]
    colours = cmap(np.arange(len(components)))

    for i, c in enumerate(components):
        ax.plot(station.index,station[f'B_{c}_NEZ'],lw=0.7,alpha=alpha,color=colours[i],label=f'{c.upper()}')

    if quantity=='phi':
        colour = 'magenta'

        if coords=='aGSE':
            column = f'{param}_y_{coords}'
            label  = r'$H_\phi$'

        elif phi:
            degs = int(np.degrees(phi))
            column = f'H_{degs}'
            label  = f'$H_{{{degs}^\\circ}}$'
            x_col  = f'{param}_x_{coords}'
            if coords=='GSM':
                x_col = f'{param}_x_GSE'
            station.loc[:, column] = (station.loc[:, x_col]*np.sin(phi) + station.loc[:, f'{param}_y_{coords}']*np.cos(phi)).astype(np.float32)

        else:
            column = f'{param}_y_{coords}'
            label = r'$H_y$'

    elif quantity=='tr':
        colour = 'green'
        column = 'H_T'
        label  = r'$H_T$'
        x_col  = f'{param}_x_{coords}'

        if coords=='GSM':
            x_col = f'{param}_x_GSE'

        station.loc[:, column] = ((station.loc[:, x_col]**2 + station.loc[:, f'{param}_y_{coords}']**2) ** 0.5).astype(np.float32)

    if quantity=='mag' or column not in station:
        colour = blue
        column = f'{param}_mag'
        label  = f'|{param}|'


    corr_quantity = station[column].corr(sw[ind_col])
    ax.plot(station.index, station[column], c=colour, lw=0.7, label=label)

    add_legend(fig, ax, rows=1)

    ax2.plot(pc.index,pc['PCN'],c='r',lw=0.7)
    ax2.set_xlabel(f'{sw.index[0].strftime("%b")} {sw.index.year[0]}')
    ax.set_ylabel(f'{station_name} [nT]')
    ax2.set_ylabel('PCN [mV/m]')

    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(mdates.num2date(x).day)))

    corr_pcn = pc['PCN'].corr(sw[ind_col])

    ax2.text(0.95, 0.95, f'r = {corr_pcn:.2f}', transform=ax2.transAxes, color='r', ha='right', va='top')
    ax.text(0.95, 0.95, f'r = {corr_quantity:.2f}', transform=ax.transAxes, color=colour, ha='right', va='top')
    t = ax0.text(0.1, 0.95, coords, transform=ax0.transAxes, color=black, ha='right', va='top')
    t.set_bbox(dict(facecolor=white))

    if (station.index[-1] - station.index[0]).total_seconds()<86400:
        ax_MLT = ax0.twiny()
        ax_MLT.set_xlim(ax.get_xlim())
        ax_MLT.set_xlabel('MLT')

        top_times = station.index[::180]
        top_ticks = ax.convert_xunits(top_times)
        top_labels = station.loc[top_times, 'MLT'].round().astype(int)

        ax_MLT.set_xticks(top_ticks)
        ax_MLT.set_xticklabels(top_labels)

    fig.align_ylabels()

    formatter = FuncFormatter(custom_date_formatter)
    ax2.xaxis.set_major_formatter(formatter)

    save_figure(fig, sub_directory=folder)
    plt.show()
    plt.close()


def plot_find_optimum(station, sw, param='H', coords='GSM', ind='Ey', folder=None):

    if ind=='Ey':
        ind_col = f'E_y_{coords}'
    elif ind=='ER':
        ind_col = 'E_R'

    if ind_col not in sw:
        print(f'"{ind_col}" not in df.')
        ind_col = 'E_y'

    x_col  = f'{param}_x_{coords}'
    if coords=='GSM':
        x_col = f'{param}_x_GSE'

    phi_range = np.arange(-np.pi, np.pi, np.pi/200)
    corrs     = np.empty(len(phi_range))
    rmse_norm = np.empty(len(phi_range))

    y = sw[ind_col].values

    for i, phi in enumerate(phi_range):
        projected = station.loc[:, x_col].values * np.sin(phi) + station.loc[:, f'{param}_y_{coords}'].values * np.cos(phi)

        mask = ~np.isnan(projected) & ~np.isnan(y)
        p = projected[mask]
        s = y[mask]

        # z-score both series
        pz = (p - p.mean()) / p.std()
        sz = (s - s.mean()) / s.std()

        corrs[i] = np.corrcoef(pz, sz)[0, 1]
        rmse_norm[i] = np.sqrt(np.mean((pz - sz) ** 2))

    fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True, dpi=300, figsize=(10,8))
    plt.subplots_adjust(hspace=0)

    ax1.plot(np.degrees(phi_range), corrs, c='b', lw=0.8)
    ax1.set_ylabel('Correlation')

    ax2.plot(np.degrees(phi_range), rmse_norm, c='r', lw=0.8)
    ax2.set_ylabel('RMSE (z-scored)')
    ax2.set_xlabel(r'Phi [$^\circ$]')

    maximum = np.argmax(corrs)
    best_phi = np.degrees(phi_range)[maximum]
    ax1.axvline(best_phi,c='k',ls=':',label=f'{int(best_phi)} $^\\circ$')
    ax2.axvline(best_phi,c='k',ls=':')
    add_legend(fig, ax1)

    save_figure(fig, sub_directory=folder)
    plt.show()
    plt.close()

    return phi_range[maximum]

# %% map

def plot_magnetometer_map(df_field, coords='GSE', param='H', invert=True, show_dp2=False, show_mag_pole=False, folder=None):

    colours = np.where(df_field['H_y_GSE'] > 0, 'red', 'orange')

    fig, ax = plt.subplots(dpi=300, figsize=(8,8))

    # default position is Thule
    positions = convert_GEO_position(df_field.attrs.get('glat',77.46999), df_field.attrs.get('glon',290.76996), df_field.index)
    axis      = convert_GEO_position(90, 0, df_field.index)

    ax.scatter(positions[f'r_x_{coords}'], positions[f'r_y_{coords}'], marker='.', c='b')
    ax.scatter(axis[f'r_x_{coords}'], axis[f'r_y_{coords}'], marker='.', c='g')

    # if show_mag_pole:
    #     df_mag    = get_magnetic_north_pole(df_field.index[0], df_field.index[-1])
    #     mag_axis  = convert_GEO_positions(df_mag)
    #     ax.scatter(mag_axis[f'r_x_{coords}'], mag_axis[f'r_y_{coords}'], marker='.', c='purple')

    if show_dp2:
        print('Not implemented - need to make more accurate')


    if invert:
        ax.quiver(positions[f'r_x_{coords}'], positions[f'r_y_{coords}'], -df_field[f'{param}_x_{coords}'], -df_field[f'{param}_y_{coords}'], color=colours)
        ax.invert_xaxis()
        ax.invert_yaxis()
    else:
        ax.quiver(positions[f'r_x_{coords}'], positions[f'r_y_{coords}'], df_field[f'{param}_x_{coords}'], df_field[f'{param}_y_{coords}'], color=colours)

    ax.set_xlabel(f'X {coords} [$R_E$]')
    ax.set_ylabel(f'Y {coords} [$R_E$]')

    ax.set_title(f'Magnetometer location in {coords}')
    plt.axis('equal')

    plt.grid(True)
    save_figure(fig, sub_directory=folder)
    plt.show()
    plt.close()

# %% overview

def plot_magnetometer_overview(df_station, df_omni, df_pc, start_time, end_time=None):

    if not end_time:
        end_time = start_time + pd.Timedelta(days=1)

    Ap = daily_ap(str(start_time))
    folder = f'Ap_{Ap}_{start_time.strftime("%Y%m%d")}'

    station_sub  = df_station.loc[(df_station.index>=start_time)&(df_station.index<end_time)]
    omni_sub     = df_omni.loc[(df_omni.index>=start_time)&(df_omni.index<end_time)]
    indices_sub  = df_pc.loc[(df_pc.index>=start_time)&(df_pc.index<end_time)]


    plot_mag_data(station_sub, omni_sub, indices_sub, ind='Ey', folder=folder)
    phi = plot_find_optimum(station_sub, omni_sub, folder=folder)

    plot_combos = it.product(('ER','Ey'), ('GSE','GSM'), ('mag','phi','tr'))

    for (ind, coords, quantity) in plot_combos:

        if (ind, coords, quantity) == ('ER', 'GSM', 'mag'):
            continue

        plot_mag_data_corr(station_sub, omni_sub, indices_sub, coords=coords, quantity=quantity, ind=ind, phi=phi, folder=folder)

    plot_magnetometer_map(station_sub, coords='GSE', folder=folder)

def daily_ap(date_str):
    """
    Returns daily Ap value for YYYY-MM-DD
    """
    df = sw.sw_daily(update=True)
    if date_str not in df.index:
        raise ValueError(f'No Ap data for {date_str}')

    for col in df.columns:
        if col.lower() == 'apavg':
            return int(df.loc[date_str, col])

    raise RuntimeError('Ap column not found')

