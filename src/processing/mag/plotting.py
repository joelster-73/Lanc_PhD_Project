# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:42:09 2025

@author: richarj2
"""


import matplotlib
import numpy as np

import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt

from ...analysing.calculations import circular_mean

from ...plotting.config import blue, dark_mode, black
from ...coordinates.spatial import convert_GEO_position

# %% mag

def plot_mag_data(station, sw, pc, param='H', coords='aGSE', quantity='phi', ind='ER', show_sw=True):

    # add argument to account for shift but probably will be taken care off when using full data

    if param=='H':
        components = ('n','e')
    elif param=='B':
        components = ('n','e','z')

    station_name = station.attrs.get('id',param)

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

        ax0.plot(sw.index,sw[ind_col],lw=0.7,c='orange')

    alpha, colour_map = 0.2, 'tab10'
    if dark_mode:
        alpha, colour_map = 0.4, 'Pastel1'

    cmap = matplotlib.colormaps[colour_map]
    colours = cmap(np.arange(len(components)))

    for i, c in enumerate(components):
        ax.plot(station.index,station[f'B_{c}_NEZ'],lw=0.7,alpha=alpha,color=colours[i],label=f'{c.upper()}')

    column = f'{param}_y_{coords}'
    if quantity=='mag' or column not in station:
        colour = blue
        corr_quantity = station[f'{param}_mag'].corr(sw[ind_col])
        ax.plot(station.index, station[f'{param}_mag'], c=colour, lw=0.7, label=f'|{param}|')

    elif quantity=='phi':
        colour = 'magenta'
        corr_quantity = station[column].corr(sw[ind_col])
        label = r'$H_\phi$' if coords=='aGSE' else r'$H_0$'
        ax.plot(station.index, station[column], c=colour, lw=0.7, label=label)

    elif quantity=='tr':
        colour = 'green'
        station.loc[:,'H_T'] = (station.loc[:,f'{param}_x_{coords}']**2 + station.loc[:,f'{param}_y_{coords}']**2) ** 0.5
        column = 'H_T'
        corr_quantity = station[column].corr(sw[ind_col])
        label = r'$H_T$'
        ax.plot(station.index, station[column], c=colour, lw=0.7, label=label)

    ax.legend(loc='upper left')

    ax2.plot(pc.index,pc['PCN'],c='r',lw=0.7)
    ax2.set_xlabel(f'{sw.index[0].strftime("%b")} {sw.index.year[0]}')
    ax.set_ylabel(f'{station_name} [nT]')
    ax2.set_ylabel('PCN [mV/m]')

    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: int(mdates.num2date(x).day)))

    corr_pcn = pc['PCN'].corr(sw[ind_col])

    ax2.text(0.95, 0.95, f'r = {corr_pcn:.2f}', transform=ax2.transAxes, color='r', ha='right', va='top')
    ax.text(0.95, 0.95, f'r = {corr_quantity:.2f}', transform=ax.transAxes, color=colour, ha='right', va='top')

    if (station.index[-1] - station.index[0]).total_seconds()<86400:
        ax_MLT = ax0.twiny()
        ax_MLT.set_xlim(ax.get_xlim())
        ax_MLT.set_xlabel('MLT')

        top_times = station.index[::180]
        top_ticks = ax.convert_xunits(top_times)
        top_labels = station.loc[top_times, 'MLT'].round().astype(int)

        ax_MLT.set_xticks(top_ticks)
        ax_MLT.set_xticklabels(top_labels)

    plt.show()
    plt.close()



# %% map

def plot_magnetometer_map(df_field, coords='aGSE', param='H', df_sw=None, invert=True, show_dp2=False):

    colours = np.where(df_field['H_y_GSE'] > 0, 'red', 'orange')

    fig, ax = plt.subplots(dpi=300, figsize=(8,8))

    if df_sw is not None:
        # default position is Thule
        positions = convert_GEO_position(df_field.attrs.get('glat',77.46999), df_field.attrs.get('glon',290.76996), df_field.index, coords=coords, df_sw=df_sw)
        axis      = convert_GEO_position(90, 0, df_field.index, coords=coords, df_sw=df_sw)

        ax.scatter(positions[f'r_x_{coords}'], positions[f'r_y_{coords}'], marker='.', c='b')
        ax.scatter(axis[f'r_x_{coords}'], axis[f'r_y_{coords}'], marker='.', c='g')

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
    plt.show()
    plt.close()
