# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 10:50:18 2025

@author: richarj2
"""

# Because finding the orbital configurations of the spacecraft is more difficult than it should be


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..reading import import_processed_data
from ...plotting.config import colour_dict

cmap = ('#FF0000','#00FF00','#0000FF','#FFFF00','#FF00FF','#00FFFF','#000000','#FFFFFF','#FFA500','#800080')


def plot_themis_position(proc_directories, probes=('tha','thb','thc','thd','the'), year=None, y_axis='r_x', **kwargs):

    show_apogees = kwargs.get('show_apogees',True)
    scheme       = kwargs.get('scheme','distinct')
    window_size  = kwargs.get('window_size',30) # days

    fig          = kwargs.get('fig',None)
    ax           = kwargs.get('ax',None)
    return_objs  = kwargs.get('return_objs',False)

    if fig is None or ax is None:
        fig, ax = plt.subplots(dpi=400)

    window_size *= 24 * 60 # minutes

    for i, spacecraft in enumerate(('tha','thb','thc','thd','the')):

        if spacecraft not in probes:
            continue

        spacecraft_dir = proc_directories[spacecraft]
        pos_dir        = os.path.join(spacecraft_dir, 'STATE', '1min')

        if isinstance(year,tuple):
            pos_df = import_processed_data(pos_dir)
            pos_df = pos_df.loc[pos_df.index.year.isin(year)]
        else:
            pos_df = import_processed_data(pos_dir, year=year)

        if y_axis=='r_mag':

            r_mag = np.linalg.norm(pos_df[[f'r_{comp}_GSE' for comp in ('x','y','z')]].to_numpy(), axis=1)
            positions = pd.Series(r_mag, index=pos_df.index)

            y_label = r'Radial Distance [$\mathrm{R_E}$]'

        elif y_axis=='r_x':

            positions = pos_df['r_x_GSE']

            y_label = r'X [$\mathrm{R_E}$]'

        if positions.empty:
            print(f'{spacecraft} in {year} is empty.')
            print(positions)
            continue

        elif show_apogees:

            if scheme=='green':
                colour = colour_dict.get(spacecraft.upper(),'red')
            else:
                colour = cmap[i]

            # Compute the rolling maximum
            apogees = positions.rolling(window=window_size, center=True).max()
            ax.plot(apogees, color=colour, lw=2.25)

            if y_axis=='r_mag':
                perigees = positions.rolling(window=window_size, center=True).min()
                ax.plot(perigees, color=colour, lw=2.25)

        else:
            if scheme=='green':
                colour = colour_dict.get(spacecraft.upper(),'grey')
            else:
                colour = cmap[i]

            ax.plot(positions, color=colour, lw=0.5)

        ax.plot([], [], '-', color=colour, lw=2.25, label=spacecraft)

    if y_axis=='r_x':
        ax.set_ylim(0)

    ax.set_xlabel('Time')
    ax.set_ylabel(y_label)
    ax.legend(loc='upper left')
    #add_figure_title(fig, 'Cluster\'s orbit showing Apogee and Perigee')

    if return_objs:
        return fig, ax

    plt.tight_layout();
    #save_figure(fig)
    plt.show()
    plt.close()

