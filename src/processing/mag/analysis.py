# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:42:09 2025

@author: richarj2
"""

from src.processing.mag.supermag import process_supermag_data

station = 'THL'
THL = process_supermag_data(station)

# %%
from matplotlib import pyplot as plt
import numpy as np

from src.processing.reading import import_processed_data

from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates

omni = import_processed_data('omni', resolution='1min')
indices  = import_processed_data('indices', file_name='combined_1min')

thl_year  = THL.loc[(THL.index.year==2015)]
omni_year = omni.loc[(omni.index.year==2015)]
indices_year = indices.loc[(indices.index.year==2015)]


# %%
from src.plotting.config import blue, dark_mode
import matplotlib

def plot_mag_data(station, sw, pc, components=('n','e','z'), show_sw=True, station_name='B'):

    # add argument to account for shift but probably will be taken care off when using full data

    if show_sw:
        fig, (ax0,ax,ax2) = plt.subplots(nrows=3, figsize=(10,8), dpi=300, sharex=True, gridspec_kw={'height_ratios': [1, 1.5, 1]})
    else:
        fig, (ax,ax2) = plt.subplots(nrows=2, figsize=(10,6), dpi=300, sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})

    plt.subplots_adjust(hspace=0)

    if show_sw:
        ax0.plot(sw.index,sw['E_y_GSM'],c='orange',label='Ey')
        ax0.set_ylabel('Ey [mV/m]')

    alpha = 0.2
    colour_map = 'tab10'
    if dark_mode:
        colour_map = 'Pastel1'
        alpha = 0.4

    cmap = matplotlib.colormaps[colour_map]
    colours = cmap(np.arange(len(components)))

    for i, c in enumerate(components):
        ax.plot(station.index,station[f'B_{c}_NEZ'],lw=0.7,alpha=alpha,color=colours[i],label=f'{c.upper()}')

    ax.legend(loc='upper left')
    ax.plot(station.index,np.linalg.norm(station[[f'B_{c}_NEZ' for c in components]],axis=1),c=blue,lw=0.7,label='|B|')
    ax2.plot(pc.index,pc['PCN'],c='r',lw=0.7)

    ax2.set_xlabel(f'{sw.index[0].strftime("%b")} {sw.index.year[0]}')
    ax.set_ylabel(f'{station_name} [nT]')
    ax2.set_ylabel('PCN [mV/m]')
    ax2.axhline(0,c='k',ls=':')

    ax2.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: int(mdates.num2date(x).day))
    )

    plt.show()


# %% max_PCN

thl_mar  = thl_year.loc[(thl_year.index.month==3)&(thl_year.index.day>=14)&(thl_year.index.day<=20)]
omni_mar = omni_year.loc[(omni_year.index.month==3)&(omni_year.index.day>=14)&(omni_year.index.day<=20)]
indices_mar = indices_year.loc[(indices_year.index.month==3)&(indices_year.index.day>=14)&(indices_year.index.day<=20)]


plot_mag_data(thl_mar, omni_mar, indices_mar, components=('n','e','z'), station_name=station)
plot_mag_data(thl_mar, omni_mar, indices_mar, components=('n','e'), station_name=station)



# %% min_PCN

minimum = indices_year['PCN'].min()
min_time = indices_year['PCN'].idxmin()
print(min_time,minimum)

thl_nov  = thl_year.loc[(thl_year.index.month==11)&(thl_year.index.day<=10)]
omni_nov = omni_year.loc[(omni_year.index.month==11)&(omni_year.index.day<=10)]
indices_nov = indices_year.loc[(indices_year.index.month==11)&(indices_year.index.day<=10)]



plot_mag_data(thl_nov, omni_nov, indices_nov, components=('n','e','z'), station_name=station)
plot_mag_data(thl_nov, omni_nov, indices_nov, components=('n','e'), station_name=station)


