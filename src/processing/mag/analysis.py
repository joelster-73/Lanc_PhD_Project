# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:42:09 2025

@author: richarj2
"""

from src.processing.mag.supermag import process_supermag_data


THL = process_supermag_data('THL')

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

def plot_mag_data(station, sw, pc, components=('n','e','z'), show_sw=True):

    # add argument to account for shift but probably will be taken care off when using full data

    if show_sw:
        fig, (ax0,ax,ax2) = plt.subplots(nrows=3,figsize=(8,6),dpi=300,sharex=True)
    else:
        fig, (ax,ax2) = plt.subplots(nrows=2,figsize=(8,6),dpi=300,sharex=True)

    plt.subplots_adjust(hspace=0)

    if show_sw:
        ax0.plot(sw.index,sw['E_y_GSM'],c='orange',lw=0.8,label='Ey')
        ax0.set_ylabel('Ey [mV/m]')

    for c in components:
        ax.plot(station.index,station[f'B_{c}_GEO'],lw=0.8,alpha=0.2,label=f'B_{c}_GEO')

    ax.legend(loc='upper left')
    ax.plot(station.index,np.linalg.norm(station[[f'B_{c}_GEO' for c in components]],axis=1),c='b',lw=0.8)
    ax2.plot(pc.index,pc['PCN'],c='r',alpha=0.8,lw=0.8)

    ax2.set_xlabel(f'{sw.index[0].strftime("%b")} {sw.index.year[0]}')
    ax.set_ylabel('|B| [nT]')
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


plot_mag_data(thl_mar, omni_mar, indices_mar, components=('n','e','z'))
plot_mag_data(thl_mar, omni_mar, indices_mar, components=('n','e'))



# %% min_PCN

minimum = indices_year['PCN'].min()
min_time = indices_year['PCN'].idxmin()
print(min_time,minimum)

thl_nov  = thl_year.loc[(thl_year.index.month==11)&(thl_year.index.day<=10)]
omni_nov = omni_year.loc[(omni_year.index.month==11)&(omni_year.index.day<=10)]
indices_nov = indices_year.loc[(indices_year.index.month==11)&(indices_year.index.day<=10)]



plot_mag_data(thl_nov, omni_nov, indices_nov, components=('n','e','z'))
plot_mag_data(thl_nov, omni_nov, indices_nov, components=('n','e'))


