# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""

import numpy as np
import pandas as pd

# %% Imports
from src.config import HELSINKI_DIR
from src.processing.reading import import_processed_data
from src.processing.shocks.helsinki import get_list_of_events_simple

helsinki_shocks = import_processed_data(HELSINKI_DIR)
event_list = get_list_of_events_simple(helsinki_shocks)


# %% Imports

from collections import Counter
spacecraft_counts = Counter(helsinki_shocks['spacecraft'].str.upper())


# %% Compression
from uncertainties import ufloat, unumpy as unp
from src.processing.speasy.retrieval import retrieve_omni_value
from src.processing.speasy.config import speasy_variables

monitors = ('ACE','WIND','DSC','C1','C3','C4','OMNI')

sc_ups          = []
sc_dws          = []
compressions_up = []
compressions_dw = []
event_number    = []
time_diffs      = []


for eventNum, event in enumerate(event_list):

    if len(event)<=1:
        continue

    for i in range(2):

        if i==0: # Earliest and latest
            event_copy = event.copy()
            if 'OMNI' in event_copy:
                del event_copy['OMNI']
            upstream   = min(event_copy, key=lambda k: event[k][0])
            downstream = max(event_copy, key=lambda k: event[k][0])
            if upstream==downstream:
                continue

        else: # OMNI
            if 'OMNI' not in event:
                continue
            omni_time = event['OMNI'][0]
            omni_sc = retrieve_omni_value(speasy_variables, omni_time, 'OMNI_sc')
            if omni_sc is None:
                continue
            omni_sc = omni_sc.upper()

            if omni_sc=='WIND-V2':
                upstream = 'WIND'
            else:
                upstream = omni_sc
            downstream = 'OMNI'

            if upstream not in event:
                continue

        up_time, up_unc = event.get(upstream)
        up_comp = helsinki_shocks.loc[up_time, ['B_ratio','B_ratio_unc']]
        if isinstance(up_comp, pd.DataFrame):
            up_comp = up_comp.iloc[0].to_numpy()
        else:
            up_comp = up_comp.to_numpy()


        dw_time_u = event.get(downstream)
        dw_time, dw_unc = dw_time_u


        dw_comp = helsinki_shocks.loc[dw_time, ['B_ratio','B_ratio_unc']]
        if isinstance(dw_comp, pd.DataFrame):
            dw_comp = dw_comp.iloc[0].to_numpy()
        else:
            dw_comp = dw_comp.to_numpy()

        sc_ups.append(upstream)
        sc_dws.append(downstream)
        compressions_up.append(ufloat(up_comp[0],up_comp[1]))
        compressions_dw.append(ufloat(dw_comp[0],dw_comp[1]))
        event_number.append(eventNum)

        dt = (dw_time-up_time).total_seconds()
        dt_unc = ufloat(dt,dw_unc)-ufloat(0,up_unc)
        time_diffs.append(ufloat(dt,dt_unc.s))

sc_ups = np.array(sc_ups)
sc_dws = np.array(sc_dws)
compressions_up = np.array(compressions_up)
compressions_dw = np.array(compressions_dw)
event_number = np.array(event_number)
time_diffs = np.array(time_diffs)


# %%

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

from src.processing.speasy.config import colour_dict
from src.plotting.utils import save_figure
from src.analysing.fitting import gaussian, gaussian_fit
from src.analysing.fitting import straight_best_fit


import itertools as it

marker_dict = {'WIND': 'x', 'ACE': '+', 'DSC': '^', 'Cluster': 'o'}

monitors = ('ACE','WIND','DSC','Cluster','OMNI')

plot_type = 'all'

selection = 'omni' # omni or other

title_info = 'All spacecraft'
if selection=='omni':
    title_info = 'Spacecraft in OMNI'

sc_mask_initial = np.all(len(compressions_up))

if selection=='omni':
    sc_mask_initial &= (sc_dws=='OMNI')
    x_label = r'$B_\mathrm{ratio}$ (sc)'
    y_label = r'$B_\mathrm{ratio}$ (OMNI)'
    count_label = r'$B_\mathrm{ratio}$ (OMNI / sc)'
    title_info = 'OMNI and Detector'
else:
    sc_mask_initial &= (sc_dws!='OMNI')
    x_label = r'$B_\mathrm{ratio}$ (sc #1)'
    y_label = r'$B_\mathrm{ratio}$ (sc #2)'
    count_label = r'$B_\mathrm{ratio}$ (sc # 2 / sc #1)'
    title_info = 'Earliest and Latest Spacecraft'

x_lim = (1,5)
y_lim = (1,5)

sc_mask_initial &= (compressions_up>=x_lim[0]) & (compressions_up<=x_lim[1])
sc_mask_initial &= (compressions_dw>=y_lim[0]) & (compressions_dw<=y_lim[1])

xs     = unp.nominal_values(compressions_up[sc_mask_initial])
xs_unc = unp.std_devs(compressions_up[sc_mask_initial])

ys     = unp.nominal_values(compressions_dw[sc_mask_initial])
ys_unc = unp.std_devs(compressions_dw[sc_mask_initial])

ts     = unp.nominal_values(time_diffs[sc_mask_initial])/60
ts_unc = unp.std_devs(time_diffs[sc_mask_initial])/60

sc_ups_masked = sc_ups[sc_mask_initial]
sc_dws_masked = sc_dws[sc_mask_initial]

events_masked = event_number[sc_mask_initial]
num_events = len(np.unique(events_masked))

changes = (compressions_dw[sc_mask_initial])/(compressions_up[sc_mask_initial])
slope   = np.mean(changes)

cs = unp.nominal_values(changes)
cs_unc = unp.std_devs(changes)



step = 0.05

if plot_type in ('all','hist'):

    change_data = unp.nominal_values(changes)
    change_errs = unp.std_devs(changes)

    fig, ax = plt.subplots()

    bin_edges = np.arange(np.floor(np.min(change_data)/step)*step,np.ceil(np.max(change_data)/step)*step,step)

    counts, bins = np.histogram(change_data, bin_edges)
    mids = 0.5*(bins[1:]+bins[:-1])

    plot_counts, _, patches = ax.hist(change_data, bin_edges, color='k', edgecolor='#333333')


    ###----------COLOURING----------###
    clipping = 1.3 if selection=='omni' else 1.2
    norm = mcolors.Normalize(vmin=min(plot_counts), vmax=max(plot_counts)*clipping)
    if selection=='omni':
        cmap = plt.colormaps['Oranges']
    else:
        cmap = ListedColormap(plt.colormaps['PiYG'](np.linspace(0, 0.45, 128))[::-1])
    for count, patch in zip(plot_counts, patches):
        colour = cmap(norm(count))
        patch.set_facecolor(colour)

    ###----------GAUSSIAN----------###
    x_values = np.linspace(min(change_data), max(change_data), 1000)

    A, mu, sig = gaussian_fit(mids,counts,detailed=True,simple_bounds=True)
    y_values = gaussian(x_values, A.n, mu.n, sig.n)
    ax.plot(x_values,y_values,c='k', lw=2)

    #ax.axvline(x=np.median(change_data),c='k',ls='--',lw=1.25)

    text_info = f'$\\mu$: ${mu:L}$\n$\\sigma$: ${sig:L}$'
    ax.text(0.9*mu.n,0.9*np.max(counts),text_info, ha='right',va='top')

    ###----------LABELS----------###

    ax.set_xlim(np.min(mids)-2*step,np.max(mids)+2*step)
    ax.axvline(x=1,ls=':',c='grey',lw=1)

    ax.set_xlabel(count_label)
    ax.set_ylabel('Counts')

    ax.set_title(f'Compression - {title_info}: N={num_events}')

    plt.tight_layout()
    save_figure(fig)

    plt.show()
    plt.close()



if plot_type in ('all','scatter'):

    fig, ax = plt.subplots()
    ax.errorbar(xs, ys, xerr=xs_unc, yerr=ys_unc, fmt='.', ms=0, ecolor='k', capsize=1, capthick=0.5, lw=0.5, zorder=1, alpha=0.2)

    for upstream, downstream in it.permutations(monitors, 2):
        sc_mask = np.all(len(xs))
        if upstream=='Cluster':
            sc_mask &= np.isin(sc_ups_masked, ('C1','C3','C4'))
        else:
            sc_mask &= (sc_ups_masked==upstream)

        if downstream=='Cluster':
            sc_mask &= np.isin(sc_dws_masked, ('C1','C3','C4'))
            sc_c=colour_dict['C1']
        else:
            sc_mask &= (sc_dws_masked==downstream)
            sc_c=colour_dict[downstream]

        count = np.sum(sc_mask)
        if count==0:
            continue
        ax.scatter(xs[sc_mask], ys[sc_mask], c=sc_c, s=40, alpha=0.6, marker=marker_dict[upstream], label=f'{upstream} | {downstream}: {count}')


    for m in np.arange(np.floor(np.min(change_data)/step)*step,np.ceil(np.max(change_data)/step)*step,step):
        if m==1:
            continue
        ax.axline([0,0],slope=m,c='grey',ls=':',lw=1,alpha=0.125)


    ax.axline([0,0],slope=slope.n,c='k',ls='--',lw=1.25,label=f'Mean: $B_{{r,2}}$ = (${slope:L}$) $B_{{r,1}}$')
    ax.axline([0,0],slope=1,c='grey',ls=':',lw=1.25)


    ax.set_xlim(1,5)
    ax.set_ylim(1,5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{title_info}: N={num_events}')
    ax.legend(loc='upper left',fontsize=8)
    plt.tight_layout()
    save_figure(fig)

    plt.show()
    plt.close()


if plot_type in ('all','scatter_dt'):

    fig, ax = plt.subplots()
    ax.errorbar(ts, cs, xerr=ts_unc, yerr=cs_unc, fmt='.', ms=0, ecolor='k', capsize=1, capthick=0.5, lw=0.5, zorder=1, alpha=0.2)

    for upstream, downstream in it.permutations(monitors, 2):
        sc_mask = np.all(len(xs))
        if upstream=='Cluster':
            sc_mask &= np.isin(sc_ups_masked, ('C1','C3','C4'))
        else:
            sc_mask &= (sc_ups_masked==upstream)

        if downstream=='Cluster':
            sc_mask &= np.isin(sc_dws_masked, ('C1','C3','C4'))
            sc_c=colour_dict['C1']
        else:
            sc_mask &= (sc_dws_masked==downstream)
            sc_c=colour_dict[downstream]

        count = np.sum(sc_mask)
        if count==0:
            continue
        ax.scatter(ts[sc_mask], cs[sc_mask], c=sc_c, s=40, alpha=0.6, marker=marker_dict[upstream], label=f'{upstream} | {downstream}: {count}')


    slope, intercept, r2 = straight_best_fit(ts,cs,cs_unc,detailed=True)

    ax.axline([0,intercept.n],slope=slope.n,c='k',ls='--',lw=1)
    text_info = f'y = $({slope:L})\\cdot x$ + ${intercept:L}$\n$R^2=${r2:.3f}'
    ax.text(np.max(ts),np.max(cs),text_info, ha='right',va='top')

    ax.set_xlabel('Time diff [mins]')
    ax.set_ylabel(count_label)
    ax.set_title(f'{title_info}: N={num_events}')
    ax.legend(loc='upper left',fontsize=8)
    plt.tight_layout()
    save_figure(fig)

    plt.show()
    plt.close()



