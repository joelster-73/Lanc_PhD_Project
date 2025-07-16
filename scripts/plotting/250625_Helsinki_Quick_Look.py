# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""

# %% Imports
from src.config import HELSINKI_DIR
from src.processing.reading import import_processed_data
from src.processing.shocks.helsinki import get_list_of_events_simple

helsinki_shocks = import_processed_data(HELSINKI_DIR)
event_list = get_list_of_events_simple(helsinki_shocks)


# %% Imports
from src.processing.speasy.config import colour_dict
import matplotlib.pyplot as plt
from collections import Counter
from src.plotting.utils import save_figure

import numpy as np
from src.analysing.fitting import straight_best_fit
from src.config import R_E

import pandas as pd

from src.processing.speasy.retrieval import retrieve_omni_value, retrieve_modal_omni_sc
from src.processing.speasy.config import speasy_variables
from datetime import timedelta

# %%

spacecraft_counts = Counter(helsinki_shocks['spacecraft'].str.upper())

fig, ax = plt.subplots()
for i, (sc,count) in enumerate(spacecraft_counts.items()):
    ax.bar(i,count,color=colour_dict.get(sc,'k'),width=1)
    ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)  # Add text above bars


ax.set_xticks(range(len(spacecraft_counts)))
ax.set_xticklabels(spacecraft_counts.keys())

ax.set_ylabel('Count')
ax.set_title('Spacecraft Observation Counts in Helsinki')
plt.tight_layout()
plt.show()

# %% Pair_counts

omni_and_L1 = 0
omni_and_wind = 0
omni_and_ace  = 0
omni_and_dsc  = 0
for event in event_list:
    if 'OMNI' in event:
        if 'WIND' in event or 'ACE' in event or 'DSC' in event:
            if 'WIND' in event:
                omni_and_wind += 1

            if 'ACE' in event:
                omni_and_ace += 1

            if 'DSC' in event:
                omni_and_dsc += 1

            omni_and_L1 += 1

event_count_dict = {
    'OMNI': omni_and_L1,
    'WIND': omni_and_wind,
    'ACE':  omni_and_ace,
    'DSC':  omni_and_dsc}

fig, ax = plt.subplots()
for i, (sc,count) in enumerate(event_count_dict.items()):
    ax.bar(i,count,color=colour_dict.get(sc,'k'),width=1)
    ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)  # Add text above bars


ax.set_xticks(range(len(event_count_dict)))
ax.set_xticklabels(event_count_dict.keys())

ax.set_ylabel('Count')
ax.set_title('Shock Events Measured by OMNI & L1')
plt.tight_layout()
plt.show()

# %%
import itertools as it

distances  = []
times      = []
times_unc  = []
sc_up      = []
sc_dw      = []
event_number   = []
omni_sc_used   = []
omni_timeshift = []


monitors = ('ACE','WIND','DSC','C1','C3','C4','OMNI')

for eventNum, event in enumerate(event_list):

    for upstream, downstream in it.combinations(monitors, 2):
        up_time_u = event.get(upstream,None)
        if up_time_u is None:
            continue
        up_time, up_unc = up_time_u
        up_pos  = helsinki_shocks.loc[up_time,['r_x_GSE','r_y_GSE','r_z_GSE']]
        if isinstance(up_pos, pd.DataFrame):
            up_pos = up_pos.iloc[0].to_numpy()
        else:
            up_pos = up_pos.to_numpy()

        omni_sc = None
        omni_ts = np.nan
        if 'OMNI' in event:
            omni_time_u = event.get('OMNI',None)
            if omni_time_u is None:
                omni_sc = None
            else:
                omni_time, omni_unc = omni_time_u
                omni_sc = retrieve_omni_value(speasy_variables, omni_time,'OMNI_sc')
                if omni_sc is not None:
                    omni_sc = omni_sc.upper()
            omni_ts = retrieve_omni_value(speasy_variables, omni_time)

        dw_time_u = event.get(downstream,None)
        if dw_time_u is None:
            continue
        dw_time, dw_unc = dw_time_u

        dw_pos  = helsinki_shocks.loc[dw_time,['r_x_GSE','r_y_GSE','r_z_GSE']]
        if isinstance(dw_pos, pd.DataFrame):
            dw_pos = dw_pos.iloc[0].to_numpy()
        else:
            dw_pos = dw_pos.to_numpy()

        if downstream=='OMNI' and np.sum(np.abs(dw_pos)>=9999)>1: # Bad data flag
            continue

        time_diff = (up_time-dw_time).total_seconds()
        dist_diff = (up_pos-dw_pos)[0]

        distances.append(dist_diff)
        times.append(time_diff)
        times_unc.append(np.sqrt(up_unc**2+dw_unc**2))

        sc_up.append(upstream)
        sc_dw.append(downstream)

        event_number.append(eventNum)
        omni_sc_used.append(omni_sc)
        omni_timeshift.append(omni_ts)

distances      = np.array(distances)
times          = np.array(times)/60
times_unc      = np.array(times_unc)/60
sc_up          = np.array(sc_up)
sc_dw          = np.array(sc_dw)
event_number   = np.array(event_number)
omni_sc_used   = np.array(omni_sc_used)
omni_timeshift = np.array(omni_timeshift)/60

# %%

#max_dist = 300, selection = not_used_sc_omni, plot_type = dt_hist
#max_dist = 60,  selection = not_used_sc_omni, plot_type = dt_hist
#max_dist = 300, selection = used_sc_omni,     plot_type = dt_vs_dx
#max_dist = 300, selection = all_other_sc,     plot_type = dt_vs_dx
#Compressions used by omni
#Compressions for combos

mask = np.all(len(distances))

max_distance = 300
selection = 'all_other_sc'
plot_type = 'dt_vs_dx'

mask &= (distances<max_distance)
if selection=='used_sc_omni':
    mask &= ((omni_sc_used=='WIND-V2') & (sc_up=='WIND') & (sc_dw=='OMNI') | (omni_sc_used!='WIND-V2') & (sc_up==omni_sc_used) & (sc_dw=='OMNI'))
elif selection=='not_used_sc_omni':
    mask &= ((omni_sc_used == 'WIND-V2') & (sc_up != 'WIND') & (sc_dw=='OMNI')) | ((omni_sc_used != 'WIND-V2') & (sc_up != omni_sc_used) & (sc_dw=='OMNI'))
elif selection=='all_other_sc':
    mask &= sc_dw!='OMNI'


ds         = distances[mask]
ts         = times[mask]
ts_unc     = times_unc[mask]
sc_ups     = sc_up[mask]
sc_dws     = sc_dw[mask]
event_nums = event_number[mask]
omni_scs   = omni_sc_used[mask]


num_events = len(np.unique(event_nums))
title_info = 'All spacecraft'
if selection=='not_used_sc':
    title_info = 'Spacecraft Non-OMNI'
elif selection=='used_sc':
    title_info = 'Spacecraft in OMNI'

monitors = ('ACE', 'WIND', 'DSC', 'Cluster', 'OMNI')
marker_dict = {'WIND': 'x', 'ACE': '+', 'DSC': '^', 'Cluster': 'o'}

if 'omni' in selection:
    x_label = r'$X_{SC}$ - $X_{BSN}$ [$R_E$]'
    t_label = r'$t_{SC}$ - $t_{OMNI}$ [$R_E$]'
else:
    x_label = r'$X_{SC,1}-X_{SC,2}$ [$R_E$]'
    t_label = r'$t_{SC,1}-t_{SC,2}$ [mins]'


if plot_type in ('both','dt_vs_dx'):
    fig, ax = plt.subplots()

    ax.errorbar(ds, ts, yerr=ts_unc, fmt='.', ms=0, ecolor='k', capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

    for upstream, downstream in it.combinations(monitors, 2):
        sc_mask = np.all(ts)
        if upstream=='Cluster':
            sc_mask &= np.isin(sc_ups, ('C1','C3','C4'))
        else:
            sc_mask &= (sc_ups==upstream)

        if downstream=='Cluster':
            sc_mask &= np.isin(sc_dws, ('C1','C3','C4'))
            sc_c=colour_dict['C1']
        else:
            sc_mask &= (sc_dws==downstream)
            sc_c=colour_dict[downstream]

        count = np.sum(sc_mask)
        if count==0:
            continue
        ax.scatter(ds[sc_mask], ts[sc_mask], c=sc_c, s=20, marker=marker_dict[upstream], label=f'{upstream} | {downstream}: {count}')
    slope, intercept, r2 = straight_best_fit(ds,ts,ts_unc,detailed=True)

    ax.axline([0,intercept.n],slope=slope.n,c='k',ls='--',lw=1)
    slope_speed = 1/slope * R_E / 60 # km/s
    ax.axhline(y=0,ls=':',c='grey',alpha=0.5)
    ax.axvline(x=0,ls=':',c='grey',alpha=0.5)

    if intercept.n<0:
        sign = '-'
    else:
        sign = '+'
    middle = (np.max(ds)+np.min(ds))/2
    top = np.max(ts)
    ax.text(middle,top,f'$\\Delta t$ = (${slope:L}$)$\\cdot\\Delta x$ {sign} (${abs(intercept):L}$) mins\n$R^2$={r2:.3f}, $v_x={slope_speed:L}\\,\\mathrm{{km\\ s^{{-1}}}}$', ha='center',va='center')

    ax.set_xlabel(x_label)
    ax.invert_xaxis()
    ax.set_ylabel(t_label)

    ax.set_title(f'{title_info}: $\\Delta X<{max_distance}$, N={num_events}')
    ax.legend(loc='upper left',title=f'{len(ds)} times')

    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

if plot_type in ('both','dt_hist'):
    fig, ax = plt.subplots()

    step = 5
    bin_edges = np.arange(np.floor(np.min(ts)/step)*step,np.ceil(np.max(ts/step)*step),step)

    counts, bins = np.histogram(ts, bin_edges)
    mids = 0.5*(bins[1:]+bins[:-1])

    ax.hist(ts, bin_edges, color='k')

    ax.axvline(x=np.median(ts),ls='--',lw=1,c='c',label=f'Median: {np.median(ts):.3g} mins')
    ax.axvline(x=0,ls=':',c='w',lw=1)


    ax.set_xlabel(t_label)
    ax.set_ylabel('Counts / 5mins')

    ax.legend()
    ax.set_title(f'{title_info}: $\\Delta X<{max_distance}$, N={num_events}')
    ax.legend(loc='upper left',title=f'{len(ds)} times')

    plt.tight_layout()
    save_figure(fig)

    plt.show()
    plt.close()

# %% Compression
from uncertainties import ufloat, unumpy as unp
from src.analysing.fitting import straight_best_fit
import numpy as np
import matplotlib.pyplot as plt

monitors = ('ACE','WIND','DSC','C1','C3','C4','OMNI')

sc_ups          = []
sc_dws          = []
compressions_up = []
compressions_dw = []
event_number    = []


for eventNum, event in enumerate(event_list):

    if len(event)<=1:
        continue

    for i in range(2):

        if i==1:
            if 'OMNI' not in event:
                continue
            omni_time = event['OMNI'][0]
            omni_sc = retrieve_modal_omni_sc(speasy_variables, omni_time-timedelta(minutes=20), omni_time+timedelta(minutes=20), return_counts=False)
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

        else:
            event_copy = event.copy()
            if 'OMNI' in event_copy:
                del event_copy['OMNI']
            upstream   = min(event_copy, key=lambda k: event[k][0])
            downstream = max(event_copy, key=lambda k: event[k][0])
            if upstream==downstream:
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



sc_ups = np.array(sc_ups)
sc_dws = np.array(sc_dws)
compressions_up = np.array(compressions_up)
compressions_dw = np.array(compressions_dw)
event_number = np.array(event_number)


# %%
monitors = ('ACE','WIND','DSC','Cluster','OMNI')

plot_type = 'hist'
selection = 'other'

title_info = 'All spacecraft'
if selection=='not_used_sc':
    title_info = 'Spacecraft Non-OMNI'
elif selection=='used_sc':
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

sc_ups_masked = sc_ups[sc_mask_initial]
sc_dws_masked = sc_dws[sc_mask_initial]

events_masked = event_number[sc_mask_initial]
num_events = len(np.unique(events_masked))

changes = (compressions_dw[sc_mask_initial])/(compressions_up[sc_mask_initial])
slope   = np.mean(changes)

if plot_type in ('both','scatter'):

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

    ax.axline([0,0],slope=slope.n,c='k',ls='--',lw=1,label=f'Mean: $B_{{r,2}}$ = (${slope:L}$) $B_{{r,1}}$')
    ax.axline([0,0],slope=1,c='grey',ls=':',lw=1)


    ax.set_xlim(1,5)
    ax.set_ylim(1,5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f'{title_info}: N={num_events}')
    ax.legend(loc='upper left',title=f'{len(ds)} times',fontsize=8)
    plt.tight_layout()
    save_figure(fig)

    plt.show()
    plt.close()

if plot_type in ('both','hist'):

    change_data = unp.nominal_values(changes)

    fig, ax = plt.subplots()

    step = 0.1
    bin_edges = np.arange(np.floor(np.min(change_data)/step)*step,np.ceil(np.max(change_data/step)*step),step)

    counts, bins = np.histogram(change_data, bin_edges)
    mids = 0.5*(bins[1:]+bins[:-1])

    ax.hist(change_data, bin_edges, color='k')

    ax.axvline(x=np.median(change_data),ls='--',lw=1,c='c',label=f'Median: {np.median(change_data):.3g} mins')
    ax.axvline(x=np.mean(change_data),ls='--',lw=1,c='r',label=f'Mean: {np.mean(changes):.3g} mins')
    ax.axvline(x=0,ls=':',c='w',lw=1)


    ax.set_xlabel(count_label)
    ax.set_ylabel('Counts / 5mins')

    ax.legend()
    ax.set_title(f'{title_info}: N={num_events}')
    ax.legend(loc='upper left',title=f'{len(change_data)} times')

    plt.tight_layout()
    save_figure(fig)

    plt.show()
    plt.close()


# %%
import matplotlib.pyplot as plt
from src.processing.speasy.config import colour_dict
from src.plotting.config import save_fig
from src.analysing.fitting import straight_best_fit

omni_times_mask = (sc_up==omni_sc_used) & (sc_dw=='OMNI')

detectors = sc_up[omni_times_mask]
xs = omni_timeshift[omni_times_mask]
ys = -times[omni_times_mask]
ys_unc = times_unc[omni_times_mask]

colours = pd.Series(detectors).map(colour_dict).fillna('k').to_numpy()

fig, ax = plt.subplots()

ax.errorbar(xs, ys, yerr=ys_unc, fmt='.', ms=0, ecolor='k', capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

ax.scatter(xs, ys, s=2, c=colours)
ax.axline([0,0],slope=1,c='k',ls=':')

slope, intercept, r2 = straight_best_fit(xs,ys,ys_unc,detailed=True)

ax.axline([0,intercept.n],slope=slope.n,c='r',ls='--',lw=1)
if intercept.n<0:
    sign = '-'
else:
    sign = '+'
middle = (np.max(xs)+np.min(xs))/2
location = np.max(np.abs(ys))
ax.text(middle,location,f'$\\Delta t$ = (${slope:L}$)$\\Delta r$ {sign} (${abs(intercept):L}$) mins\n$R^2$={r2:.3f}',
        ha='center',va='center')

ax.set_xlabel('OMNI timeshift [mins]')
ax.set_ylabel('Helsinki Shock timelag [mins]')
ax.set_title('Comparing Helsinki Shock Timelag against OMNI Timeshift')

plt.show()