# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""



# %% Imports
from src.config import HELSINKI_DIR
from src.processing.reading import import_processed_data

helsinki_shocks = import_processed_data(HELSINKI_DIR)

# %% Counts
from src.processing.speasy.config import colour_dict
import matplotlib.pyplot as plt
from collections import Counter

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

# %% Events
event_list = []
total_events = len(helsinki_shocks)
iterator = helsinki_shocks.iterrows()


prev_index, prev_shock = next(iterator)
prev_sc    = prev_shock['spacecraft']
event_dict = {prev_sc: prev_index}

while True:
    try:
        index, shock = next(iterator)
        sc = shock['spacecraft']
        if sc in event_dict:
            same_event = False
        elif (index-prev_index).total_seconds()>=(90*60):
            same_event = False
        else:
            same_event = True

        if same_event:
            event_dict[sc] = index

        else:
            event_list.append(event_dict)
            event_dict = {sc: index}
            prev_index = index

    except StopIteration:
        break

event_list = event_list[::-1] # reverses order so begins with most recent

print(event_list)


# %%
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.facecolor'] = 'w'
plt.rcParams['legend.edgecolor'] = 'k'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 15

from src.analysing.fitting import straight_best_fit

marker_dict = {'B_mag': 'x', 'B_GSE_x': 's', 'B_GSE_y': 'o', 'B_GSE_z': '+'}

def plot_comparison(xs, ys, ys_unc, params, zs, title, total):
    fig, ax = plt.subplots()

    slope, intercept, r2 = straight_best_fit(xs,ys,ys_unc,detailed=True)

    ax.errorbar(xs, ys, yerr=ys_unc, fmt='.', ms=0, ecolor='k', capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

    for param, marker in marker_dict.items():
        ms    = 20 if marker in ('x','+') else 10
        mask  = params == param
        count = np.sum(mask)
        if count==0:
            continue

        ax.scatter(xs[mask], ys[mask], c=zs[mask], s=ms, marker=marker, cmap='plasma_r', vmin=corr_lim, vmax=1)

        ax.scatter([], [], c='k', s=ms, marker=marker, label=f'{param}: {count} / {total}')

    scatter = ax.scatter(xs, ys, c=zs, s=0, cmap='plasma_r', vmin=corr_lim, vmax=1)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Correlation Coeff')

    ax.axline(slope=1,xy1=[0,0],c='k',ls=':')
    ax.axline([0,intercept.n],slope=slope.n,c='r',ls='--',lw=1)

    ax.axhline(y=0,c='grey',lw=0.2, ls='--')
    ax.axvline(x=0,c='grey',lw=0.2, ls='--')

    if intercept.n<0:
        sign = '-'
    else:
        sign = '+'

    x_lim = max(np.abs(ax.get_xlim()))+10
    y_lim = max(np.abs(ax.get_ylim()))+10

    middle = 0
    height = y_lim-10

    ax.text(middle,height,f'$\\Delta t_c$ = (${slope:L}$)$\\cdot$$\\Delta t_H$ {sign} (${abs(intercept):L}$) mins\n$R^2$={r2:.3f}', ha='center',va='center')

    ax.set_xlabel('Helsinki delays [mins]')
    ax.set_ylabel('Correlated delays [mins]')


    ax.set_xlim(-x_lim,x_lim)
    ax.set_ylim(-y_lim,y_lim)
    ax.set_title(title)

    ax.legend(loc='upper left', fontsize=10)
    plt.show()
    plt.close()

# %%
from src.analysing.shocks.intercepts import train_propagation_time
import numpy as np
R_E = 6370

# Only parameter left to train is minimum compression and min overlap maybe - don't bother testing below buffer up

buffer_up_values = range(10, 61)
buffer_dw_values = range(10, 61)

slopes = np.zeros((len(buffer_dw_values), len(buffer_up_values)))
counts = np.zeros((len(buffer_dw_values), len(buffer_up_values)))
r2_val = np.zeros((len(buffer_dw_values), len(buffer_up_values)))

for i, buffer_dw in enumerate(buffer_dw_values):
    for j, buffer_up in enumerate(buffer_up_values):
        # if buffer_dw <= buffer_up:
        #     for struc in (slopes,counts,r2_val):
        #         struc[i, j] = np.nan
        #     continue


        correlated_times = []
        helsinki_times   = []
        coefficients     = []
        correlated_uncs  = []

        total_num_events = 0
        total_num_diffs  = 0
        for event in event_list:

            for upstream in ('ACE','WIND'):
                for downstream in ('OMNI','C1','C3','C4'):
                    up_time = event.get(upstream,None)
                    dw_time = event.get(downstream,None)
                    if up_time is None or dw_time is None:
                        continue
                    total_num_diffs += 1

                    up_pos  = helsinki_shocks.loc[up_time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
                    dw_pos  = helsinki_shocks.loc[dw_time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()


                    if np.sum(np.abs(up_pos)>=9999)>1: # Bad data flag
                        up_pos = None

                    if np.sum(np.abs(dw_pos)>=9999)>1: # Bad data flag
                        dw_pos = None

                    helsinki_delay = (dw_time-up_time).total_seconds()

                    delay, coeff, _ = train_propagation_time(up_time, upstream, downstream, 'B_mag', position=up_pos, buffer_up=buffer_up, buffer_dw=buffer_dw, intercept_pos=dw_pos)

                    if delay is None:
                        #print('None')
                        continue


                    #print('Helsinki',(omni_time-wind_time).total_seconds())
                    #print()

                    correlated_times.append(delay.n)
                    correlated_uncs.append(delay.s)
                    helsinki_times.append(helsinki_delay)
                    coefficients.append(coeff)


            # for erros, neeed to consider what the shock time in helsinki database uncertainty is
            # look at documentation - then need to propagate

        if len(correlated_times)<=2:
            continue
        total_num_events += 1

        helsinki_times   = np.array(helsinki_times)/60
        correlated_times = np.array(correlated_times)/60
        correlated_uncs  = np.array(correlated_uncs)/60
        coefficients     = np.array(coefficients)


        ###-------------------CLOSEST-------------------###
        coeff_lim = 0.7
        coeff_mask = coefficients>=coeff_lim

        x_vals     = helsinki_times[coeff_mask]
        y_vals     = correlated_times[coeff_mask]
        z_vals     = coefficients[coeff_mask]
        y_uncs     = correlated_uncs[coeff_mask]

        slope, intercept, r2 = straight_best_fit(x_vals,y_vals,y_uncs,detailed=True)

        slopes[i, j] = slope.n
        counts[i, j] = len(x_vals)
        r2_val[i, j] = r2

        #title = f'Buffer up: {buffer_up} mins; Buffer dw: {buffer_dw} mins'

        #plot_comparison(x_vals, y_vals, y_uncs, y_params, z_vals, title, total_num)



        # when get some good idea for values, then do ace and wind as different markers
        # and omni and cluster as different colours
        # then plot comparison
# %%
for name, struc in zip(('slopes', 'counts', 'R2'), (slopes,counts,r2_val)):
    # Plot heatmap
    fig, ax = plt.subplots()
    ax.set_facecolor('k')
    heat = ax.imshow(struc, cmap='viridis', aspect='auto',
              extent=[min(buffer_up_values), max(buffer_up_values),
                      max(buffer_dw_values), min(buffer_dw_values)])  # Flip y-axis
    cbar = plt.colorbar(heat)
    cbar.set_label(name)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel('Buffer up [mins]')
    ax.set_ylabel('Buffer dw [mins]')
    ax.set_title(f'Best fit {name} for range of buffers for {total_num_events} events / {total_num_diffs} pairs. $\\rho\\geq${coeff_lim}')
    plt.tight_layout()
    plt.show()
    plt.close()
# %%
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


# %% Cluster_OMNI
distances = []
times = []
spacecraft = []

for event in event_list:
    if 'OMNI' not in event:
        continue

    if 'C1' in event:
        cluster = 'C1'
    elif 'C3' in event:
        cluster = 'C3'
    elif 'C4' in event:
        cluster = 'C4'
    else:
        continue

    omni_pos = helsinki_shocks.loc[event['OMNI'], ['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
    clus_pos = helsinki_shocks.loc[event[cluster],['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()

    omni_time = event['OMNI']
    cluster_time = event[cluster]

    time_diff = (cluster_time - omni_time).total_seconds()/60
    dist_diff = clus_pos[0] - omni_pos[0]
    #dist_diff = np.linalg.norm(clus_pos-omni_pos)

    distances.append(dist_diff)
    times.append(time_diff)
    spacecraft.append(cluster)


# %%
from src.analysing.fitting import straight_best_fit
from src.processing.speasy.config import colour_dict
from src.config import R_E

import pandas as pd
import numpy as np

fig, ax = plt.subplots()

colours = pd.Series(spacecraft).map(colour_dict).fillna('k').to_numpy()
ax.scatter(distances,times,c=colours)

slope, intercept, r2 = straight_best_fit(distances,times,None,detailed=True)

ax.axline([0,intercept.n],slope=slope.n,c='k',ls='--',lw=1)

slope_speed = -1/slope * R_E / 60 # km/s

if intercept.n<0:
    sign = '-'
else:
    sign = '+'
middle = (np.max(distances)+np.min(distances))/2
location = np.max(np.abs(times))
ax.text(middle,location,f'$\\Delta t$ = (${slope:L}$)$\\Delta r$ {sign} (${abs(intercept):L}$) mins\n$R^2$={r2:.3f}, $v={slope_speed:L}$ km/s',
        ha='center',va='center')

ax.set_xlabel(r'$X_{sc}$ - $X_{BSN}$ [$R_E$]')
ax.invert_xaxis()
ax.set_ylabel(r'$t_{SC}$ - $t_{OMNI}$ [mins]')
ax.set_title(f'{len(distances)} shock events')
plt.tight_layout()

plt.show()
plt.close()
