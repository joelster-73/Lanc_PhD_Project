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
from src.analysing.shocks.intercepts import train_propagation_time
from src.analysing.fitting import straight_best_fit

import numpy as np
R_E = 6370

# Only parameter left to train is minimum compression and min overlap maybe - don't bother testing below buffer up

buffer_up_values = list(range(15, 31))
buffer_dw_values = list(range(20, 51))

slopes = np.zeros((len(buffer_dw_values), len(buffer_up_values)))
counts = np.zeros((len(buffer_dw_values), len(buffer_up_values)))
r2_val = np.zeros((len(buffer_dw_values), len(buffer_up_values)))

for i, buffer_dw in enumerate(buffer_dw_values):
    for j, buffer_up in enumerate(buffer_up_values):
        if buffer_dw <= buffer_up:
            for struc in (slopes,counts,r2_val):
                struc[i, j] = np.nan
            continue


        correlated_times = []
        helsinki_times   = []
        coefficients     = []
        correlated_uncs  = []

        total_num_diffs  = 0
        for event in event_list:

            for upstream in ('ACE','WIND'):
                up_time = event.get(upstream,None)
                if up_time is None:
                    continue
                up_pos  = helsinki_shocks.loc[up_time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
                for downstream in ('OMNI','C1','C3','C4'):
                    dw_time = event.get(downstream,None)
                    if dw_time is None:
                        continue
                    total_num_diffs += 1


                    dw_pos  = helsinki_shocks.loc[dw_time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
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
            for struc in (slopes,counts,r2_val):
                struc[i, j] = np.nan
            continue


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


# %%

for limits in (None,((23,28),(27,32))):

    if limits is not None:

        limits_ind = ((buffer_dw_values.index(limits[1][0]),buffer_dw_values.index(limits[1][1])),
                      (buffer_up_values.index(limits[0][0]),buffer_up_values.index(limits[0][1])))

        up_lims = range(limits[0][0],limits[0][1])
        dw_lims = range(limits[1][0],limits[1][1])

    else:
        up_lims = buffer_up_values
        dw_lims = buffer_dw_values

    for name, struc in zip(('slopes', 'counts', r'$R^2$'), (slopes,counts,r2_val)):
        # Plot heatmap
        fig, ax = plt.subplots()
        ax.set_facecolor('k')

        structure = struc
        if limits is not None:
            structure = struc[limits_ind[0][0]:limits_ind[0][1],limits_ind[1][0]:limits_ind[1][1]].copy()

        struc_zeros = structure.copy()
        struc_zeros[np.isnan(struc_zeros)] = 0
        best = np.max(struc_zeros)

        max_coords = np.unravel_index(np.argmax(struc_zeros), struc_zeros.shape)
        x_coord = up_lims[max_coords[1]]+0.5
        y_coord = dw_lims[max_coords[0]]+0.5

        heat = ax.imshow(structure, cmap='Blues_r', aspect='auto',
                  extent=[min(up_lims), max(up_lims)+1,
                          max(dw_lims)+1, min(dw_lims)])

        if limits is not None:
            min_val = np.min(structure[~np.isnan(structure)])
            max_val = np.max(structure[~np.isnan(structure)])
            mid_val = (min_val+max_val)/2

            for i in range(len(structure)):
                for j in range(len(structure[0])):
                    value = structure[i, j]

                    x = up_lims[j]+0.5
                    y = dw_lims[i]+0.5

                    if np.isnan(value):
                        tc = 'k'
                    elif value == best:
                        tc = 'r'
                    elif value < mid_val:
                        tc = 'w'
                    else:
                        tc = 'k'

                    ax.text(x, y, f'{value:.4g}', color=tc, ha='center', va='center', fontsize=12)
        else:
            ax.scatter(x_coord, y_coord, color='red', label=f"Max: {best:.2f}")


        cbar = plt.colorbar(heat)
        cbar.set_label(name)
        ax.invert_yaxis()

        ax.set_xticks(np.array(up_lims)+0.5)
        ax.set_xticklabels(up_lims)

        ax.set_yticks(np.array(dw_lims)+0.5)
        ax.set_yticklabels(dw_lims)

        ax.get_xticklabels()[max_coords[1]].set_color('red')
        ax.get_yticklabels()[max_coords[0]].set_color('red')


        if limits is not None:
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])

        ax.set_aspect('equal')

        ax.set_xlabel('Buffer up [mins]')
        ax.set_ylabel('Buffer dw [mins]')
        ax.set_title(f'Best fit {name} of {total_num_diffs} differences; $\\rho\\geq${coeff_lim}')
        plt.tight_layout()
        plt.show()
        plt.close()

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

import pandas as pd

marker_dict = {'WIND': 'x', 'ACE': '+'}
colour_dict = {'OMNI': 'orange', 'C1': 'b', 'C3': 'b', 'C4': 'b'}

def plot_comparison(xs, ys, ys_unc, zs, title, total, sc_ups, sc_dws, colouring='coeff', coeff_lim=0.7):
    fig, ax = plt.subplots()

    slope, intercept, r2 = straight_best_fit(xs,ys,ys_unc,detailed=True)

    ax.errorbar(xs, ys, yerr=ys_unc, fmt='.', ms=0, ecolor='k', capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

    if colouring=='coeff':
        ax.scatter(xs, ys, c=zs, s=20, marker='x', cmap='plasma_r', vmin=coeff_lim, vmax=1)
        ax.scatter([], [], c='k', s=20, marker='x', label=total)

        scatter = ax.scatter(xs, ys, c=zs, s=0, cmap='plasma_r', vmin=coeff_lim, vmax=1)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Correlation Coeff')
    elif colouring=='sc':

        for sc_up in ('WIND','ACE'):
            for sc_dw in ('OMNI','Cluster'):
                if sc_dw=='Cluster':
                    sc_mask = (sc_ups==sc_up)&(np.isin(sc_dws, ('C1','C3','C4')))
                    sc_c=colour_dict['C1']
                else:
                    sc_mask = (sc_ups==sc_up)&(sc_dws==sc_dw)
                    sc_c=colour_dict[sc_dw]
                count = np.sum(sc_mask)
                ax.scatter(xs[sc_mask], ys[sc_mask], c=sc_c, s=20, marker=marker_dict[sc_up], label=f'{sc_up} | {sc_dw}: {count}')


    ax.axline(slope=1,xy1=[0,0],c='k',ls=':')
    ax.axline([0,intercept.n],slope=slope.n,c='r',ls='--',lw=1)

    ax.axhline(y=0,c='grey',lw=0.2, ls='--')
    ax.axvline(x=0,c='grey',lw=0.2, ls='--')

    if intercept.n<0:
        sign = '-'
    else:
        sign = '+'

    low_lim = min(ax.get_xlim()[0],ax.get_ylim()[0])
    high_lim = max(ax.get_xlim()[1],ax.get_ylim()[1])

    middle = (low_lim+high_lim)/2
    height = ax.get_ylim()[1]-10

    ax.text(middle,height,f'$\\Delta t_c$ = (${slope:L}$)$\\cdot$$\\Delta t_H$\n{sign} (${abs(intercept):L}$) mins, $R^2$={r2:.3f}', ha='center',va='center')

    ax.set_xlabel('Helsinki delays [mins]')
    ax.set_ylabel('Correlated delays [mins]')

    ax.set_xlim(low_lim,high_lim)
    ax.set_ylim(low_lim,high_lim)
    ax.set_title(title)
    ax.set_aspect('equal')

    ax.legend(loc='upper left', fontsize=8)
    plt.show()
    plt.close()

 # %%

buffer_up = 26
buffer_dw = 28

shock_times      = []
correlated_times = []
helsinki_times   = []
coefficients     = []
correlated_uncs  = []
detectors        = []
interceptors     = []

total_num_diffs  = 0
for event in event_list:

    for upstream in ('ACE','WIND'):
        up_time = event.get(upstream,None)
        if up_time is None:
            continue
        up_pos  = helsinki_shocks.loc[up_time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
        for downstream in ('OMNI','C1','C3','C4'):
            dw_time = event.get(downstream,None)
            if dw_time is None:
                continue
            total_num_diffs += 1


            dw_pos  = helsinki_shocks.loc[dw_time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
            if np.sum(np.abs(dw_pos)>=9999)>1: # Bad data flag
                dw_pos = None

            helsinki_delay = (dw_time-up_time).total_seconds()

            delay, coeff, _ = train_propagation_time(up_time, upstream, downstream, 'B_mag', position=up_pos, buffer_up=buffer_up, buffer_dw=buffer_dw, intercept_pos=dw_pos)

            if delay is None:
                #print('None')
                continue

            shock_times.append((up_time,dw_time))

            correlated_times.append(delay.n)
            correlated_uncs.append(delay.s)
            helsinki_times.append(helsinki_delay)
            coefficients.append(coeff)

            detectors.append(upstream)
            interceptors.append(downstream)


    # for erros, neeed to consider what the shock time in helsinki database uncertainty is
    # look at documentation - then need to propagate

shock_times      = np.array(shock_times)

helsinki_times   = np.array(helsinki_times)/60
correlated_times = np.array(correlated_times)/60
correlated_uncs  = np.array(correlated_uncs)/60
coefficients     = np.array(coefficients)

detectors        = np.array(detectors)
interceptors     = np.array(interceptors)




# %%
bad_time = shock_times[correlated_times<=-50]


###-------------------CLOSEST-------------------###
coeff_lim = 0.7
coeff_mask = coefficients>=coeff_lim

x_vals  = helsinki_times[coeff_mask]
y_vals  = correlated_times[coeff_mask]
z_vals  = coefficients[coeff_mask]
y_uncs  = correlated_uncs[coeff_mask]

sc_ups  = detectors[coeff_mask]
sc_dws  = interceptors[coeff_mask]

slope, intercept, r2 = straight_best_fit(x_vals,y_vals,y_uncs,detailed=True)

slopes[i, j] = slope.n
counts[i, j] = len(x_vals)
r2_val[i, j] = r2

title = f'Buffer up: {buffer_up} mins; Buffer dw: {buffer_dw} mins'

for colour_style in ('coeff','sc'):

    plot_comparison(x_vals, y_vals, y_uncs, z_vals, title, total_num_diffs, sc_ups, sc_dws, colouring=colour_style)

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
