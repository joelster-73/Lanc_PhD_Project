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

# %% Pair_counts
from src.processing.shocks.helsinki import get_list_of_events
event_list = get_list_of_events(helsinki_shocks)

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

# %% Times_v_distances
import numpy as np

distances  = []
times      = []
times_unc  = []
sc_up     = []
sc_dw     = []

for event in event_list:

    for upstream in ('ACE','WIND','DSC'):
        up_time_u = event.get(upstream,None)
        if up_time_u is None:
            continue
        up_time, up_unc = up_time_u
        # consider removing position vector in the shock df
        up_pos  = helsinki_shocks.loc[up_time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
        for downstream in ('OMNI','C1','C3','C4'):
            dw_time_u = event.get(downstream,None)
            if dw_time_u is None:
                continue
            dw_time, dw_unc = dw_time_u

            dw_pos  = helsinki_shocks.loc[dw_time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
            if np.sum(np.abs(dw_pos)>=9999)>1: # Bad data flag
                continue

            times.append((dw_time-up_time).total_seconds())
            times_unc.append(np.sqrt(up_unc**2+dw_unc**2))

            sc_up.append(upstream)
            sc_dw.append(downstream)
            distances.append((up_pos-dw_pos)[0])

distances  = np.array(distances)
times      = np.array(times)/60
times_unc  = np.array(times_unc)/60
sc_up      = np.array(sc_up)
sc_dw      = np.array(sc_dw)

# %% Time_v_dist
from src.analysing.fitting import straight_best_fit
from src.processing.speasy.config import colour_dict
from src.config import R_E

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

marker_dict = {'WIND': 'x', 'ACE': '+', 'DSC': '^'}
ax.errorbar(distances, times, yerr=times_unc, fmt='.', ms=0, ecolor='k', capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

for upstream in ('WIND','ACE','DSC'):
    for downstream in ('OMNI','Cluster'):
        if downstream=='Cluster':
            sc_mask = (sc_up==upstream)&(np.isin(sc_dw, ('C1','C3','C4')))
            sc_c=colour_dict['C1']
        else:
            sc_mask = (sc_up==upstream)&(sc_dw==downstream)
            sc_c=colour_dict[downstream]
        count = np.sum(sc_mask)
        ax.scatter(distances[sc_mask], times[sc_mask], c=sc_c, s=20, marker=marker_dict[upstream], label=f'{upstream} | {downstream}: {count}')
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
ax.legend(loc='upper right')
plt.tight_layout()

plt.show()
plt.close()


# %% Compression
from uncertainties import ufloat, unumpy as unp
df_shocks = helsinki_shocks

compressions_up = []
compressions_dw = []

for event in event_list:

    for upstream in ('ACE','WIND','DSC'):
        up_time_u = event.get(upstream,None)
        if up_time_u is None:
            continue
        up_time, up_unc = up_time_u
        up_comp = df_shocks.loc[up_time, ['B_ratio','B_ratio_unc']].to_numpy()

        up_pos  = df_shocks.loc[up_time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
        for downstream in ('OMNI','C1','C3','C4'):
            dw_time_u = event.get(downstream,None)
            if dw_time_u is None:
                continue
            dw_time, dw_unc = dw_time_u

            dw_pos  = df_shocks.loc[dw_time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
            if np.sum(np.abs(dw_pos)>=9999)>1: # Bad data flag
                dw_pos = None

            dw_comp = df_shocks.loc[dw_time, ['B_ratio','B_ratio_unc']].to_numpy()

            compressions_up.append(ufloat(up_comp[0],up_comp[1]))
            compressions_dw.append(ufloat(dw_comp[0],dw_comp[1]))

# %% Comparing_upstream_downstream_pressures

from src.analysing.fitting import straight_best_fit
import numpy as np
import matplotlib.pyplot as plt

compressions_up = np.array(compressions_up)
compressions_dw = np.array(compressions_dw)

fig, ax = plt.subplots()


xs = unp.nominal_values(compressions_up)
xs_unc = unp.std_devs(compressions_up)

ys = unp.nominal_values(compressions_dw)
ys_unc = unp.std_devs(compressions_dw)

changes = compressions_dw/compressions_up

ax.errorbar(xs, ys, xerr=xs_unc, yerr=ys_unc, fmt='.', ms=0, ecolor='k', capsize=1, capthick=0.5, lw=0.5, zorder=1, alpha=0.8)
ax.scatter(xs,ys,c='b',s=50,marker='x',alpha=0.8)

if True:
    slope, intercept, _ = straight_best_fit(xs,ys,ys_unc,detailed=True)
else:
    slope = np.mean(changes)
    intercept = ufloat(0,0)

if intercept.n<0:
    sign = '-'
else:
    sign = '+'

ax.axline([0,intercept.n],slope=slope.n,c='r',ls='--',lw=1,label=f'Avg: dw = (${slope:L}$) up {sign} (${abs(intercept):L}$)')

ax.axline([0,0],slope=1,c='k',ls=':',lw=1)

ax.axline([0,0],slope=np.min(changes).n,c='m',ls='--',lw=1,label=f'Min: dw = (${np.min(changes):L}$) up')
ax.axline([0,0],slope=np.max(changes).n,c='m',ls='--',lw=1,label=f'Max: dw = (${np.max(changes):L}$) up')

ax.axhline(y=np.min(ys),c='grey',ls=':',lw=1,label=f'Min dw: ${np.min(compressions_dw):L}$')
ax.axhline(y=np.max(ys),c='grey',ls=':',lw=1,label=f'Max dw: ${np.max(compressions_dw):L}$')
ax.axvline(x=np.min(xs),c='grey',ls=':',lw=1,label=f'Min up: ${np.min(compressions_up):L}$')
ax.axvline(x=np.max(xs),c='grey',ls=':',lw=1,label=f'Max up: ${np.max(compressions_up):L}$')

ax.set_xlabel(r'B compression up')
ax.set_ylabel(r'B compression dw')
ax.set_title(f'{len(xs)} shock events')
ax.legend(loc='upper left',fontsize=8)
plt.tight_layout()

plt.show()
plt.close()

