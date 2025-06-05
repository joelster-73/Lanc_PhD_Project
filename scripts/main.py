# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""
# %%
from src.config import PROC_CFA_DIR
from src.processing.reading import import_processed_data

shocks = import_processed_data(PROC_CFA_DIR)

# %%
import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()

delay_times = shocks['delay_s'].to_numpy()/60
delay_times_unc = shocks['delay_s_unc'].to_numpy()/60
valids = ~np.isnan(delay_times) & ~np.isnan(delay_times_unc)

delay_times = delay_times[valids]
delay_times_unc = delay_times_unc[valids]
sorted_array1, sorted_array2 = zip(*sorted(zip(delay_times, delay_times_unc)))
delay_times = np.array(list(sorted_array1))
delay_times_unc = np.array(list(sorted_array2))

y_values = np.arange(1,2*len(delay_times)+1,2)

booleans = (delay_times_unc>np.abs(delay_times))
colours = ['r' if bol else 'grey' for bol in booleans]
ax.errorbar(delay_times, y_values, xerr=delay_times_unc, yerr=0, fmt='k.', ecolor=colours, elinewidth=0.2, markersize=0.1)

ax.set_xlim(-60,120)
ax.set_xlabel('Delay time [mins]')
ax.set_ylabel('#')
ax.set_title(f'CFA Delay times and errors {np.sum(booleans)}/{len(booleans)}')
plt.show()

# %%

fig, ax = plt.subplots()

delay_times = shocks['v_sh'].to_numpy()
delay_times_unc = shocks['v_sh_unc'].to_numpy()
valids = ~np.isnan(delay_times) & ~np.isnan(delay_times_unc)

delay_times = delay_times[valids]
delay_times_unc = delay_times_unc[valids]
sorted_array1, sorted_array2 = zip(*sorted(zip(delay_times, delay_times_unc)))
delay_times = np.array(list(sorted_array1))
delay_times_unc = np.array(list(sorted_array2))

y_values = np.arange(1,2*len(delay_times)+1,2)

booleans = (delay_times_unc>np.abs(delay_times))
colours = ['r' if bol else 'grey' for bol in booleans]
ax.errorbar(delay_times, y_values, xerr=delay_times_unc, yerr=0, fmt='k.', ecolor=colours, elinewidth=0.2, markersize=0.1)

ax.set_xlim(-100,1000)
ax.set_xlabel('Shock speed [km/s]')
ax.set_ylabel('#')
ax.set_title(f'CFA Shock speeds and errors {np.sum(booleans)}/{len(booleans)}')
plt.show()

# %% Importing
from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data

shocks_intercepts = import_processed_data(PROC_SHOCKS_DIR)

# %%
import numpy as np
import matplotlib.pyplot as plt

columns = [col for col in shocks_intercepts if '_coeff' in col]

all_coeffs = shocks_intercepts[columns].to_numpy().flatten()
all_coeffs = all_coeffs[~np.isnan(all_coeffs)]
all_coeffs = all_coeffs[(all_coeffs>=0)&(all_coeffs<1)]

bin_width=0.01
bins = np.arange(start=0, stop=1+bin_width, step=bin_width)

fig, ax = plt.subplots()

ax.hist(all_coeffs, bins=bins, color='b')
ax.set_xlabel('Cross Correlation [0,1)')
ax.set_ylabel('Count')

plt.show()

# %%
import pandas as pd

shocks_closest = pd.DataFrame(index=shocks_intercepts.index)

sc_labels      = [col.split('_')[0] for col in shocks_intercepts if '_coeff' in col]

dist_columns   = {f'{sc}_dist_diff': np.nan for sc in sc_labels if sc!='OMNI'}
time_columns   = {f'{sc}_time_diff': np.nan for sc in sc_labels if sc!='OMNI'}
shocks_closest = shocks_closest.assign(**{**{'detector': None}, **dist_columns, **time_columns, **{'closest': None}})

# %%

#distance_choice = 'crow'
distance_choice = 'path'

for index, shock in shocks_intercepts.iterrows():
    sc_xs = shock[[f'{sc}_r_x_GSE' for sc in sc_labels]]
    sc_ys = shock[[f'{sc}_r_y_GSE' for sc in sc_labels]]
    sc_zs = shock[[f'{sc}_r_z_GSE' for sc in sc_labels]]

    sc_positions = {}
    for sc in sc_labels:

        x = sc_xs.get(f'{sc}_r_x_GSE',np.nan)
        y = sc_ys.get(f'{sc}_r_y_GSE',np.nan)
        z = sc_zs.get(f'{sc}_r_z_GSE',np.nan)
        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            continue
        sc_positions[sc] = np.array([x,y,z])

    if 'OMNI' not in sc_positions:
        continue

    shocks_closest.at[index,'detector'] = shock['spacecraft']
    detector_pos = np.array(shock[['r_x_GSE','r_y_GSE','r_z_GSE']])
    bs_to_l1 = detector_pos-sc_positions['OMNI']

    sc_distances = {}
    for sc in sc_positions:
        if sc == 'OMNI':
            sc_distances[sc] = np.linalg.norm(detector_pos-sc_positions['OMNI'])
            shocks_closest.at[index,f'{sc}_dist_diff'] = sc_distances[sc]
            shocks_closest.at[index,f'{sc}_time_diff'] = (shock['OMNI_time']-index).total_seconds()
            continue

        bs_to_sc = sc_positions[sc]-sc_positions['OMNI']
        if distance_choice == 'crow':
            sc_distances[sc] = np.linalg.norm(bs_to_sc)
        elif distance_choice == 'path':
            try:
                if sc=='ACE' and np.dot(bs_to_sc,bs_to_l1)<-150:
                    print(shock['spacecraft'],detector_pos)
                sc_distances[sc] = np.dot(bs_to_sc,bs_to_l1)/np.linalg.norm(bs_to_l1)
            except:
                sc_distances[sc] = 0


        shocks_closest.at[index,f'{sc}_dist_diff'] = sc_distances[sc]
        shocks_closest.at[index,f'{sc}_time_diff'] = (shock[f'{sc}_time']-shock['OMNI_time']).total_seconds()

    sc_distances_no_omni = sc_distances.copy()
    del sc_distances_no_omni['OMNI']
    shocks_closest.at[index,'closest'] = min(sc_distances_no_omni, key=sc_distances_no_omni.get)

# %%


colour_dict = {
    'OMNI': 'orange',
    'C1':   'blue',
    'C2':   'cornflowerblue',
    'C3':   'lightskyblue',
    'C4':   'lightblue',
    'THA':  'forestgreen',
    'THB':  'seagreen',
    'THC':  'mediumseagreen',
    'THD':  'lightgreen',
    'THE':  'palegreen',
    'ACE':  'darkviolet',
    'DSC':  'deeppink',
    'GEO':  'teal',
    'IMP8': 'crimson',
    'WIND': 'magenta'
}

#plot_choice = 'closest'
plot_choice = 'all'

correlation_limit = 0.5

distances  = []
times      = []
coeffs     = []
spacecraft = []
detector   = []
dist_scale = []
time_scale = []

for index, shock in shocks_closest.iterrows():
    if shock['detector'] is None:
        continue
    if plot_choice == 'closest':
        sc_closest = shock['closest']
        if sc_closest is None:
            continue
        sc_list = (sc_closest,)
    elif plot_choice == 'all':
        sc_list = [sc for sc in sc_labels if sc!='OMNI']

    for sc in sc_list:
        if sc==shock['detector']:
            continue

        corr_coeff = shocks_intercepts.loc[index,f'{sc}_coeff']
        if np.isnan(corr_coeff) or corr_coeff<correlation_limit:
            #==1 prevents exact matches
            continue

        detector.append(shock['detector'].upper())
        spacecraft.append(sc)
        distances.append(shock[f'{sc}_dist_diff'])
        times.append(shock[f'{sc}_time_diff'])
        coeffs.append(corr_coeff)
        dist_scale.append(shock['OMNI_dist_diff'])
        time_scale.append(shock['OMNI_time_diff'])

distances = np.array(distances)
times = np.array(times)
coeffs = np.array(coeffs)
spacecraft = np.array(spacecraft)
detector = np.array(detector)
dist_scale = np.array(dist_scale)
time_scale = np.array(time_scale)

# %%
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

colouring = 'coeff'
show_best_fit = True
closish = abs(distances)<100

xs = distances[closish]
ys = times[closish]/60

fig, ax = plt.subplots()
ax.axhline(0,c='k',ls=':')

if colouring == 'coeff':

    scatter = ax.scatter(xs, ys, c=coeffs[closish], cmap='coolwarm', vmin=correlation_limit, vmax=1, s=1)
    cbar = plt.colorbar(scatter)
    cbar.set_label('correlation coefficient')

elif colouring in ('spacecraft','detector'):
    if colouring == 'spacecraft':
        spacecraft_counts = Counter(spacecraft[closish])
        spacecraft_series = pd.Series(spacecraft[closish])
    else:
        spacecraft_counts = Counter(detector[closish])
        spacecraft_series = pd.Series(detector[closish])

    colours = spacecraft_series.map(colour_dict).fillna('k').to_numpy()
    scatter = ax.scatter(xs, ys, c=colours, s=1)

    legend_elements = [Line2D([0], [0], marker='o', color=colour, label=f'{label}: {spacecraft_counts.get(label, 0)}', markersize=1,
                          linestyle='None')
                   for label, colour in colour_dict.items() if spacecraft_counts.get(label, 0) > 0
    ]
    plt.legend(handles=legend_elements, fontsize=6, loc='upper left', bbox_to_anchor=(1.01, 1.0))

else:
    ax.scatter(distances[closish], times[closish]/60, c='k', s=1)

if show_best_fit:
    slope, intercept = np.polyfit(xs, ys, 1)
    y_pred = slope * xs + intercept
    ax.plot(xs,y_pred,c='r',lw=1,ls='--')

    if intercept<0:
        sign = '-'
    else:
        sign = '+'
    ax.text(np.min(xs),np.max(ys),f'$\\Delta t$ = {slope:.2f}$\\Delta r$ {sign} {abs(intercept):.2f} mins')


if distance_choice=='path':
    ax.set_xlabel(r'($r_{SC}$ - $r_{BSN}$) $\cdot$ ($r_{L1}$ - $r_{BSN}$)/|$r_{L1}$ - $r_{BSN}$| [$R_E$]')
elif distance_choice=='crow':
    ax.set_xlabel(r'|$r_{SC}$ - $r_{BSN}$| [$R_E$]')
ax.set_ylabel(r'$t_{SC}$ - $t_{OMNI}$ [mins]')
ax.set_title(f'Closest spacecraft with coeff$\\geq${correlation_limit:.1f}, N={np.sum(closish)}')

plt.show()
plt.close()
