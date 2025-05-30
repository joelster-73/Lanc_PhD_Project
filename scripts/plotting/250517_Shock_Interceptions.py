# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""
from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data

shocks = import_processed_data(PROC_SHOCKS_DIR)

# %%
import matplotlib.pyplot as plt
import pandas as pd

total_shocks = len(shocks)
columns_to_count = [col for col in shocks.columns if shocks.attrs['units'][col]=='datetime']
spacecraft = [col.split('_')[0] for col in columns_to_count]
rows_with_data = shocks[columns_to_count].notna().any(axis=1)


# %%
counts = shocks[columns_to_count].notna().sum()
positions = range(len(columns_to_count))

fig, ax = plt.subplots()

bars = ax.bar(positions, counts.values, color='b', edgecolor='black')
ax.bar_label(bars, labels=counts.values, label_type='edge', fontsize=10)

ax.set_xticks(ticks=positions, labels=spacecraft, fontsize=7)
ax.set_title(f'{rows_with_data.sum()} shocks in total')

ax.set_xlabel('Spacecraft that recorded shock')
ax.set_ylabel('Counts')

plt.show()
plt.close()

# %%
row_counts = []
dict_of_counts = {key: [] for key in range(10)}

for index, shock in shocks.iterrows():
    shock_sc = shocks.loc[index,'spacecraft']
    row_count = 0
    for col in columns_to_count:
        if shock_sc not in col and 'OMNI' not in col and not pd.isnull(shock[col]):
            row_count += 1
    row_counts.append(row_count)
    dict_of_counts[row_count].append(index)

row_counts = pd.Series(row_counts).value_counts().sort_index()

fig, ax = plt.subplots()

bars = ax.bar(row_counts.index, row_counts.values, color='b', edgecolor='black')
ax.bar_label(bars, labels=row_counts.values, label_type='edge', fontsize=10)

ax.set_xticks(ticks=row_counts.index, labels=row_counts.index)
ax.set_title(f'{rows_with_data.sum()} shocks in total')

ax.set_xlabel('Number of spacecraft that recorded a shock (exc. self)')
ax.set_ylabel('Counts')

plt.show()
plt.close()



# %%
import numpy as np

columns_to_add = [f'{sc}_{suffix}' for sc in spacecraft for suffix in ['time_diff', 'dist_diff']] + ['closest']
shock_diffs = pd.DataFrame(columns=columns_to_add,index=shocks.index)

#choice = 'weighted'
choice = 'closest'
#choice = 'all'

# plot_type = 'hist'
plot_type = 'scatter'

x_values     = []
y_values     = []

for index, shock in shocks.iterrows():
    L1_time = index
    L1_sc   = shock['spacecraft']
    L1_pos  = np.array(shock[['r_x_GSE','r_y_GSE','r_z_GSE']])

    OMNI_time = shock['OMNI_time']
    OMNI_pos  = np.array(shock[['OMNI_r_x_GSE','OMNI_r_y_GSE','OMNI_r_z_GSE']])

    if pd.isnull(OMNI_time):
        continue

    for sc in spacecraft:
        if sc == 'OMNI':
            shock_diffs.at[index,f'{sc}_time_diff'] = (OMNI_time-L1_time).total_seconds()
            shock_diffs.at[index,f'{sc}_dist_diff'] = np.linalg.norm(OMNI_pos-L1_pos)

        position = np.array(shock[[f'{sc}_r_x_GSE',f'{sc}_r_y_GSE',f'{sc}_r_z_GSE']])
        time = shock[f'{sc}_time']
        if pd.isnull(time):
            continue

        shock_diffs.at[index,f'{sc}_time_diff'] = (OMNI_time-time).total_seconds()
        shock_diffs.at[index,f'{sc}_dist_diff']  = np.linalg.norm(OMNI_pos-position)

    distance_cells = shock_diffs.loc[index,[f'{sc}_dist_diff' for sc in spacecraft if sc!='OMNI']].dropna()
    if not distance_cells.empty:
        shock_diffs.at[index,'closest'] = distance_cells.idxmin().split('_')[0]
    else:
        shock_diffs.at[index,'closest'] = None


shock_diffs_closest = shock_diffs[shock_diffs['closest'].notna()]
if plot_type == 'scatter':

    if choice == 'all':
        y_values = np.array(y_values) / 60

        plt.scatter(np.array(x_values), np.array(y_values)/60, s=1, label='sc_time-omni_time')
        plt.ylim(-60,60)
        plt.xlim(0,75)

    elif choice == 'closest':

        x_values = shock_diffs_closest.apply(lambda row: row[f'{row["closest"]}_dist_diff'], axis=1)
        y_values = shock_diffs_closest.apply(lambda row: row[f'{row["closest"]}_time_diff'], axis=1)

        plt.scatter(x_values, y_values, s=1, label='sc_time-omni_time')

# elif plot_type == 'hist':

#     nbins = int((np.max(time_differences)-np.min(time_differences))/5)
#     counts, bins, _ = plt.hist(time_differences, bins=nbins)
#     plt.axvline(np.mean(time_differences),c='r',ls='--',label=f'{np.mean(time_differences):.3g}')
#     plt.xlim(-120,120)
#     plt.xlabel('t_sc - t_OMNI (mins)')

plt.legend()
plt.title(choice)
plt.show()

# %%

