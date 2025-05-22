# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""

# %%
import pandas as pd
from datetime import datetime

start_date = datetime(2000,1,1)
end_date = datetime(2023,12,31)
date_range = pd.date_range(start=start_date, end=end_date, freq='1min')
df = pd.DataFrame(index=date_range)

from src.config import PROC_OMNI_DIR, PROC_CLUS_DIR_1MIN
from src.processing.themis.config import PROC_THEMIS_DIRECTORIES
from src.processing.reading import import_processed_data
from src.coordinates.spatial import calc_bs_pos

# Needs to be BSN, re-process OMNI and add these columns, r_x_BSN etc.

omni = import_processed_data(PROC_OMNI_DIR, date_range=(start_date,end_date))

df['r_x_OMNI'] = omni['r_x_BSN']
df['r_y_OMNI'] = omni['r_y_BSN']
df['r_z_OMNI'] = omni['r_z_BSN']
df['B_avg_OMNI'] = omni['B_avg']
df['p_flow_OMNI'] = omni['p_flow']
df['v_x_GSE_OMNI'] = omni['v_x_GSE']
df['v_y_GSE_OMNI'] = omni['v_y_GSE']
df['v_z_GSE_OMNI'] = omni['v_z_GSE']

del omni

spacecraft = ('C1','THA','THB','THC','THD','THE')

for sc in spacecraft:
    if sc == 'C1':
        temp_df = import_processed_data(PROC_CLUS_DIR_1MIN)
    else:
        th_dir = PROC_THEMIS_DIRECTORIES.get(sc.lower())
        temp_df = import_processed_data(th_dir)

    df[f'r_x_GSE_{sc}'] = temp_df['r_x_GSE']
    df[f'r_y_GSE_{sc}'] = temp_df['r_y_GSE']
    df[f'r_z_GSE_{sc}'] = temp_df['r_z_GSE']
    df[f'B_avg_{sc}']   = temp_df['B_avg']

    #df[f'r_bs_diff_{sc}'] = calc_bs_pos(df, sc_key=sc, time_col='index')['r_bs_diff']

del temp_df



# %%
from src.processing.speasy.config import colour_dict
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots()

B_not_nan  = True
R_upstream = True
Out_BS     = False


colour = colour_dict.get('OMNI')
mask = (df['r_x_OMNI'].notna())
if B_not_nan:
    mask &= (df['B_avg_OMNI'].notna())
data = df[mask]
ax.scatter(data.index, data['r_x_OMNI'], s=0.005, alpha=1, color=colour, edgecolor='face', label=f'OMNI: {len(data):,}')


data_sets = ('C1','THB','THC','THD','THE','THA')
for key in data_sets:
    r_column = f'r_x_GSE_{key}'
    if r_column not in df:
        print(f'{key} not in dataframe.')
        continue

    colour = colour_dict.get(key)
    mask = (df[f'r_x_GSE_{key}'].notna())
    if B_not_nan:
        mask &= (df[f'B_avg_{key}'].notna())
    if R_upstream:
        mask &= (df[f'r_x_GSE_{key}']>10)
    if Out_BS:
        mask &= (df[f'r_bs_diff_{key}']>0)

    data = df[mask]
    ax.scatter(data.index, data[f'r_x_GSE_{key}'], s=0.005, alpha=1, color=colour, edgecolor='face', label=f'{key}: {len(data):,}')

#ax.axhline(0, c='k', lw=2)
ax.axhline(10, c='k', lw=1)

ax.set_xlabel('Time')
ax.set_ylabel('X GSE [Re]')
ax.set_ylim(bottom=0, top=65)

ax.legend(loc='upper left')
ax.set_title('Data Distributions')
plt.show()
plt.close()

# %%
from src.processing.speasy.config import colour_dict
import matplotlib.pyplot as plt
import numpy as np

B_not_nan  = True
R_upstream = True
Out_BS     = False

fig, ax = plt.subplots()

data_sets = ('C1', 'THA', 'THB', 'THC', 'THD', 'THE')
yearly_counts = {key: [] for key in data_sets}

years = sorted(df.index.year.unique())
year_indices = np.arange(len(years))
bar_width = 1 / len(data_sets)

for i, key in enumerate(data_sets):
    if f'r_x_GSE_{key}' not in df:
        print(f'{key} not in dataframe.')
        continue

    mask = (df[f'r_x_GSE_{key}'].notna())
    if B_not_nan:
        mask &= (df[f'B_avg_{key}'].notna())
    if R_upstream:
        mask &= (df[f'r_x_GSE_{key}'] > 10)
    if Out_BS:
        mask &= (df[f'r_bs_diff_{key}'] > 0)

    data = df[mask]
    yearly_data = data.groupby(data.index.year).size()
    yearly_counts[key] = yearly_data

    offset = i * bar_width + bar_width/2
    x_positions = year_indices + offset

    colour = colour_dict.get(key, None)
    ax.bar(x_positions, [yearly_data.get(year, 0) for year in years],
           width=bar_width, label=f'{key}: {yearly_data.sum():,}', alpha=0.7, color=colour)

ax.set_xticks(year_indices)
ax.set_xticklabels(years, rotation=90, ha='center')
ax.set_xlim(-0.5, len(years) - 0.5)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))

ax.set_xlabel('Year')
ax.set_ylabel('Number of Minutes')
ax.set_title('Yearly Data Counts per Spacecraft')
ax.legend(loc='upper left')

plt.show()
plt.close()