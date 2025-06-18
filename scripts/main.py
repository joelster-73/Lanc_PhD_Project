# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""


#

# %% Importing
from src.config import HELSINKI_DIR

import os

data_file = os.path.join(HELSINKI_DIR,'Helsinki_database.dat')

import pandas as pd
import numpy as np
from src.processing.utils import add_unit
from src.config import R_E

# Assuming data is in a structured format like CSV or similar
headers = ['year','month','day','hour','minute','second','spacecraft',
           'r_x','r_y','r_z','pos_system','type',
           'B_mag_up','B_mag_up_unc','B_x_up','B_x_up_unc','B_y_up','B_y_up_unc','B_z_up','B_z_up_unc',
           'B_mag_dw','B_mag_dw_unc','B_x_dw','B_x_dw_unc','B_y_dw','B_y_dw_unc','B_z_dw','B_z_dw_unc',
           'B_ratio','B_ratio_unc',
           'V_flow_up','V_flow_up_unc','V_x_up','V_x_up_unc','V_y_up','V_y_up_unc','V_z_up','V_z_up_unc',
           'V_flow_dw','V_flow_dw_unc','V_x_dw','V_x_dw_unc','V_y_dw','V_y_dw_unc','V_z_dw','V_z_dw_unc',
           'V_jump','V_jump_unc',
           'n_p_up','n_p_up_unc','n_p_dw','n_p_dw_unc','n_p_ratio','n_p_ratio_unc',
           'T_p_up','T_p_up_unc','T_p_dw','T_p_dw_unc','T_p_ratio','T_p_ratio_unc',
           'V_cs_up','V_cs_up_unc','V_A_up','V_A_up_unc','V_ms_up','V_ms_up_unc','Beta_up','Beta_up_unc',
           'N_x','N_x_unc','N_y','N_y_unc','N_z','N_z_unc','normal_system',
           'theta_Bn','theta_Bn_unc','v_sh','v_sh_unc','M_A','M_A_unc','M_ms','M_ms_unc',
           'radial_vel','interval','res_B','res_p']


df  = pd.read_csv(
    data_file,
    sep=',',          # Tab-separated values
    skiprows=84,        # Skip the first two rows
    header=None        # Initially treat all rows as data (no headers)
)
df.columns = headers
df.attrs = {}
df['spacecraft'] = df['spacecraft'].str.strip()

df['epoch'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
df.set_index('epoch',inplace=True)
df.sort_index(inplace=True)

df.drop(columns=['year','month','day','hour','minute','second'],inplace=True)

for col in ('T_p_up','T_p_up_unc','T_p_dw','T_p_dw_unc'):
    df[col] *= 1000 # converts to K from 10^4K

for col in ('theta_Bn','theta_Bn_unc'):
    df[col] = np.deg2rad(df[col]) # converts to radians

df['interval'] *= 60 # converts to s

df = df[df['spacecraft'].isin(['ACE', 'OMNI', 'Wind', 'DSCOVR', 'Cluster-1', 'Cluster-3', 'Cluster-4'])]
df['spacecraft'] = df['spacecraft'].replace({'Wind': 'WIND', 'DSCOVR': 'DSC', 'Cluster-1': 'C1', 'Cluster-3': 'C3', 'Cluster-4': 'C4'})

position_coordinates = np.unique(df['pos_system'])
if len(position_coordinates) == 1:
    system = position_coordinates[0].strip()
    df.rename(columns={f'r_{comp}': f'r_{comp}_{system}' for comp in ('x','y','z')},inplace=True)
    df.drop(columns=['pos_system'],inplace=True)

normal_coordinates = np.unique(df['normal_system'])
if len(normal_coordinates) == 1:
    system = normal_coordinates[0].strip()
    columns_to_rename = {}
    for comp in ('x','y','z'):
        columns_to_rename[f'N_{comp}'] = f'N_{comp}_{system}'
        columns_to_rename[f'N_{comp}_unc'] = f'N_{comp}_{system}_unc'
        for vec in ('B','V'):
            columns_to_rename[f'{vec}_{comp}_up']     = f'{vec}_{comp}_{system}_up'
            columns_to_rename[f'{vec}_{comp}_dw']     = f'{vec}_{comp}_{system}_dw'
            columns_to_rename[f'{vec}_{comp}_up_unc'] = f'{vec}_{comp}_{system}_up_unc'
            columns_to_rename[f'{vec}_{comp}_dw_unc'] = f'{vec}_{comp}_{system}_dw_unc'

    df.rename(columns=columns_to_rename,inplace=True)

    df.drop(columns=['normal_system'],inplace=True)

radial_vels = np.unique(df['radial_vel'])
if len(radial_vels) == 1:
    df.drop(columns=['radial_vel'],inplace=True)

unit_attrs = {}
for column in df:
    unit_attrs[column] = add_unit(column)
df.attrs['units'] = unit_attrs

for col in ('interval','res_B','res_p'):
    df.attrs['units'][col] = 's'


# %%

import os
from src.config import HELSINKI_DIR
from src.processing.writing import write_to_cdf

output_file = os.path.join(HELSINKI_DIR, 'helsinki_shocks.cdf')

write_to_cdf(df,output_file,reset_index=True)

# %%
from src.config import PROC_CFA_DIR
from src.processing.reading import import_processed_data
from src.processing.cfa.donki import get_donki_shocks

helsinki_shocks = import_processed_data(HELSINKI_DIR)

cfa_shocks = import_processed_data(PROC_CFA_DIR)

donki_shocks = get_donki_shocks()

# %%
from src.processing.speasy.config import colour_dict
import matplotlib.pyplot as plt
from collections import Counter

for name, df in zip(('Helsinki','CFA','Donki'),(helsinki_shocks,cfa_shocks,donki_shocks)):
    spacecraft_counts = Counter(df['spacecraft'].str.upper())

    fig, ax = plt.subplots()
    for i, (sc,count) in enumerate(spacecraft_counts.items()):
        ax.bar(i,count,color=colour_dict.get(sc,'k'),width=1)
        ax.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)  # Add text above bars


    ax.set_xticks(range(len(spacecraft_counts)))
    ax.set_xticklabels(spacecraft_counts.keys())

    ax.set_ylabel('Count')
    ax.set_title(f'Spacecraft Observation Counts in {name}')
    plt.tight_layout()  # Adjust layout for better fit
    plt.show()

# %%

omni_times = helsinki_shocks[helsinki_shocks['spacecraft']=='OMNI'].index
event_list = []
total_rows = len(helsinki_shocks)
for time in omni_times:
    row_number = helsinki_shocks.index.get_loc(time)
    if row_number==0:
        continue
    event_dict = {}
    event_dict['OMNI'] = time

    max_shift = min(row_number+1,5)
    for i in range(1,max_shift):
        prev_time = helsinki_shocks.index[helsinki_shocks.index.get_loc(time) - i]
        prev_sc   = helsinki_shocks.loc[prev_time,'spacecraft']

        if prev_sc in event_dict:
            break
        elif abs(time-prev_time).total_seconds()>5400:
            break
        event_dict[prev_sc] = prev_time

    max_shift = min(total_rows-row_number,5)
    for i in range(1,max_shift):
        next_time = helsinki_shocks.index[helsinki_shocks.index.get_loc(time) + i]
        next_sc   = helsinki_shocks.loc[next_time,'spacecraft']
        if isinstance(next_sc,pd.Series):
            next_sc = next_sc.iloc[0]
        if next_sc in event_dict:
            break
        elif abs(next_time-time).total_seconds()>1800:
            break
        elif next_sc in ('WIND','ACE','DSC'):
            break
        event_dict[next_sc] = next_time

    if len(event_dict)==1:
        continue
    event_list.append(event_dict)

# %%
distances = []
times = []
spacecraft = []

for event in event_list:
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
    print(cluster_time)

    time_diff = (cluster_time - omni_time).total_seconds()/60
    dist_diff = clus_pos[0] - omni_pos[0]
    dist_diff = np.linalg.norm(clus_pos-omni_pos)

    distances.append(dist_diff)
    times.append(time_diff)
    spacecraft.append(cluster)


# %%
from src.plotting.shocks import plot_time_differences
from src.analysing.fitting import straight_best_fit
from src.processing.speasy.config import colour_dict
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
ax.set_xlabel(r'|$r_{SC}$ - $r_{BSN}$| [$R_E$]')

if False:
    ax.set_xlabel(r'$X_{sc}$ - $X_{BSN}$ [$R_E$]')
    ax.invert_xaxis()
ax.set_ylabel(r'$t_{SC}$ - $t_{OMNI}$ [mins]')
ax.set_title(f'{len(distances)} shock events')
plt.tight_layout()

plt.show()
plt.close()

# %%

omni_distances = []
omni_times = []
l1_spacecraft = []
for event in event_list:
    omni_time = event['OMNI']
    omni_pos = helsinki_shocks.loc[event['OMNI'], ['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
    if omni_pos[0] > 5000:
        continue
    for sc, time in event.items():
        if sc not in ('WIND','ACE','DSC'):
            continue

        l1_pos   = helsinki_shocks.loc[event[sc],['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()


        time_diff = (omni_time - time).total_seconds()/60
        dist_diff = l1_pos[0] - omni_pos[0]
        dist_diff = np.linalg.norm(l1_pos-omni_pos)

        omni_distances.append(dist_diff)
        omni_times.append(time_diff)
        l1_spacecraft.append(sc)

omni_times = np.array(omni_times)
fig, ax = plt.subplots()

step = 5
bin_edges = np.arange(np.floor(np.min(omni_times)/step)*step,np.ceil(np.max(omni_times/step)*step),step)


ax.hist(omni_times, bin_edges, color='orange')
ax.set_xlabel('Time differences between upstream and OMNI [mins]')
ax.set_ylabel('Counts / 5mins')
ax.set_title(f'{len(omni_times)} time differences')
plt.show()