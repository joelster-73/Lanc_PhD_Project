# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %%
import os
import numpy as np
import itertools as it

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from src.config import MSH_DIR, OMNI_DIR
from src.analysing.comparing import difference_series
from src.processing.reading import import_processed_data

from src.plotting.comparing.parameter import compare_columns
from src.plotting.space_time import plot_orbit_msh
from src.plotting.utils import calculate_bins
from src.plotting.formatting import data_string

sample_interval = '5min'
data_pop = 'field_only'

# Solar wind data
omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_sw    = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

msh_spacecraft = 'combined'
msh_dir = os.path.join(MSH_DIR, data_pop, sample_interval)
df_merged = import_processed_data(msh_dir, f'msh_times_{msh_spacecraft}.cdf')

# Erroneous values
if msh_spacecraft=='combined':
    sc_keys = ('c1','tha','thb','thc','thd','the')
else:
    sc_keys = (msh_spacecraft,)
for sc in sc_keys:

    # Add uncertainty
    df_merged[f'Delta B_theta_{sc}'] = np.abs(difference_series(df_merged['B_clock_sw'],df_merged[f'B_clock_{sc}'],unit='rad'))
    df_merged.attrs['units'][f'Delta B_theta_{sc}'] = 'rad'

    df_merged[f'Delta Bz_{sc}'] = df_merged[f'B_z_GSM_{sc}']/df_merged['B_z_GSM_sw']
    df_merged[f'Delta Bz_{sc}'] = df_merged[f'Delta Bz_{sc}'].replace([np.inf, -np.inf], np.nan)
    df_merged.attrs['units'][f'Delta Bz_{sc}'] = '1'


# %% Orbits

n_cols = min(3,len(sc_keys))
n_rows = round(len(sc_keys)/n_cols)

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4*(n_cols+1),6*(n_rows+1)), dpi=400)

for i, sc_key in enumerate(sc_keys):

    row = i % n_rows
    col = i // n_rows

    if len(sc_keys)==1:
        ax = axs
    else:
        ax = axs[row,col]

    if sample_interval=='5min':
        title = f'{sc_key}: {5*len(df_merged[f"r_F_{sc_key}"].dropna()):,} mins'
    elif sample_interval=='1min':
        title = f'{sc_key}: {len(df_merged[f"r_F_{sc_key}"].dropna()):,} mins'

    plot_orbit_msh(df_merged, sc_keys=sc_key, title=title, fig=fig, ax=ax, return_objs=True)

plt.tight_layout()

# %%
from src.plotting.config import colour_dict

n_rows = len(sc_keys)
n_cols = 1

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*(n_cols+1),2*(n_rows+1)), dpi=400, sharex=True)

for i, sc_key in enumerate(sc_keys):

    if len(sc_keys)==1:
        ax = axs
    else:
        ax = axs[i]

    if i >= 2:
        ax.sharey(axs[1])

    years = df_merged[f'r_F_{sc_key}'].dropna().index.year.to_numpy()
    bins = calculate_bins(years,1)
    counts, _ = np.histogram(years, bins=bins)

    if sample_interval=='5min':
        label = f'{sc_key}: {5*len(years):,} mins'
    elif sample_interval=='1min':
        label = f'{sc_key}: {len(years):,} mins'

    ax.bar(bins[:-1]+0.5, counts, width=1, color=colour_dict.get(sc_key.upper(),'k'), label=label)
    ax.legend(loc='upper left')

plt.subplots_adjust(wspace=0, hspace=0)
#plt.tight_layout()

# %%
import pandas as pd
all_times = np.ones(len(df_merged),dtype=bool)
total = 0
for pair in it.combinations(sc_keys, 2):
    times = df_merged[f'r_y_GSE_{pair[0]}']*df_merged[f'r_y_GSE_{pair[1]}']<0
    all_times |= all_times
    opposite = np.nansum(times)
    if sample_interval=='5min':
        opposite *= 5
    total += opposite
    print(pair, opposite)
days = pd.to_datetime(df_merged.index).strftime('%d-%m-%y').unique()
print('total',total,'mins')
print('Unique days:',len(days))

# %% Bias

parameters_to_plot = ('B_avg','B_z_GSM','V_flow','V_x_GSE','E_R','E_y_GSM','n_p','P_flow','MA','AE')
param_width = {'B_avg': 1, 'B_z_GSM': 1, 'V_flow': 50, 'V_x_GSE': 50, 'n_p': 1, 'P_flow': 1, 'MA': 2, 'AE': 50, 'E_mag': 1, 'S_mag': 2}


n_params = len(parameters_to_plot)
n_cols = 2
n_rows = round(n_params/n_cols)

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6*(n_cols+1),4*(n_rows+1)), dpi=400)

for i, param in enumerate(parameters_to_plot):

    col = i % 2
    row = i // 2

    ax = axs[row,col]

    for j in range(2):
        if j==0:
            mask = np.ones(len(df_sw),dtype=bool)
            colour = 'orange'
            axis = ax
        else:
            mask = df_merged.index
            colour = 'blue'
            axis = ax.twinx()

        series = df_sw.loc[mask,param].dropna()
        if param=='MA':
            series = series.loc[series<200]

        axis.hist(series, bins=calculate_bins(series,param_width.get(param)), histtype='step', edgecolor=colour)
        axis.set_yscale('log')
    ax.set_xlabel(param)
    #ax.set_yscale('log')

plt.tight_layout()
plt.show()


# %%


independent = 'E_R'
ind_source  = 'sw'

dependent   = 'AE_17m'
dep_source  = 'pc'

group_param = 'Delta B_theta'
#group_param = 'Delta Bz'
#group_param = 'MA_sw'



ind_param       = '_'.join((independent,ind_source))
if ind_source in ('sw','pc'):
    ind_param_err = None
else:
    ind_param_err   = '_'.join((independent,'unc',ind_source))

if ind_source in ('sw','pc'):
    ind_param_count = None
elif '_GS' in independent:
    field, _, coords = independent.split('_')
    ind_param_count = '_'.join((field,coords,'count',ind_source))
else:
    ind_param_count = '_'.join((independent,'count',ind_source))

dep_param       = '_'.join((dependent,dep_source))
if dep_source in ('sw','pc'):
    dep_param_err = None   ### CHANGE TO USE B UNC
else:
    dep_param_err   = '_'.join((dependent,'unc',dep_source))

if dep_source in ('sw','pc'):
    dep_param_count = None ### CHANGE TO USE IMF/PLASMA COUNTS
elif '_GS' in dependent:
    field, _, coords = independent.split('_')
    dep_param_count = '_'.join((field,coords,'count',dep_source))
else:
    dep_param_count = '_'.join((dependent,'count',dep_source))



z_unit = df_merged.attrs['units'].get(group_param,'')

if z_unit in ('rad','deg','°'):
    z_unit_str = ' °'
elif z_unit is not None and z_unit not in ('1','NUM',''):
    z_unit_str = f' {z_unit}'
else:
    z_unit_str = ''


invert = False
mask = np.ones(len(df_merged),dtype=bool)
omni_mask = np.ones(len(df_sw),dtype=bool)

if 'B_' in ind_param:
    mask = df_merged[ind_param]<0
    omni_mask = df_sw[independent]<0
    bin_step = 2
    invert = True
elif 'V_' in ind_param and ind_param!='V_flow':
    mask = df_merged[ind_param]<0
    omni_mask = df_sw[independent]<0
    invert = True
    bin_step = 50
elif 'E_' in ind_param:
    mask = df_merged[ind_param]>0
    omni_mask = df_sw[independent]>0
    bin_step = 2

mask &= ~np.isnan(df_merged[dep_param])
mask &= ~np.isnan(df_merged[group_param])

omni_mask &= ~np.isnan(df_sw[dependent])

df_masked      = df_merged.loc[mask]
df_omni_masked = df_sw.loc[omni_mask]

median = np.percentile(df_masked[group_param].dropna().to_numpy(),50)

if group_param=='Delta B_theta':
    edges = [median]
    bin_width = np.pi/36
    z_labels = [f'${data_string(group_param)}$<{np.degrees(edges[0]):.1f}{z_unit_str}',
                f'${data_string(group_param)}$$\\geq${np.degrees(edges[0]):.1f}{z_unit_str}']

elif group_param=='Delta Bz':
    edges = [0]
    bin_width = 1
    z_labels = [f'$sgn({data_string(group_param)})$<0{z_unit_str}',
                f'$sgn({data_string(group_param)})$$\\geq$0{z_unit_str}']

elif group_param=='MA_sw':
    edges = [median]
    bin_width = 1
    z_labels = [f'${data_string(group_param)}$<{edges[0]:.1f}{z_unit_str}',
                f'${data_string(group_param)}$$\\geq${edges[0]:.1f}{z_unit_str}']

fig = plt.figure(figsize=(12, 10), dpi=200)
# nrows, ncols
gs = gridspec.GridSpec(5, 6, figure=fig, width_ratios=[1, 1, 1, 1.15, 1.1, 0.2])


# gs[rows occupied, columns occupied]
ax_tl  = fig.add_subplot(gs[0:2, 0:3])
ax_tr  = fig.add_subplot(gs[0:3, 3])
ax_tr2 = fig.add_subplot(gs[0:3, 4], sharex=ax_tr, sharey=ax_tr)
cax    = fig.add_subplot(gs[0:3, -1]) # for colourbar

ax_bl  = fig.add_subplot(gs[2:4, 0:3], sharex=ax_tl)
ax_bl2 = fig.add_subplot(gs[4, 0:3], sharex=ax_tl)
ax_br  = fig.add_subplot(gs[3, 3:6])
ax_br2 = fig.add_subplot(gs[4, 3:6])

omni_colour = 'k'
full_colour = 'b'

if independent == dependent:
    ax_tl.axline((0,0),slope=1,c='k',ls=':')

elif dep_source!='msh' and ind_source!='msh':

    _ = compare_columns(df_omni_masked, independent, dependent, display='rolling', window_width=bin_step, brief_title='Full OMNI', data_colour=omni_colour, region='sem', fig=fig, ax=ax_tl, return_objs=True)

    df_omni_param = df_omni_masked.loc[:,independent]
    ax_bl2.hist(df_omni_param, bins=calculate_bins(df_omni_param,bin_step), color=omni_colour)

_ = compare_columns(df_masked, ind_param, dep_param, col1_err=ind_param_err, col1_counts=ind_param_count, col2_err=dep_param_err, col2_counts=dep_param_count, display='rolling', window_width=bin_step, data_colour=full_colour, error_colour=full_colour, region='sem', fig=fig, ax=ax_tl, return_objs=True)

_ = compare_columns(df_masked, ind_param, dep_param, col1_err=ind_param_err, col1_counts=ind_param_count, col2_err=dep_param_err, col2_counts=dep_param_count, col3=group_param, display='rolling_multiple', zs_edges=edges, window_width=bin_step, region='sem', want_legend=False, fig=fig, ax=ax_bl, return_objs=True)


# colour meshses
cms = []

for i, (axis, group_region, colour, label) in enumerate(zip((ax_tr, ax_tr2), ('low', 'high'), ('c','m'), z_labels)):

    if group_region=='low':
        filter_mask = df_masked[group_param]<edges[0]
    elif group_region=='high':
        filter_mask = df_masked[group_param]>=edges[0]

    _, cm = plot_orbit_msh(axis, df_masked.loc[filter_mask], title=label, colourbar=False)
    cms.append(cm)

    df_msh_param = df_masked.loc[filter_mask,ind_param]
    df_msh_years = df_masked.loc[(~df_masked[ind_param].isna())&(filter_mask)].index.year.to_numpy()

    # Counts each grouping bin
    bins = calculate_bins(df_masked.loc[filter_mask,group_param],bin_width)
    if bins[-2]<edges[0]<bins[-1]:
        bins[-1] = edges[0]
    elif bins[1]>edges[0]>bins[0]:
        bins[0] = edges[0]

    ax_br.hist(df_masked.loc[filter_mask,group_param], bins=bins, color=colour, label=label)
    if group_param=='Delta Bz':
        ax_br.set_xscale('symlog', linthresh=10)

    # Counts each independent bin
    bins = calculate_bins(df_msh_param,bin_step)
    counts, _ = np.histogram(df_msh_param, bins=bins)

    ax_bl2.bar(bins[:-1] + (i+0.5)*bin_step/2, counts, width=bin_step/2, color=colour, alpha=0.9)

    # Counts each year
    bins = calculate_bins(df_msh_years,1)
    counts, _ = np.histogram(df_msh_years, bins=bins)

    ax_br2.bar(bins[:-1] + (i-0.5)*0.5, counts, width=0.5, color=colour, label=f'{np.sum(filter_mask):,} counts')



ax_br.axvline(x=edges[0],c='k',ls='--')
if z_unit =='rad':
    formatter = FuncFormatter(lambda val, pos: f'{np.degrees(val):.0f}°')
    ax_br.xaxis.set_major_formatter(formatter)

vmin = min(cm.get_array().min() for cm in cms)
vmax = max(cm.get_array().max() for cm in cms)

for cm in cms:
    cm.set_clim(vmin, vmax)

# One shared colorbar
cbar = fig.colorbar(cms[0], cax=cax)
cbar.set_label('Counts')


cax.yaxis.set_ticks_position('left')

ax_tl.set_xlabel(None)
ax_tr.set_xlabel(None)
ax_tr.set_ylabel(None)
ax_tr2.set_ylabel('YZ', rotation='horizontal', ha='right')

ax_tr2.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,
    labelleft=False)

ax_bl2.set_yscale('log')
ax_bl2.set_ylabel('counts')
ax_br.set_yscale('log')
ax_br2.set_yscale('log')

ax_br.legend(loc='upper right', fontsize=8)
ax_br2.legend(loc='upper right', fontsize=8)

# if invert:
#     ax_bl.invert_xaxis()


plt.tight_layout()
plt.show()





