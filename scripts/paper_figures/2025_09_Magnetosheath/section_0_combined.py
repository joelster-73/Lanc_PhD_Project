# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %% Import
import os
import numpy as np
import itertools as it

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter, MultipleLocator

from src.config import MSH_DIR, OMNI_DIR
from src.analysing.comparing import difference_series
from src.processing.reading import import_processed_data

from src.plotting.comparing.parameter import compare_columns
from src.plotting.space_time import plot_orbit_msh
from src.plotting.utils import calculate_bins
from src.plotting.formatting import data_string, create_label
from src.plotting.config import colour_dict

sample_interval = '1min'
data_pop = 'field_only'

# Solar wind data
omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_sw    = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

msh_dir = os.path.join(MSH_DIR, data_pop, sample_interval)
df_merged = import_processed_data(msh_dir, 'msh_times_combined.cdf')

if sample_interval=='1min':
    data_type = 'mins'
else:
    data_type = 'counts'

# Rotation of clock angle
# Ignoring sw uncertainty for time being
df_merged['Delta B_theta_msh'] = np.abs(difference_series(df_merged['B_clock_sw'],df_merged['B_clock_msh'],unit='rad'))
not_nan = ~np.isnan(df_merged['Delta B_theta_msh'])
df_merged.loc[not_nan,'Delta B_theta_unc_msh'] = df_merged.loc[not_nan,'B_clock_unc_msh']
df_merged.attrs['units']['Delta B_theta_msh'] = 'rad'

# Reversal of Bz
# Not interested in error
df_merged['Delta Bz_msh'] = df_merged['B_z_GSM_msh']/df_merged['B_z_GSM_sw']
df_merged['Delta Bz_msh'] = df_merged['Delta Bz_msh'].replace([np.inf, -np.inf], np.nan)
df_merged.attrs['units']['Delta Bz_msh'] = '1'


# %% Orbits

sc_keys = ('c1','m1','tha','thc','thd','the')

n_cols = min(3,len(sc_keys))
n_rows = round(len(sc_keys)/n_cols)

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4*(n_cols+1),6*(n_rows+1)), dpi=400)

for i, sc_key in enumerate(sc_keys):

    df_sc = import_processed_data(msh_dir, f'msh_times_{sc_key}.cdf')

    row = i % n_rows
    col = i // n_rows

    if len(sc_keys)==1:
        ax = axs
    elif n_rows==1:
        ax = axs[col]
    else:
        ax = axs[row,col]

    title = f'{sc_key}: {len(df_sc[f"B_avg_{sc_key}"].dropna()):,} {data_type}'

    plot_orbit_msh(df_sc, sc_keys=sc_key, title=title, fig=fig, ax=ax, return_objs=True)

plt.tight_layout()

# %% Years

# Each spacecraft on a separate row

sc_keys = ('c1','m1','tha','thb','thc','thd','the')


n_rows = len(sc_keys)
n_cols = 1

fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(5*(n_cols+1),2*(n_rows+1)), dpi=400, sharex=True)

for i, sc_key in enumerate(sc_keys):

    df_sc = import_processed_data(msh_dir, f'msh_times_{sc_key}.cdf')

    if len(sc_keys)==1:
        ax = axs
    else:
        ax = axs[i]

    if i >= 3:
        ax.sharey(axs[2])

    years = df_sc[f'B_avg_{sc_key}'].dropna().index.year.to_numpy()
    bins = calculate_bins(years,1)
    counts, _ = np.histogram(years, bins=bins)

    label = f'{sc_key}: {len(years):,} {data_type}'

    ax.bar(bins[:-1]+0.5, counts, width=1, color=colour_dict.get(sc_key.upper(),'k'), label=label)
    ax.legend(loc='upper left')
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'{val:,.0f}'))


plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

# %% Years_One_Axis

# One bar for each spacecraft for each year
# THEMIS combined into one bar

th_keys = ('tha','thb','thc','thd','the')
sc_keys = ('c1','m1','th')

fig, ax = plt.subplots(figsize=(10, 5), dpi=400)

width = 1 / len(sc_keys)  # total group width = 0.8, divided among spacecraft

for i, sc_key in enumerate(sc_keys):

    if sc_key=='th':
        years = []
        for sc in th_keys:
            df_sc = import_processed_data(msh_dir, f'msh_times_{sc}.cdf')
            years.append(df_sc[f'B_avg_{sc}'].dropna().index.year.to_numpy())
        years = np.concatenate(years)
    else:
        df_sc = import_processed_data(msh_dir, f'msh_times_{sc_key}.cdf')
        years = df_sc[f'B_avg_{sc_key}'].dropna().index.year.to_numpy()

    bins = calculate_bins(years, 1)
    counts, _ = np.histogram(years, bins=bins)

    offset = i * width

    ax.bar(bins[:-1] + offset, counts, width=width, color=colour_dict.get(sc_key.upper(), 'k'), label=f'{sc_key}: {len(years):,} {data_type}', align='edge')

# Formatting
ax.legend(loc='upper left', framealpha=1)
ax.set_xlabel('Year')
ax.set_ylabel('Count')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'{val:,.0f}'))

plt.tight_layout()
plt.show()


# %% Bias_Driver

# Shows driver for full OMNI and for times when contemp. MSH

fig, ax = plt.subplots(figsize=(10,6), dpi=400, sharex=True)

omni_col = 'E_R'

for i, (colour, label) in enumerate(zip(('orange','blue'),('All OMNI','Contemp MSH'))):
    mask = np.ones(len(df_sw),dtype=bool)
    if i==1:
        mask = df_merged.index
    # full
    monthly_max = df_sw.loc[mask,omni_col].resample('ME').max()
    rolling_max = monthly_max.rolling(window=6, min_periods=1).mean()

    month_datetimes = monthly_max.index
    fractional_years = month_datetimes.year + (month_datetimes.month - 1)/12

    ax.plot(fractional_years, rolling_max.values, linestyle='-', color=colour, label=label)

y_label = create_label(omni_col, units=df_sw.attrs['units'])

ax.set_ylabel(y_label)
ax.legend(loc='upper left')

plt.show()

# %% Time_Activity

from src.plotting.comparing.space_time import plot_compare_datasets_with_activity

plot_compare_datasets_with_activity(df_sw, df_merged, df_colours=('orange','blue'), df_names=('All OMNI','Contemp MSH'))



# %% Asymmetry
import pandas as pd
all_times = np.ones(len(df_merged),dtype=bool)
total = 0
for pair in it.combinations(sc_keys, 2):
    times = df_merged[f'r_y_GSE_{pair[0]}']*df_merged[f'r_y_GSE_{pair[1]}']<0
    all_times |= all_times
    opposite = np.nansum(times)
    total += opposite
    print(pair, opposite)
days = pd.to_datetime(df_merged.index).strftime('%d-%m-%y').unique()

print(f'total: {total} {data_type}')
print(f'Unique days: {len(days)}')

# %% Bias

parameters_to_plot = ('B_avg','B_z_GSM','V_flow','V_x_GSE','E_R','E_y_GSM','n_p','P_flow','M_A','AE')
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


# %% Overview


independent = 'B_avg'
ind_source  = 'sw'
restrict    = True

dependent   = 'AE_17m'
dep_source  = 'pc'


grouping = 'Delta B_theta'
#group_param = 'Delta Bz_msh'
#group_param = theta_Bn_msh
#group_param = 'MA_sw'

group_source = 'msh'
group_param = f'{grouping}_{group_source}'

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
    z_unit_str = '°'
elif z_unit is not None and z_unit not in ('1','NUM',''):
    z_unit_str = f' {z_unit}'
else:
    z_unit_str = ''


invert = False
mask = np.ones(len(df_merged),dtype=bool)
omni_mask = np.ones(len(df_sw),dtype=bool)

if 'B_' in ind_param and independent!='B_avg':
    mask = df_merged[ind_param]<0
    omni_mask = df_sw[independent]<0
    bin_step = 2
    invert = True
    if restrict:
        limit = -15
        mask &= df_merged[ind_param] >= limit
        omni_mask &= df_sw[independent] >= limit
        bin_step = 1
elif independent=='B_avg':
    bin_step = 2
    if restrict:
        limit = 10
        mask &= df_merged[ind_param] <= limit
        omni_mask &= df_sw[independent] <= limit
        bin_step = 1
elif 'V_' in ind_param and independent!='V_flow':
    mask = df_merged[ind_param]<0
    omni_mask = df_sw[independent]<0
    invert = True
    bin_step = 50
elif 'E_' in ind_param:
    mask = df_merged[ind_param]>0
    omni_mask = df_sw[independent]>0
    bin_step = 2
    if restrict:
        limit = 12
        mask &= df_merged[ind_param] <= limit
        omni_mask &= df_sw[independent] <= limit
        bin_step = 1

mask &= ~np.isnan(df_merged[dep_param])
mask &= ~np.isnan(df_merged[group_param])

omni_mask &= ~np.isnan(df_sw[dependent])

df_masked      = df_merged.loc[mask]
df_omni_masked = df_sw.loc[omni_mask]

median = np.percentile(df_masked[group_param].dropna().to_numpy(),50)

if group_param=='Delta B_theta_msh':
    edges = [median]
    bin_width = np.pi/36
    z_labels = [f'${data_string(grouping)}$<{np.degrees(edges[0]):.1f}{z_unit_str}',
                f'${data_string(grouping)}$$\\geq${np.degrees(edges[0]):.1f}{z_unit_str}']

elif group_param=='Delta Bz_msh':
    edges = [0]
    bin_width = 1
    z_labels = [f'$sgn({data_string(grouping)})$<0{z_unit_str}',
                f'$sgn({data_string(grouping)})$$\\geq$0{z_unit_str}']

elif group_param=='M_A_sw':
    edges = [median]
    bin_width = 1
    z_labels = [f'${data_string(grouping)}$<{edges[0]:.1f}{z_unit_str}',
                f'${data_string(grouping)}$$\\geq${edges[0]:.1f}{z_unit_str}']

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

_ = compare_columns(df_masked, ind_param, dep_param, col1_err=ind_param_err, col1_counts=ind_param_count, col2_err=dep_param_err, col2_counts=dep_param_count, display='rolling', window_width=bin_step, data_colour=full_colour, error_colour=full_colour, line_style=':', region='none', fig=fig, ax=ax_bl, return_objs=True)

_ = compare_columns(df_masked, ind_param, dep_param, col1_err=ind_param_err, col1_counts=ind_param_count, col2_err=dep_param_err, col2_counts=dep_param_count, col3=group_param, display='rolling_multiple', zs_edges=edges, window_width=bin_step, region='sem', want_legend=False, fig=fig, ax=ax_bl, return_objs=True)


# colour meshses
cms = []

for i, (axis, group_region, colour, label) in enumerate(zip((ax_tr, ax_tr2), ('low', 'high'), ('c','m'), z_labels)):

    if group_region=='low':
        filter_mask = df_masked[group_param]<edges[0]
    elif group_region=='high':
        filter_mask = df_masked[group_param]>=edges[0]

    _, _, cm = plot_orbit_msh(df_masked.loc[filter_mask], title=label, colourbar=False, fig=fig, ax=axis, return_objs=True)
    cms.append(cm)

    df_msh_param = df_masked.loc[filter_mask,ind_param]
    df_msh_years = df_masked.loc[(~df_masked[ind_param].isna())&(filter_mask)].index.year.to_numpy()

    # Counts each grouping bin
    df_grouping = df_masked.loc[filter_mask,group_param]
    bins = calculate_bins(df_grouping,bin_width)
    if bins[-2]<edges[0]<bins[-1]:
        bins[-1] = edges[0]
    elif bins[0]<edges[0]<bins[1]:
        bins[0] = edges[0]

    ax_br.hist(df_grouping, bins=bins, color=colour, label=label)
    if group_param=='Delta Bz':
        ax_br.set_xscale('symlog', linthresh=10)

    # Counts each independent bin
    ax_bl2.axhline(y=100, ls=':', lw=1, c='w')
    ax_bl2.hist(df_msh_param, bins=calculate_bins(df_msh_param,bin_step), histtype='step', edgecolor=colour)

    # Counts each year
    bins = calculate_bins(df_msh_years,1)
    counts, _ = np.histogram(df_msh_years, bins=bins)

    ax_br2.bar(bins[:-1] + (i-0.5)*0.5, counts, width=0.5, color=colour, label=f'{np.sum(filter_mask):,} counts')



ax_br.axvline(x=edges[0],c='k',ls='--')
if z_unit =='rad':
    formatter = FuncFormatter(lambda val, pos: f'{np.degrees(val):.0f}°')
    ax_br.xaxis.set_major_formatter(formatter)
    tick_spacing_deg = 30
    ax_br.xaxis.set_major_locator(MultipleLocator(np.radians(tick_spacing_deg)))

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

if invert:
    ax_bl.invert_xaxis()


plt.tight_layout()
plt.show()





