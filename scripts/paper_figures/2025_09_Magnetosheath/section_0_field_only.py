# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %%
import os
import numpy as np
import pandas as pd

from src.config import PROC_MSH_FIELD_5MIN, OMNI_DIR
from src.processing.reading import import_processed_data

from src.plotting.comparing.parameter import compare_columns
from src.plotting.space_time import plot_orbit

from src.coordinates.boundaries import calc_msh_r_diff
from src.analysing.comparing import difference_series

# Solar wind data
omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_sw    = import_processed_data(omni_dir, 'omni_5min.cdf')

column_names = {
    'r_x_name'  : 'r_x_GSE',
    'r_y_name'  : 'r_y_GSE',
    'r_z_name'  : 'r_z_GSE',
    'r_name'    : 'r_mag',
    'r_ax_name' : 'r_x_aGSE',
    'r_ay_name' : 'r_y_aGSE',
    'r_az_name' : 'r_z_aGSE',
    'v_x_name'  : 'V_x_GSE',
    'v_y_name'  : 'V_y_GSE',
    'v_z_name'  : 'V_z_GSE',
    'p_name'    : 'P_flow'
}


df_merged = import_processed_data(PROC_MSH_FIELD_5MIN)
df_merged_attrs = df_merged.attrs

df_positions = calc_msh_r_diff(df_merged, 'BOTH', position_key='msh', data_key='sw', column_names=column_names)
df_merged = pd.concat([df_merged,df_positions[['r_MP','r_BS','r_phi','r_F']]],axis=1)
df_merged.attrs = df_merged_attrs

# Erroneous values
df_merged.loc[np.abs(df_merged['B_z_GSM_msh'])>100,'B_z_GSM_msh'] = np.nan

df_merged.loc[df_merged['MA_sw']>60,'MA_sw'] = np.nan

# Add uncertainty
df_merged['Delta B_theta'] = np.abs(difference_series(df_merged['B_clock_sw'],df_merged['B_clock_msh'],unit='rad'))
df_merged.attrs['units']['Delta B_theta'] = 'rad'

df_merged['Delta Bz'] = df_merged['B_z_GSM_msh']/df_merged['B_z_GSM_sw']
df_merged['Delta Bz'] = df_merged['Delta Bz'].replace([np.inf, -np.inf], np.nan)
df_merged.attrs['units']['Delta Bz'] = '1'


# %% Orbit

plot_orbit(df_merged, sc_key='msh', plane='x-rho', bin_width=0.2, centre_Earth='quarter')


# %%
from src.plotting.additions import create_half_circle_marker

def plot_orbit_msh(ax, df, param='count', title='', colourbar=True):

    f_step = 0.1
    f_min = 0
    f_max = 1
    f_num = int((f_max-f_min)/f_step)+1

    theta_step = 3
    theta_min = -90
    theta_max = 90
    theta_num = int((theta_max-theta_min)/theta_step)+1

    f_edges = np.linspace(f_min, f_max, f_num)
    theta_edges = np.linspace(np.radians(theta_min), np.radians(theta_max), theta_num)

    phi   = df['r_phi']*np.sign(df['r_y_GSE_msh'])
    f     = df['r_F']

    counts, f_edges, theta_edges = np.histogram2d(f, phi, bins=[f_edges, theta_edges])

    # meshgrid
    f_centres = 0.5 * (f_edges[:-1] + f_edges[1:])
    theta_centres = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    f_centres = f_edges
    theta_centres = theta_edges

    F, P = np.meshgrid(f_centres, theta_centres, indexing='ij')

    # convert back to radius
    R = mp_shue1998(P) + F * (bs_jelinek2012(P) - mp_shue1998(P))

    X = R * np.cos(P)
    Y = R * np.sin(P)

    #cm = axis.pcolormesh(X, Y, counts, cmap='hot', norm=mpl.colors.LogNorm())
    cm = axis.pcolormesh(X, Y, counts, cmap='hot')


    ax.plot(mp_shue1998(theta_edges)*np.cos(theta_edges),mp_shue1998(theta_edges)*np.sin(theta_edges),c='w',ls='--')
    ax.plot(bs_jelinek2012(theta_edges)*np.cos(theta_edges),bs_jelinek2012(theta_edges)*np.sin(theta_edges),c='w',ls='--')

    create_half_circle_marker(ax, angle_start=270, full=False)

    if colourbar:
        z_label = 'Counts'

        cbar = fig.colorbar(cm, ax=axis)
        cbar.set_label(z_label)

    ax.set_xlabel('X [aGSE]')
    ax.set_ylabel(r'$\sqrt{Y^2+Z^2}$')
    ax.xaxis.set_inverted(True)
    ax.yaxis.set_inverted(True)

    ax.set_facecolor('k')

    ax.set_title(title)

    if colourbar:
        return ax, cm, cbar

    return ax, cm

# %%
from src.coordinates.boundaries import mp_shue1998, bs_jelinek2012
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from src.plotting.utils import calculate_bins
from src.plotting.formatting import data_string

independent = 'E_R'
ind_source  = 'sw'

dependent   = 'AE_17m'
dep_source  = 'pc'

group_param = 'Delta B_theta'
#group_param = 'Delta Bz'
#group_param = 'MA_sw'

### CONSIDER LOOKING AT MSH IN THE DAWN AND IN THE DUSK FLANKS SEPARATELY


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