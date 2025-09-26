# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %%
import os
import numpy as np
import pandas as pd

from src.config import PROC_MSH_PLASMA_5MIN, OMNI_DIR
from src.processing.reading import import_processed_data

from src.plotting.comparing.parameter import compare_columns, plot_nguyen
from src.plotting.space_time import plot_orbit

from src.coordinates.boundaries import calc_msh_r_diff
from src.analysing.comparing import difference_series

# Solar wind data
omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_sw    = import_processed_data(omni_dir, 'omni_5min')

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

df_merged = import_processed_data(PROC_MSH_PLASMA_5MIN)
df_merged_attrs = df_merged.attrs


df_positions = calc_msh_r_diff(df_merged, 'BOTH', position_key='msh', data_key='sw', column_names=column_names)
df_merged = pd.concat([df_merged,df_positions[['r_MP','r_BS','r_phi','r_F']]],axis=1)
df_merged.attrs = df_merged_attrs

for param, ratio_limit in zip(('E_y_GSM','B_z_GSM'),(5,8)):

    df_merged.loc[np.abs(df_merged[f'{param}_msh'])>ratio_limit*np.abs(df_merged[f'{param}_sw']),f'{param}_msh'] = np.nan

df_merged.loc[df_merged['beta_msh']>50,'beta_msh'] = np.nan
df_merged.loc[df_merged['MA_sw']>60,'MA_sw'] = np.nan

# Add uncertainty
df_merged['Delta B_theta'] = np.abs(difference_series(df_merged['B_clock_sw'],df_merged['B_clock_msh'],unit='rad'))
df_merged.attrs['units']['Delta B_theta'] = 'rad'

df_merged['Delta Bz'] = df_merged['B_z_GSM_msh']/df_merged['B_z_GSM_sw']
df_merged.attrs['units']['Delta Bz'] = '1'

### Look at relative strength in terms of E_Y = (-V x B)_Y
### Do histograms of parameters in raw databases and then in MSH times
### Consider plotting every interval sw and msh parameter

### HERE DO LOOP FOR INDEPENDENT/DEPENDENT RELTIOSN
### SO AE BUT ALSO SATURATION IN MSH BASED ON FACTOR E.G. CLOCK ANGLE
### THEN CREATE FIGURES USING ENTIRE OMNI/CLUSTER DATASETS

# Create similar plots looking at saturation against Electric field and how velocity and magnetic field affect


# %% Nguyen

plot_nguyen(df_merged, 'N_ion', 'B_avg', col_n_sw='n_p')

plot_orbit(df_merged, sc_key='msh', plane='x-rho', bin_width=0.2, centre_Earth='quarter')


# %%
from src.coordinates.boundaries import mp_shue1998, bs_jelinek2012
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

f_step = 0.1
f_num = int(1/f_step)+1

theta_step = 3
theta_num = int(90/theta_step)+1

f_edges = np.linspace(0, 1, f_num)
theta_edges = np.linspace(0, np.pi/2, theta_num)

param_dict = {'beta': 'MA', 'V_x_GSM': 'V_x_GSE'}

for param in ('count','B_avg','V_mag','E_mag','N_tot','P_tot'):

    for filter_zone in ('all','low','high'):

        phi   = df_merged['r_phi']
        f     = df_merged['r_F']

        if filter_zone=='all':
            filter_mask = np.ones(len(phi),dtype=bool)
            title = 'All beta'
        elif filter_zone=='low':
            filter_mask = df_merged['beta_msh']<1
            title = r'$\beta<1$'
        elif filter_zone=='high':
            filter_mask = df_merged['beta_msh']>1
            title = r'$\beta>1$'

        if np.sum(filter_mask)==0:
            print(f'No data for {param} with {filter_zone} filter.')
            continue

        if param == 'count':
            counts, f_edges, theta_edges = np.histogram2d(f[filter_mask], phi[filter_mask], bins=[f_edges, theta_edges])
        else:
            param_sw = param_dict.get(param,param)
            v = df_merged[f'{param}_msh'] / df_merged[f'{param_sw}_sw']

            mask = filter_mask.copy()

            for series in (phi,f,v):
                mask &= ~np.isnan(series)

            mean_v, f_edges, theta_edges, _ = binned_statistic_2d(f[mask], phi[mask], v[mask], statistic='mean', bins=[f_edges, theta_edges])

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

        fig, ax = plt.subplots(figsize=(7,7), dpi=200)

        if param == 'count':
            #cm = ax.pcolormesh(X, Y, counts, cmap='hot', norm=mpl.colors.LogNorm())
            cm = ax.pcolormesh(X, Y, counts, cmap='hot')
            z_label = 'Minutes'
        else:
            z_min, z_max = None, None
            if param == 'beta':
                z_min = 0
                z_max = 2
            elif param == 'B_avg':
                z_min = 0
                z_max = 6
            elif param == 'E_mag':
                z_min = 0
                z_max = 4
            elif param == 'N_tot':
                z_min = 1
                z_max = 8
            elif param == 'V_mag':
                z_min = 0
                z_max = 0.8
            elif param == 'P_tot':
                z_min = 0
                z_max = 0.7

            cm = ax.pcolormesh(X, Y, mean_v, cmap='cool', vmin=z_min, vmax=z_max)
            z_label = f'{param} ratio'

        ax.plot(mp_shue1998(theta_edges)*np.cos(theta_edges),mp_shue1998(theta_edges)*np.sin(theta_edges),c='w',ls='--')
        ax.plot(bs_jelinek2012(theta_edges)*np.cos(theta_edges),bs_jelinek2012(theta_edges)*np.sin(theta_edges),c='w',ls='--')

        ax.set_xlabel('X [aGSE]')
        ax.set_ylabel(r'$\sqrt{Y^2+Z^2}$')
        ax.xaxis.set_inverted(True)

        ax.set_facecolor('k')
        ax.set_aspect('equal')

        cbar = fig.colorbar(cm, ax=ax)
        cbar.set_label(z_label)

        if param == 'count':
            ticks = cbar.get_ticks()
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{int(t*5):,}' for t in ticks])

        ax.set_title(title)

        plt.tight_layout()
        plt.show()


# %% Pulkkinen

rules = {'B_avg':   (2, (1, 5)),
         'B_z_GSM': (2, (1, 5)),
         'B_clock': (np.pi/5, np.pi/36),
         'E_y_GSM': (1, (0.5, 1)),
         'V_mag':   (50, 5),
         'N_tot':   (1, (1, 5)),
         'S_mag':   (10, (4, 100))
         }

# Also include beta MA plot
for param in ('B_avg','B_z_GSM','B_clock','E_y_GSM','V_mag','N_tot','S_mag'):

    if param!='B_z_GSM':
        continue

    sw_param  = f'{param}_sw'

    # Include sw error if B else ignore

    msh_param = f'{param}_msh'
    msh_err   = f'{param}_unc_msh'
    msh_count = f'{param}_count_msh'


    step, bin_width = rules.get(param,(1, (1, 5)))

    compare_columns(df_merged, sw_param, msh_param, display='heat', bin_width=bin_width)

    if '_GS' in param:
        field, _, coords = param.split('_')
        msh_count = f'{field}_{coords}_count_msh'

    compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='MA_sw', display='scatter', scatter_size=20, z_max=12, zero_lines=(param=='B_z_GSM'))

    if param=='B_z_GSM':
        compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='region_duration', display='scatter', scatter_size=20, z_max=60000, data3_name='region duration', zero_lines=(param=='B_z_GSM'))

    compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='beta_msh', display='scatter_binned', col2_counts=msh_count, scatter_size=20, bin_step=step, z_min=0, z_max=2)

# %% beta

# Different format
ma   = 'MA'
beta = 'beta'

sw_param  = f'{ma}_sw'

msh_param = f'{beta}_msh'
msh_err   = f'{beta}_unc_msh'
msh_count = f'{beta}_count_msh'


compare_columns(df_merged[(df_merged['beta_msh']<50)&(df_merged['MA_sw']<60)], sw_param, msh_param, display='heat', bin_width=1)

compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='E_y_GSM_sw', display='scatter', scatter_size=20, z_min=0, z_max=8)

compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='E_y_GSM_sw', display='scatter_binned', col2_counts=msh_count, scatter_size=20, bin_step=1, z_min=0, z_max=8)

# %% Correlation

for param in ('E_y_GSM','E_R'):
    sw_param  = f'{param}_sw'
    msh_param = f'{param}_msh'
    msh_err   = f'{param}_unc_msh'

    if '_GS' in param:
        field, _, coords = param.split('_')
        msh_count = f'{field}_{coords}_count_msh'

    step = 1

    mask = np.ones(len(df_merged),dtype=bool)

    if param=='E_y_GSM':
        mask &= (df_merged.loc[:, sw_param]>0) & (df_merged.loc[:, msh_param]>0) & (np.abs(df_merged.loc[:, msh_param])<1e6)

    compare_columns(df_merged.loc[mask], sw_param, 'AE_17m_pc', col3='MA_sw', display='scatter_binned', scatter_size=20, bin_step=step, z_max=8)

    compare_columns(df_merged.loc[mask], msh_param, 'AE_pc', col1_err=msh_err, col3='MA_sw', display='scatter_binned', scatter_size=20, bin_step=step)


# %% Saturation

for couple in ('E_R_sw','E_y_GSM_sw','B_z_GSM_sw'):

    bin_step = 1

    compare_columns(df_sw[df_sw[couple.replace('_sw','')]>0], couple.replace('_sw',''), 'AE_17m', display='scatter_binned', scatter_size=20, bin_step=bin_step)

    mask = np.ones(len(df_merged),dtype=bool)

    if 'E_' in couple:
        mask &= df_merged[couple]>0
    elif 'B_' in couple:
        mask &= df_merged[couple]<0
    elif 'V_' in couple and 'V_mag' not in couple:
        mask &= df_merged[couple]<0
        bin_step = 50

    for z_param in ('MA_sw','beta_msh','Delta B_theta'):

        if z_param=='MA_sw':
            edges = [5]
        elif z_param=='MA_med':
            edges = [9.2]
            z_param = 'MA_sw'
        elif z_param=='beta_msh':
            edges = [1]
        elif z_param=='Delta B_theta':
            edges = [np.pi/6]
        #if z_param=='V_mag':
        #    edges = [400,500,600,700,800]

        compare_columns(df_merged[mask], couple, 'AE_17m_pc', col3=z_param, display='scatter_binned_multiple', scatter_size=20, zs_edges=edges, bin_step=bin_step)

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
        z_label = 'Minutes'

        cbar = fig.colorbar(cm, ax=axis)
        cbar.set_label(z_label)

        if param == 'count':
            ticks = cbar.get_ticks()
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{int(t*5):,}' for t in ticks])

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
from scipy.stats import binned_statistic_2d
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter

from src.plotting.utils import calculate_bins
from src.plotting.formatting import data_string

independent = 'E_y_GSM'
ind_source  = 'sw'

dependent   = 'AE_17m'
dep_source  = 'pc'

reference = False

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
    dep_param_err = None
else:
    dep_param_err   = '_'.join((dependent,'unc',dep_source))

if dep_source in ('sw','pc'):
    dep_param_count = None
elif '_GS' in dependent:
    field, _, coords = independent.split('_')
    dep_param_count = '_'.join((field,coords,'count',dep_source))
else:
    dep_param_count = '_'.join((dependent,'count',dep_source))

group_param  = 'Delta B_theta'
#group_param = 'MA_sw'
#group_param = 'beta_msh'
#group_param = 'Delta Bz'

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
    bin_step = 3
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

df_masked      = df_merged.loc[mask]
df_omni_masked = df_sw.loc[omni_mask]


median = np.percentile(df_masked[group_param].dropna(),50)

if group_param=='MA_sw':
    edges = [median]
    bin_width = 1
    z_labels = [f'${data_string(group_param)}$<{edges[0]:.1f}{z_unit_str}',
                f'${data_string(group_param)}$$\\geq${edges[0]:.1f}{z_unit_str}']
elif group_param=='beta_msh':
    edges = [1]
    bin_width = 1
    z_labels = [f'${data_string(group_param)}$<{edges[0]:.1f}{z_unit_str}',
                f'${data_string(group_param)}$$\\geq${edges[0]:.1f}{z_unit_str}']
elif group_param=='Delta B_theta':
    edges = [median]
    bin_width = np.pi/18
    z_labels = [f'${data_string(group_param)}$<{np.degrees(edges[0]):.1f}{z_unit_str}',
                f'${data_string(group_param)}$$\\geq${np.degrees(edges[0]):.1f}{z_unit_str}']
elif group_param=='Delta Bz':
    edges = [0]
    bin_width = 1
    z_labels = [f'$sgn({data_string(group_param)})$<0{z_unit_str}',
                f'$sgn({data_string(group_param)})$$\\geq$0{z_unit_str}']

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
scale_factor = 5

if independent == dependent:
    ax_tl.axline((0,0),slope=1,c='k',ls=':')
elif dep_source!='msh' and ind_source!='msh':

    _ = compare_columns(df_omni_masked, independent, dependent, display='scatter_binned', scatter_size=20, bin_step=bin_step, brief_title='Full OMNI', data_colour=omni_colour, scat_marker='x', fig=fig, ax=ax_tl, return_objs=True)

    df_omni_param = df_omni_masked.loc[:,independent]
    ax_bl2.hist(df_omni_param, bins=calculate_bins(df_omni_param,bin_step), color=omni_colour)

_ = compare_columns(df_masked, ind_param, dep_param, col1_err=ind_param_err, col1_counts=ind_param_count, col2_err=dep_param_err, col2_counts=dep_param_count, display='scatter_binned', scatter_size=20, bin_step=bin_step, data_colour=full_colour, error_colour=full_colour, scat_marker='+', fig=fig, ax=ax_tl, return_objs=True)

_ = compare_columns(df_masked, ind_param, dep_param, col1_err=ind_param_err, col1_counts=ind_param_count, col2_err=dep_param_err, col2_counts=dep_param_count, col3=group_param, display='scatter_binned_multiple', scatter_size=20, zs_edges=edges, bin_step=bin_step, want_legend=False, reference=reference, fig=fig, ax=ax_bl, return_objs=True)


# colour meshses
cms = []

for i, (axis, group_region, colour, label) in enumerate(zip((ax_tr, ax_tr2), ('low', 'high'), ('c', 'm'), z_labels)):

    if group_region=='low':
        filter_mask = df_masked[group_param]<edges[0]
    elif group_region=='high':
        filter_mask = df_masked[group_param]>=edges[0]

    _, cm = plot_orbit_msh(axis, df_masked.loc[filter_mask], title=label, colourbar=False)
    cms.append(cm)

    df_msh_param = df_masked.loc[filter_mask,ind_param]
    df_msh_years = df_masked.loc[(~df_masked[ind_param].isna())&(filter_mask)].index.year.to_numpy()

    # Counts each grouping bin
    weights=np.full(len(df_msh_param), scale_factor)

    bins = calculate_bins(df_masked.loc[filter_mask,group_param],bin_width)
    if bins[-2]<edges[0]<bins[-1]:
        bins[-1] = edges[0]
    elif bins[1]>edges[0]>bins[0]:
        bins[0] = edges[0]

    ax_br.hist(df_masked.loc[filter_mask,group_param], bins=bins, color=colour, weights=np.full(np.sum(filter_mask), scale_factor), label=label)

    # Counts each independent bin
    bins = calculate_bins(df_msh_param,bin_step)
    counts, _ = np.histogram(df_msh_param, bins=bins)

    ax_bl2.bar(bins[:-1] + (i+0.5)*bin_step/2, scale_factor*counts, width=bin_step/2, color=colour, alpha=0.9)

    # Counts each year
    bins = calculate_bins(df_msh_years,1)
    counts, _ = np.histogram(df_msh_years, bins=bins)

    ax_br2.bar(bins[:-1] + (i-0.5)*0.5, scale_factor*counts, width=0.5, color=colour, label=f'{5*np.sum(filter_mask):,} mins')



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
cbar.set_label(z_label)

ticks = cbar.get_ticks()
cbar.set_ticks(ticks)
cbar.set_ticklabels([f'{int(t*5):,}' for t in ticks])
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
ax_br.set_yscale('log')
ax_br2.set_yscale('log')
ax_bl2.set_ylabel('Mins')

ax_br.legend(loc='upper right', fontsize=8)
ax_br2.legend(loc='upper right', fontsize=8)

# if invert:
#     ax_bl.invert_xaxis()


plt.tight_layout()
plt.show()