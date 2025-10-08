# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:55:08 2025

@author: richarj2
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, MultipleLocator

from ...processing.reading import import_processed_data

from ...plotting.space_time import plot_orbit_msh
from ...plotting.utils import save_figure, calculate_bins
from ...plotting.config import colour_dict
from ...plotting.formatting import data_string, create_label
from ...plotting.comparing.parameter import compare_columns



def plot_sc_orbits(sc_dir, sc_keys=None, data_type='mins'):

    if sc_keys is None:
        sc_keys = ('c1','m1','tha','thc','thd','the')

    n_cols = min(3,len(sc_keys))
    n_rows = round(len(sc_keys)/n_cols)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4*(n_cols+1),6*(n_rows+1)), dpi=400)

    for i, sc_key in enumerate(sc_keys):

        df_sc = import_processed_data(sc_dir, f'msh_times_{sc_key}.cdf')

        row = i % n_rows
        col = i // n_rows

        if len(sc_keys)==1:
            ax = axs
        elif n_rows==1:
            ax = axs[col]
        else:
            ax = axs[row,col]

        title = f'{sc_key}: {len(df_sc[f"B_avg_{sc_key}"].dropna()):,} {data_type}'

        _, _, cbar, _ = plot_orbit_msh(df_sc, sc_keys=sc_key, title=title, fig=fig, ax=ax, return_objs=True)

        if col!=0:
            ax.set_ylabel(None)
        if col!=(n_cols-1):
            cbar.set_label(None)
        if row!=(n_rows-1) and n_rows!=1:
            ax.set_xlabel(None)

    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()


def plot_sc_years(msh_dir, sc_keys=None, combined=True, data_type='mins'):

    """
    Combined flag: show all years on one axis, rather than split per spacecraft
    """

    if sc_keys is None:
        sc_keys = ('c1','m1','tha','thb','thc','thd','the')
        if combined:
            sc_keys = ('c1','m1','th')

    n_rows = len(sc_keys)
    n_cols = 1
    width  = 1
    if combined:
        n_rows = 1
        width  = 1/len(sc_keys)

    dims = (4.5*(n_cols+1),2*(n_rows+1))


    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=dims, dpi=400, sharex=True)

    for i, sc_key in enumerate(sc_keys):

        if sc_key=='th':
            years = []
            for sc in [f'th{x}' for x in ('a','b','c','d','e')]:
                try:
                    df_sc = import_processed_data(msh_dir, f'msh_times_{sc}.cdf')
                except:
                    print(f'{sc} data not found in directory')
                    continue
                years.append(df_sc[f'B_avg_{sc}'].dropna().index.year.to_numpy())
            years = np.concatenate(years)
        else:
            try:
                df_sc = import_processed_data(msh_dir, f'msh_times_{sc_key}.cdf')
            except:
                print(f'{sc_key} data not found in directory')
                continue
            years = df_sc[f'B_avg_{sc_key}'].dropna().index.year.to_numpy()

        if len(years)==0:
            continue

        if n_rows==1:
            ax = axs
        else:
            ax = axs[i]

        bins = calculate_bins(years,1)
        counts, _ = np.histogram(years, bins=bins)

        label = f'{sc_key}: {len(years):,} {data_type}'

        offset = 0.5
        if combined:
            offset = i*width

        ax.bar(bins[:-1]+offset, counts, width=width, color=colour_dict.get(sc_key.upper(),'k'), label=label)

        ax.legend(loc='upper right', framealpha=1)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'{val:,.0f}'))
        ax.set_ylabel('Count')

    ax.set_xlabel('Year')

    if not combined:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.tight_layout()

    save_figure(fig)
    plt.show()
    plt.close()


def plot_bias_over_years(df_sw, df_msh, sw_col='AE'):

    # Shows driver for full OMNI and for times when contemp. MSH

    fig, ax = plt.subplots(figsize=(10,6), dpi=400, sharex=True)

    for i, (colour, label) in enumerate(zip(('orange','blue'),('All OMNI','Contemp MSH'))):
        mask = np.ones(len(df_sw),dtype=bool)

        if i==1:
            mask = df_msh.index

        monthly_max = df_sw.loc[mask,sw_col].resample('ME').max()
        rolling_max = monthly_max.rolling(window=6, min_periods=1).mean()

        month_datetimes = monthly_max.index
        fractional_years = month_datetimes.year + (month_datetimes.month - 1)/12

        ax.plot(fractional_years, rolling_max.values, linestyle='-', color=colour, label=label)

    y_label = create_label(sw_col, units=df_sw.attrs['units'])

    ax.set_ylabel(y_label)
    ax.legend(loc='upper left')

    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()


def plot_bias_in_parameters(df_sw, df_msh, params=None):

    if params is None:
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
                mask = df_msh.index
                colour = 'blue'
                axis = ax.twinx()

            series = df_sw.loc[mask,param].dropna()
            if param=='MA':
                series = series.loc[series<200]

            axis.hist(series, bins=calculate_bins(series,param_width.get(param)), histtype='step', edgecolor=colour)
            axis.set_yscale('log')

        x_label = create_label(param, units=df_sw.attrs.get('units',{}))
        ax.set_xlabel(x_label)

    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

# %%


imf_cols = ['B_avg', 'B_x_GSE', 'B_y_GSE', 'B_z_GSE', 'B_y_GSM', 'B_z_GSM', 'B_avg_rms', 'B_vec_rms', 'R_x_BSN', 'R_y_BSN', 'R_z_BSN', 'prop_time_s', 'E_y', 'M_A', 'beta']

plasma_cols = ['P_flow', 'n_p', 'T_p', 'na_np_ratio', 'V_flow', 'V_x_GSE', 'V_y_GSE', 'V_z_GSE', 'R_x_GSE', 'R_y_GSE', 'R_z_GSE', 'E_y', 'M_A', 'M_ms', 'beta', 'E_mag', 'E_x_GSM', 'E_y_GSM', 'E_z_GSM', 'S_mag', 'S_x_GSM', 'S_y_GSM', 'S_z_GSM', 'E_R']

def plot_saturation_overview(df_sw, df_msh, ind_var, dep_var, grp_var, ind_src='sw', dep_src='pc', grp_src='msh', grp_split=None, bounds=None, restrict=True, data_type='counts', plot_test=False):

    ###----------PARAMETER NAMINGS----------###

    ind_param       = '_'.join((ind_var,ind_src))
    if ind_src in ('sw','pc'):
        ind_param_err = None # Need to include
    else:
        ind_param_err   = '_'.join((ind_var,'unc',ind_src))

    if ind_src in ('sw','pc'):
        if ind_var in plasma_cols:
            ind_var_count   = 'plasma_counts'
            ind_param_count = 'plasma_counts_sw'
        elif ind_var in imf_cols:
            ind_var_count   = 'imf_counts'
            ind_param_count = 'imf_counts_sw'
        else:
            ind_var_count = None
            ind_param_count = None
    elif '_GS' in ind_var:
        field, _, coords = ind_var.split('_')
        ind_param_count = '_'.join((field,coords,'count',ind_src))
    else:
        ind_param_count = '_'.join((ind_var,'count',ind_src))

    dep_param       = '_'.join((dep_var,dep_src))
    if dep_src in ('sw','pc'):
        dep_param_err = None # Not present in OMNI (well, needs to be implemented/propagated from B)
    else:
        dep_param_err   = '_'.join((dep_var,'unc',dep_src))

    if dep_src in ('sw','pc'):
        if dep_var in plasma_cols:
            dep_var_count   = 'plasma_counts'
            dep_param_count = 'plasma_counts_sw'
        elif dep_var in imf_cols:
            dep_var_count   = 'imf_counts'
            dep_param_count = 'imf_counts_sw'
        else:
            dep_var_count = None
            dep_param_count = None
    elif '_GS' in dep_var:
        field, _, coords = ind_var.split('_')
        dep_param_count = '_'.join((field,coords,'count',dep_src))
    else:
        dep_param_count = '_'.join((dep_var,'count',dep_src))

    group_param = f'{grp_var}_{grp_src}'

    ###----------MASKS----------###

    mask      = np.ones(len(df_msh),dtype=bool)
    omni_mask = np.ones(len(df_sw),dtype=bool)
    limits    = [None, None]
    invert    = False

    if ind_var=='B_avg':
        bin_step, limits[0] = 2, 0
        if restrict:
            bin_step, limits[1] = 1, 15

    elif 'B_' in ind_var:
        invert = True
        bin_step, limits[1] = 2, 0
        if restrict:
            bin_step, limits[0] = 1, -25

    elif 'V_' in ind_var and ind_var!='V_flow':
        invert = True
        bin_step, limits[1] = 50, 0

    elif 'E_' in ind_var:
        bin_step, limits[0] = 2, 0
        if restrict:
            bin_step, limits[1] = 1, 12

     # Overwrites with those passed in
    if bounds is not None:
        limits = bounds

    if limits[0] is not None:
        mask      &= df_msh[ind_param] >= limits[0]
        omni_mask &= df_sw[ind_var] >= limits[0]

    if limits[-1] is not None:
        mask      &= df_msh[ind_param] <= limits[1]
        omni_mask &= df_sw[ind_var] <= limits[1]

    mask      &= ~df_msh[[ind_param, dep_param, group_param]].isna().any(axis=1)
    omni_mask &= ~df_sw[[ind_var, dep_var]].isna().any(axis=1)

    df_masked      = df_msh.loc[mask]
    df_omni_masked = df_sw.loc[omni_mask]

    ###----------GROUPING PARAMETER LABEL----------###

    median = np.percentile(df_masked[group_param].dropna().to_numpy(),50)
    used_median = True

    z_unit = df_msh.attrs['units'].get(group_param,'')

    if z_unit in ('rad','deg','°'):
        z_unit_str = '°'
    elif z_unit is not None and z_unit not in ('1','NUM',''):
        z_unit_str = f' {z_unit}'
    else:
        z_unit_str = ''

    grp_string = data_string(grp_var)
    grp_label = create_label(grp_var, '°' if z_unit in ('rad','deg','°') else z_unit)


    if grp_var=='Delta B_theta':
        edges = [median]
        bin_width = np.pi/36

    elif grp_var=='theta_Bn':
        used_median = False
        edges = [np.pi/4]
        bin_width = np.pi/36

    elif grp_var=='B_clock':
        used_median = False
        edges = [0]
        bin_width = np.pi/36

    elif grp_var=='Delta B_z':
        used_median = False
        edges = [0]
        bin_width = 10

    elif grp_var=='M_A':
        edges = [median]
        bin_width = 1

    elif 'E_' in grp_var:
        edges = [median]
        bin_width = 0.5

    elif 'V_' in grp_var:
        used_median = False
        edges = [400]
        bin_width = 50
    else:
        raise Exception(f'Grouping parameter "{group_param}" not implemented.')

    if grp_split is not None:
        used_median = False
        edges = [grp_split]

    if z_unit in ('rad','deg','°'):

        z_labels = [f'${grp_string}$<{np.degrees(edges[0]):.1f}{z_unit_str}',
                    f'${grp_string}$$\\geq${np.degrees(edges[0]):.1f}{z_unit_str}',
                    f'${grp_string}$ = {np.degrees(edges[0]):.1f}{z_unit_str}']

    else:

        z_labels = [f'${grp_string}$<{edges[0]:.1f}{z_unit_str}',
                    f'${grp_string}$$\\geq${edges[0]:.1f}{z_unit_str}',
                    f'${grp_string}$ = {edges[0]:.1f}{z_unit_str}']

    if plot_test:

        fig, ax = plt.subplots()

        # Splits contemp MSH OMNI by grouping
        compare_columns(df_masked, ind_param, dep_param, col1_err=ind_param_err, col1_counts=ind_param_count, col2_err=dep_param_err, col2_counts=dep_param_count, col3=group_param, display='rolling_multiple', zs_edges=edges, window_width=bin_step, region='sem', want_legend=True, fig=fig, ax=ax, return_objs=False)
        return

    ###----------PLOT GRIDS----------###

    fig = plt.figure(figsize=(12, 10), dpi=200)
    gs = gridspec.GridSpec(5, 6, figure=fig, width_ratios=[1, 1, 1, 1.15, 1.1, 0.2]) # nrows, ncols

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
    colour_100  = 'w'

    if ind_var == dep_var:
        colour_100 = 'k'
        ax_tl.axline((0,0),slope=1,c='k',ls=':')

    elif not (dep_src=='msh' or ind_src=='msh'):
        # All OMNI Driver vs Response
        _ = compare_columns(df_omni_masked, ind_var, dep_var, col1_counts=ind_var_count, col2_counts=dep_var_count, display='rolling', window_width=bin_step, data_colour=omni_colour, region='sem', fig=fig, ax=ax_tl, return_objs=True)

        # OMNI counts
        ax_bl2.hist(df_omni_masked[ind_var], bins=calculate_bins(df_omni_masked[ind_var],bin_step), color=omni_colour)

    for ax, ls, reg in zip((ax_tl,ax_bl),('-',':'),('sem','none')):
        # OMNI with contemp. MSH times
        _ = compare_columns(df_masked, ind_param, dep_param, col1_err=ind_param_err, col1_counts=ind_param_count, col2_err=dep_param_err, col2_counts=dep_param_count, display='rolling', window_width=bin_step, data_colour=full_colour, error_colour=full_colour, line_style=ls, region=reg, fig=fig, ax=ax, return_objs=True)

    # Splits contemp MSH OMNI by grouping
    _ = compare_columns(df_masked, ind_param, dep_param, col1_err=ind_param_err, col1_counts=ind_param_count, col2_err=dep_param_err, col2_counts=dep_param_count, col3=group_param, display='rolling_multiple', zs_edges=edges, window_width=bin_step, region='sem', want_legend=True, fig=fig, ax=ax_bl, return_objs=True)

    # colour meshses
    cms = []

    for i, (axis, group_region, colour, label) in enumerate(zip((ax_tr, ax_tr2), ('low', 'high'), ('c','m'), z_labels[:2])):

        # MSH Orbit
        if group_region=='low':
            filter_mask = df_masked[group_param]<edges[0]
        elif group_region=='high':
            filter_mask = df_masked[group_param]>=edges[0]

        _, _, cm = plot_orbit_msh(df_masked.loc[filter_mask], title=label, colourbar=False, fig=fig, ax=axis, return_objs=True)
        cms.append(cm)

        # Counts each grouping bin
        df_grouping = df_masked.loc[filter_mask,group_param]
        bins = calculate_bins(df_grouping,bin_width)
        if bins[-2]<edges[0]<bins[-1]:
            bins[-1] = edges[0]
        elif bins[0]<edges[0]<bins[1]:
            bins[0] = edges[0]

        ax_br.hist(df_grouping, bins=bins, color=colour)
        if grp_var=='Delta B_z':
            ax_br.set_xscale('symlog', linthresh=10)
        ax_br.set_xlabel(grp_label)

        # Counts each ind_var bin
        df_msh_param = df_masked.loc[filter_mask,ind_param]
        ax_bl2.hist(df_msh_param, bins=calculate_bins(df_msh_param,bin_step), histtype='step', edgecolor=colour, linewidth=1.2)

        # Counts each year split by grouping
        df_msh_years = df_masked.loc[filter_mask].index.year.to_numpy()
        bins = calculate_bins(df_msh_years,1)
        counts, _ = np.histogram(df_msh_years, bins=bins)

        ax_br2.bar(bins[:-1] + (i-0.5)*0.5, counts, width=0.5, color=colour, label=f'{len(df_msh_years):,}{data_type}' if not used_median else None)

    ax_bl2.axhline(y=100, ls=':', lw=1, c=colour_100)

    if used_median:
        ax_br2.plot([], [], ' ', label=f'$n/2$ = {np.sum(filter_mask):,} {data_type}')

    ###----------FORMATTING----------###

    ax_br.axvline(x=edges[0],c='k',ls='--',label=z_labels[2])
    if z_unit =='rad':
        formatter = FuncFormatter(lambda val, pos: f'{np.degrees(val):.0f}°')
        ax_br.xaxis.set_major_formatter(formatter)
        tick_spacing_deg = 30
        ax_br.xaxis.set_major_locator(MultipleLocator(np.radians(tick_spacing_deg)))

    # Colourbar

    vmin = min(cm.get_array().min() for cm in cms)
    vmax = max(cm.get_array().max() for cm in cms)

    for cm in cms:
        cm.set_clim(vmin, vmax)

    cbar = fig.colorbar(cms[0], cax=cax)
    cbar.set_label(data_type.capitalize())
    cax.yaxis.set_ticks_position('left')

    # Axes

    ax_tl.set_xlabel(None)
    ax_tr.set_xlabel(None)
    ax_tr.set_ylabel(None)
    ax_tr2.set_ylabel('YZ', rotation='horizontal', ha='right')
    ax_tr2.tick_params(axis='y', which='both', left=False, labelleft=False)

    ax_bl2.set_yscale('log')
    ax_bl2.set_ylabel(data_type.capitalize())
    ax_br.set_yscale('log')
    #ax_br2.set_yscale('log')
    ax_br2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'{val:,.0f}'))

    ax_br.legend(loc='upper right', fontsize=10)
    ax_br2.legend(loc='upper right', fontsize=8)

    if invert:
        ax_bl.invert_xaxis()


    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()