# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:55:08 2025

@author: richarj2
"""

import numpy as np
import itertools as it

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, MultipleLocator

from .plotting_utils import minimum_counts, def_param_names, get_var_bin_width, get_variable_range, grp_param_splitting, get_lagged_columns

from ...plotting.space_time import plot_orbit_msh
from ...plotting.utils import save_figure, calculate_bins
from ...plotting.formatting import create_label, add_legend, shifted_angle_ticks, format_string
from ...plotting.comparing.parameter import compare_dataframes
from ...plotting.relationships import plot_fit_params_against_z
from ...plotting.config import black, blue, grey, pink, green


def shift_angular_data(df, *cols):
    # Shift angular data to centre lies at +-180 rather than 0

    df = df.copy()

    for to_shift in cols:
        shift_unit = df.attrs.get('units',{}).get(to_shift,'')
        if shift_unit == 'rad':
            df[to_shift] = (df[to_shift] + 2*np.pi) % (2*np.pi)
        elif shift_unit in ('deg','°'):
            df[to_shift] = (df[to_shift] + 360) % 360

    return df

def mask_df(df, col, limits=None):

        mask = ~df[col].isna()

        if limits:
            if limits[0] is not None:
                mask &= df[col] >= limits[0]
            if limits[-1] is not None:
                mask &= df[col] <= limits[1]

        return df.loc[mask]

# %% Overview


# ADD EARTH-SUN LINE CALCULATION AND GROUPING INTO OVERVIEW
# df_omni['L1_rho'] = np.sqrt(df_omni['R_y_GSE']**2+df_omni['R_z_GSE']**2)
# df_msh['L1_rho_sw'] = np.sqrt(df_msh['R_y_GSE_sw']**2+df_msh['R_z_GSE_sw']**2)


def plot_saturation_overview(df_sw, df_msh, df_pc, ind_var, dep_var, grp_var, ind_src='sw', dep_src='pc', grp_src='msh', min_count=None, plot_test=False, same_var=False, invert_x=False, invert_y=False, restrict=True, bounds=None, **kwargs):

    kwargs['data1_name'] = create_label(kwargs['data1_name'])
    kwargs['data2_name'] = create_label(kwargs['data2_name'])

    file_name = f'Saturation_{dep_var}_with_{ind_var}_splitby_{grp_var}'

    sample_interval = df_msh.attrs.get('sample_interval','5min')
    data_type = 'mins' if sample_interval == '1min' else 'counts'

    if min_count is None:
        min_count = minimum_counts[data_type]

    ind_err, ind_count = def_param_names(df_sw, ind_var)
    dep_err, dep_count = def_param_names(df_pc, dep_var)

    ###----------MASKS----------###

    bin_width, limits, invert = get_variable_range(ind_var, ind_src, dep_var=dep_var, restrict=restrict, bounds=bounds)

    df_ind = mask_df(df_sw, ind_var, limits)
    df_dep = mask_df(df_pc, dep_var)

    intersect = df_ind.index.intersection(df_dep.index)
    df_ind = df_ind.loc[intersect]
    df_dep = df_dep.loc[intersect]


    ###----------GROUPING PARAMETER LABEL----------###

    z_unit = df_msh.attrs['units'].get(grp_var,'')
    edges, grp_bin_width, grp_label, z_labels, used_median = grp_param_splitting(df_ind, grp_var, grp_var, z_unit, **kwargs)
    kwargs['zs_edges'] = edges

    if plot_test:

        fig, ax = plt.subplots(figsize=(8,5), dpi=200)

        # Splits contemp MSH OMNI by grouping
        kwargs.update({'display': 'rolling_multiple', 'region': 'sem', 'cmap_name': 'autumn_r'})
        _ = compare_dataframes(df_ind, df_dep, ind_var, dep_var, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_err, col2_counts=dep_count, df3=df_ind, col3=grp_var, want_legend=True, fig=fig, ax=ax, return_objs=True, **kwargs)

        plt.tight_layout()
        save_figure(fig, sub_directory='Overview', file_name=file_name)
        plt.show()
        plt.close()

        return

    ###----------PLOT GRIDS----------###

    fig = plt.figure(figsize=(12, 10), dpi=200)
    gs = gridspec.GridSpec(5, 6, figure=fig, width_ratios=[1, 1, 1, 1.15, 1.1, 0.2]) # nrows, ncols
    kwargs.update({'fig': fig, 'return_objs': True, 'display': 'rolling', 'region': 'sem'})

    # gs[rows occupied, columns occupied]

    ax_tl  = fig.add_subplot(gs[0:2, 0:3])
    ax_tr  = fig.add_subplot(gs[0:3, 3])
    ax_tr2 = fig.add_subplot(gs[0:3, 4], sharex=ax_tr, sharey=ax_tr)
    cax    = fig.add_subplot(gs[0:3, -1]) # for colourbar

    ax_bl  = fig.add_subplot(gs[2:4, 0:3], sharex=ax_tl)
    ax_bl2 = fig.add_subplot(gs[4, 0:3], sharex=ax_tl)
    ax_br  = fig.add_subplot(gs[3, 3:6])
    ax_br2 = fig.add_subplot(gs[4, 3:6])

    omni_colour = black
    full_colour = 'b'

    if ind_var == dep_var or same_var:
        slope = 1
        if invert_y and not invert_x:
            slope = -1
        ax_tl.axline((limits[0],limits[0]), slope=slope, c='k', ls=':')
        ax_tl.grid(ls=':', c='grey', lw=0.5)

    if ind_src=='sw':
        # OMNI counts
        ax_bl2.hist(df_ind[ind_var], bins=calculate_bins(df_ind[ind_var],bin_width), color=omni_colour)

    if not (ind_src=='msh' or dep_src=='msh'):

        # All OMNI Driver vs Response
        _ = compare_dataframes(df_ind, ind_var, dep_var, col1_counts=ind_count, col2_counts=dep_count, data_colour=omni_colour,  ax=ax_tl, **kwargs)

    kwargs.update({'col1_counts': ind_count, 'col1_err': ind_err, 'col2_counts': dep_count, 'col2_err': dep_err, 'col3': grp_var})
    for i, (ax, ls, reg) in enumerate(zip((ax_tl,ax_bl),('-',':'),('sem','none'))):

        # OMNI with contemp. MSH times
        kwargs.update({'region': reg, 'ax': ax, 'line_style': ls})
        _ = compare_dataframes(df_ind, df_dep, ind_var, dep_var, data_colour=full_colour, error_colour=full_colour,  **kwargs)

    # Splits contemp MSH OMNI by grouping
    kwargs.update({'display': 'rolling_multiple', 'region': 'sem', 'ax': ax_bl})
    _ = compare_dataframes(df_ind, df_dep, ind_var, dep_var, want_legend=True, **kwargs)

    # colour meshses
    cms = []

    for i, (axis, group_region, colour, label) in enumerate(zip((ax_tr, ax_tr2), ('low', 'high'), ('c','m'), z_labels[:2])):

        # MSH Orbit
        if group_region=='low':
            filter_mask = df_ind[grp_var]<edges[0]
        elif group_region=='high':
            filter_mask = df_ind[grp_var]>=edges[0]

        _, _, cm = plot_orbit_msh(df_ind.loc[filter_mask], title=label, colourbar=False, fig=fig, ax=axis, return_objs=True)
        cms.append(cm)

        # Counts each grouping bin
        df_grouping = df_ind.loc[filter_mask,grp_var]
        bins = calculate_bins(df_grouping,grp_bin_width)
        if bins[-2]<edges[0]<bins[-1]:
            bins[-1] = edges[0]
        elif bins[0]<edges[0]<bins[1]:
            bins[0] = edges[0]

        ax_br.hist(df_grouping, bins=bins, color=colour)
        if grp_var=='Delta B_z':
            ax_br.set_xscale('symlog', linthresh=10)
        elif grp_var=='beta':
            ax_br.set_xscale('symlog', linthresh=1, linscale=1)
        ax_br.set_xlabel(grp_label)

        # Counts each ind_var bin
        df_msh_param = df_ind.loc[filter_mask,ind_var]
        ax_bl2.hist(df_msh_param, bins=calculate_bins(df_msh_param,bin_width), histtype='step', edgecolor=colour, linewidth=1.2)

        # Counts each year split by grouping
        df_msh_years = df_ind.loc[filter_mask].index.year.to_numpy()
        bins = calculate_bins(df_msh_years,1)
        counts, _ = np.histogram(df_msh_years, bins=bins)

        ax_br2.bar(bins[:-1] + (i-0.5)*0.5, counts, width=0.5, color=colour, label=f'{len(df_msh_years):,} {data_type}' if not used_median else None)

    ax_bl2.axhline(min_count, c='k', ls='-')
    ax_bl2.axhline(min_count, c='w', ls=':')

    if used_median:
        ax_br2.plot([], [], ' ', label=f'$n/2$ = {np.sum(filter_mask):,} {data_type}')

    ###----------FORMATTING----------###

    ax_br.axvline(x=edges[0], c=black, ls='--', label=z_labels[2])
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

    ###----------AXES----------###

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

    if invert or invert_x:
        ax_bl.invert_xaxis()
    if (invert and ind_var == dep_var) or invert_y:
        ax_tl.invert_yaxis()
        ax_bl.invert_yaxis()


    plt.tight_layout()
    save_figure(fig, sub_directory='Overview', file_name=file_name)
    plt.show()
    plt.close()


# %% Compare_OMNI

def plot_driver_response(df_sw, df_msh, df_pc, ind_var, dep_var, dep_src='pc', sw_colour=black, msh_colour=pink, bounds=None, restrict=True, shift_centre=True, min_count=None, compare_colour=green, **kwargs):
    """
    col1: PC vs SW
    col2: PC vs MSH
    col3: MSH vs SW
    """
    sample_interval = df_msh.attrs.get('sample_interval','5min')
    data_type = 'mins' if sample_interval == '1min' else 'counts'

    msh_map = kwargs.get('msh_map',{})

    if min_count is None:
        min_count = minimum_counts[data_type]

    kwargs['min_count'] = min_count
    kwargs['display']   = kwargs.get('display','rolling')
    if kwargs['display']=='rolling':
        kwargs['region'] = kwargs.get('region','sem')

    if ind_var=='B_clock':
        shift_centre = True and shift_centre
    else:
        shift_centre = False

    enumerator = zip((df_sw, df_msh, df_sw),
                     (df_pc, df_pc, df_msh),
                     (ind_var, msh_map.get(ind_var,ind_var), ind_var),
                     (dep_var, dep_var, msh_map.get(ind_var,ind_var)),
                     ('sw','msh','sw'),
                     (sw_colour, msh_colour, compare_colour))

    ###----------PLOT GRIDS----------###
    n_rows, n_cols = 2, 3

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows), dpi=200, height_ratios=[3,2], sharex='col')

    # [row] [col]
    axs[0][0].sharey(axs[0][1])


    for i, (df_ind, df_dep, ind, dep, ind_src, colour), in enumerate(enumerator):

        ax0 = axs[0][i]
        ax1 = axs[1][i]

        ind_err, ind_count = def_param_names(df_ind, ind)
        dep_err, dep_count = def_param_names(df_dep, dep)

        bin_width, limits, invert = get_variable_range(ind, ind_src, dep_var=dep, restrict=restrict, bounds=bounds, shift_centre=shift_centre)

        if shift_centre:
            df_ind = shift_angular_data(df_ind, ind)

        df_ind = mask_df(df_ind, ind, limits)
        df_dep = mask_df(df_dep, dep)

        intersect = df_ind.index.intersection(df_dep.index)
        df_ind = df_ind.loc[intersect]
        df_dep = df_dep.loc[intersect]

        kwargs['window_width'] = bin_width
        kwargs['data_colour']  = colour
        kwargs['error_colour'] = colour

        kwargs_source = kwargs.copy()

        if i==2:
            if ind_var in msh_map:
                independent = msh_map.get(ind_var,ind_var)
                kwargs_source['data1_name'] = independent

            kwargs_source['data2_name'] = create_label(f'{kwargs_source["data1_name"]}_msh')
            kwargs_source['data1_name'] = kwargs['data1_name']

        else:
            if i==1 and ind_var in msh_map:
                kwargs_source['data1_name'] = independent

            if shift_centre:
                df_dep = shift_angular_data(df_dep, dep)

        kwargs_source['data1_name'] = create_label(f'{kwargs_source["data1_name"]}_{ind_src}')

        if kwargs_source['display']=='scatter':
            ind_err       = None
            dep_err = None

        ###----------PLOTS----------###
        _ = compare_dataframes(df_ind, df_dep, ind, dep, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_err, col2_counts=dep_count, fig=fig, ax=ax0, return_objs=True, **kwargs_source)

        if i==2:
            bins_x = calculate_bins(df_ind[ind], bin_width)
            bins_y = calculate_bins(df_dep[dep], bin_width)

            ax1.hist2d(df_ind[ind], df_dep[dep], bins=[bins_x, bins_y], cmap='hot', norm=mpl.colors.LogNorm())
            ax1.set_facecolor('k')

            ax1.axline((limits[0],limits[0]), slope=1, c='w', ls=':')

        else:
            ax1.hist(df_ind[ind], bins=calculate_bins(df_ind[ind],bin_width), color=colour)

            ax1.axhline(min_count, c='k', ls='-')
            ax1.axhline(min_count, c='w', ls=':')
            ax1.set_yscale('log')

        if i==0:
            kwargs_source['data_colour']  = blue
            kwargs_source['error_colour'] = blue

            _ = compare_dataframes(df_ind, df_dep, ind, dep, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_err, col2_counts=dep_count, fig=fig, ax=ax0, return_objs=True, **kwargs_source)

            ax1.hist(df_ind[ind], bins=calculate_bins(df_ind[ind],bin_width), color=blue)

        # Formatting
        ax0.grid(ls=':', c=grey, lw=0.5)
        if df_ind.attrs.get('units',{}).get(ind,'')==df_dep.attrs.get('units',{}).get(dep,''):
            ax0.axline((limits[0],limits[0]), slope=1, c=black, ls=':')

        if shift_centre:
            ax0.axvline(x=np.pi, c=grey, ls=':')
            shifted_angle_ticks(ax1, 'x')

            if i==2:
                shifted_angle_ticks(ax0, 'y')

        if invert:
            ax0.invert_xaxis()
            if i==2:
                ax0.invert_yaxis()

        if i==1:
            axs[0][i].set_ylabel(None)

        ax1.tick_params(labelbottom=False)
        ax0.tick_params(labelbottom=True)

    axs[0][0].text(0.02, 0.95, kwargs.get('region',''), transform=axs[0][0].transAxes, va='top', ha='left')
    axs[1][0].set_ylabel(data_type.capitalize())

    plt.tight_layout()
    save_figure(fig, file_name=f'{dep_var}_vs_{ind_var}_sw_msh_{kwargs.get("region","scatter")}', sub_directory='Pulkkinen')
    plt.show()
    plt.close()

def plot_driver_multi_responses(df_omni, df_sc, df_pc, ind_var, *dep_vars, ind_src='sw', dep_src='pc', omni_colour=black, contemp_colour=blue, sc_colour=pink, bounds=None, restrict=True, shift_centre=True, bottom_axis='scatter', **kwargs):
    """
    Look at OMNI and in-situ data in driver-response
    So dependent variable is a PC index
    """

    if df_pc is None:
        raise ValueError('Polar cap dataframe is none.')

    if df_sc is not None:
        sample_interval = df_sc.attrs.get('sample_interval','5min')
    elif df_omni is not None:
        sample_interval = df_omni.attrs.get('sample_interval','5min')
    data_type = 'mins' if sample_interval == '1min' else 'counts'

    kwargs['min_count'] = kwargs.get('min_count',minimum_counts[data_type])
    kwargs['display']   = kwargs.get('display','rolling')
    if kwargs['display']=='rolling':
        kwargs['region'] = kwargs.get('region','sem')

    if 'data1_name' in kwargs:
        kwargs['data1_name'] = create_label(kwargs['data1_name'])

    if ind_var=='B_clock':
        shift_centre = True and shift_centre
    else:
        shift_centre = False

    ###----------PLOT GRIDS----------###
    ind = ind_var

    n_rows, n_cols = 2, len(dep_vars)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows), dpi=200, height_ratios=[3,2], sharex='col')

    for i, dep_var in enumerate(dep_vars):

        ax0 = axs[0][i]
        ax1 = axs[1][i]

        if ind_src=='msh' or df_omni is None:
            omni_j = -1
            enumerator = ((df_sc,dep_var,sc_colour),)
        elif df_sc is None:
            omni_j = 0
            enumerator = ((df_omni,dep_var,omni_colour),)
        else:
            omni_j = 0
            overlap = df_omni.index.intersection(df_sc.index)
            enumerator = zip((df_omni,df_omni.loc[overlap],df_sc),(dep_var,dep_var,dep_var),(omni_colour,contemp_colour,sc_colour))

        for j, (df, dep, colour) in enumerate(enumerator):
            print(dep)
            if len(df)==0:
                print('df is empty')

            ind_err, ind_count = def_param_names(df, ind_var)
            dep_err, dep_count = def_param_names(df, dep_var)

            bin_width, limits, invert = get_variable_range(ind_var, ind_src, dep_var=dep_var, restrict=restrict, bounds=bounds, shift_centre=shift_centre)

            if shift_centre:
                df = shift_angular_data(df, ind_var)

            df_ind = mask_df(df, ind, limits)
            df_dep = mask_df(df_pc, dep)

            intersect = df_ind.index.intersection(df_dep.index)
            df_ind = df_ind.loc[intersect]
            df_dep = df_dep.loc[intersect]

            kwargs['window_width'] = bin_width
            kwargs['window_step']  = bin_width/10
            kwargs['data_colour']  = colour
            kwargs['error_colour'] = colour

            if 'data_name_map' in kwargs:
                kwargs['data2_name'] = create_label(kwargs['data_name_map'].get(dep_var,dep_var))

            # Rolling window
            _ = compare_dataframes(df_ind, df_dep, ind, dep, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_err, col2_counts=dep_count, fig=fig, ax=ax0, return_objs=True, **kwargs)

            if bottom_axis=='heat' and j==0:
                bins_x = calculate_bins(df_ind[ind], bin_width)
                bins_y = calculate_bins(df_dep[dep], bin_width)

                ax1.hist2d(df_ind[ind], df_dep[dep], bins=[bins_x, bins_y], cmap='hot', norm=mpl.colors.LogNorm())
                ax1.set_facecolor('k')

                ax1.axline((limits[0],limits[0]), slope=1, c='w', ls=':')

            elif bottom_axis=='hist':
                hist_type = 'bar' if j==omni_j else 'step'
                ax1.hist(df_ind[ind], bins=calculate_bins(df_ind[ind],bin_width), color=colour, histtype=hist_type)

                ax1.axhline(kwargs['min_count'], c='k', ls='-')
                ax1.axhline(kwargs['min_count'], c='w', ls=':')
                ax1.set_yscale('log')

            elif bottom_axis=='scatter':
                # Scatter
                kwargs_copy = kwargs.copy()
                kwargs_copy['display'] = 'scatter'
                ind_err = None
                dep_err = None

                _ = compare_dataframes(df_ind, df_dep, ind, dep, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_err, col2_counts=dep_count, fig=fig, ax=ax1, return_objs=True, **kwargs_copy)

        # Formatting
        ax0.grid(ls=':', c=grey, lw=0.5)
        ax1.grid(ls=':', c=grey, lw=0.5)
        if df_omni.attrs.get('units',{}).get(ind_var,'')==df_pc.attrs.get('units',{}).get(dep_var,'')=='mV/m':
            ax0.axline((limits[0],limits[0]), slope=1, c=black, ls=':')

        if shift_centre:
            ax0.axvline(x=np.pi, c=grey, ls=':')
            shifted_angle_ticks(ax1, 'x')

        if invert:
            ax0.invert_xaxis()

        ax1.set_ylabel(None)
        ax1.set_xlabel(None)

        ax1.tick_params(labelbottom=False)
        ax0.tick_params(labelbottom=True)

    axs[0][0].text(0.02, 0.95, kwargs.get('region',''), transform=axs[0][0].transAxes, va='top', ha='left')

    file_name = f'Responses_to_{ind_var}_driver'
    if df_sc is None:
        file_name += '_OMNI'

    plt.tight_layout()
    save_figure(fig, file_name=file_name, sub_directory='Pulkkinen')
    plt.show()
    plt.close()

def plot_different_lags_saturation(df, df_pc, ind_var, dep_var, ind_src='sw', bounds=None, restrict=True, skip_zero=False, **kwargs):

    kwargs['min_count'] = kwargs.get('min_count',minimum_counts['counts'])
    kwargs['display']   = kwargs.get('display','rolling')
    if kwargs['display']=='rolling':
        kwargs['region'] = kwargs.get('region','sem')

    ind_err, ind_count = def_param_names(df, ind_var)

    bin_width, limits, invert = get_variable_range(ind_var, ind_src, restrict=restrict, bounds=bounds)
    kwargs['window_width'] = bin_width

    df_ind = mask_df(df, ind_var, limits)

    dep_cols = [col for col in df_pc.columns if col.startswith(dep_var)]

    if skip_zero and dep_var in dep_cols:
        dep_cols.remove(dep_var)

    cmap = plt.get_cmap('autumn_r')
    norm = plt.Normalize(vmin=0, vmax=len(dep_cols)-1)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

    for i, col in enumerate(dep_cols):

        dep = col
        num_parts = len(col.split('_'))
        if num_parts==3:
            lag = col.split('_')[1] + r' + $\delta t$'
        elif num_parts==2:
            lag = col.split('_')[1]
        else:
            lag = '0m'

        df_dep = mask_df(df_pc, dep_var)
        intersect = df_ind.index.intersection(df_dep.index)
        df_ind_masked = df_ind.loc[intersect]
        df_dep_masked = df_dep.loc[intersect]

        colour = cmap(norm(i))
        kwargs['data_colour'] = colour
        kwargs['error_colour'] = colour

        _ = compare_dataframes(df_ind_masked, df_dep_masked, ind_var, dep, col1_err=ind_err, col1_counts=ind_count, fig=fig, ax=ax, return_objs=True, **kwargs)

        ax.plot([], [], ls='-', color=colour, label=lag)

    ax.set_ylabel(create_label(dep_var,units=df_pc.attrs['units']))

    add_legend(fig, ax)
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

def plot_different_lags_fits(df, df_pc, ind_var, dep_var, ind_src='sw', bounds=None, restrict=True, skip_zero=True, **kwargs):

    kwargs['fit_type'] = kwargs.get('fit_type','saturation')

    ind_err, ind_count = def_param_names(df, ind_var)
    dep_err, dep_count = def_param_names(df_pc, dep_var)

    _, limits, _ = get_variable_range(ind_var, ind_src, restrict=restrict, bounds=bounds)

    df_ind = mask_df(df, ind_var, limits)

    dep_vars = get_lagged_columns(df_pc, dep_var, skip_zero)

    fig, (ax, ax2) = plot_fit_params_against_z(df_ind, ind_var, dep_vars, df_dep=df_pc, col1_err=ind_err, col2_err=dep_err, col1_counts=ind_count, col2_counts=dep_count, **kwargs)
    ax.set_xlabel('Lag [mins]')

    ind_str = format_string(ind_var)
    dep_str = format_string(dep_var)
    ax.set_title(f'Fitting ${dep_str}$ against {ind_src.upper()} ${ind_str}$')

    file_name = f'Fitting_{kwargs["fit_type"]}_of_{dep_var}_to_{ind_src}_{ind_var}_lags'
    save_figure(fig, file_name=file_name, sub_directory='Comparing_Lags')
    plt.show()
    plt.close()

def plot_pulkkinen_grid(df_omni, df_sc, params, source='msh', bounds=None, restrict=True, shift_centre=True, compare_colour=green, **kwargs):
    """
    Pulkkinen comparisons

    So plots OMNI vs sc (sw or msh) for range of parameters
    """

    msh_map  = kwargs.get('msh_map',{})
    name_map = kwargs.get('data_name_map',{})

    num_columns = kwargs.get('num_columns',2)

    kwargs['min_count'] = kwargs.get('min_count',minimum_counts['counts'])
    kwargs['display']   = kwargs.get('display','rolling')
    if kwargs['display']=='rolling':
        kwargs['region'] = kwargs.get('region','sem')

    kwargs['data_colour']  = compare_colour
    kwargs['error_colour'] = compare_colour

    if source=='msh':
        enumerator = [(p, p) if i == 0 else (p, msh_map[p]) for p in params for i in (0, 1) if i == 0 or p in msh_map]
        width, height = 8, 5
    else:
        enumerator = [(p,p) for p in params]
        width, height = 6, 5

    for n in range(num_columns,0,-1):
        if len(enumerator) % n == 0:
            n_cols, n_rows = n, len(enumerator) // n
            break

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width*n_cols, height*n_rows), dpi=400)

    ###----------PLOT GRIDS----------###
    for i, (independent, dependent), in enumerate(enumerator):

        col, row = i % n_cols, i // n_cols

        ax = axs[row][col]

        data_name = name_map.get(independent,independent)

        ind_err, ind_count = def_param_names(df_omni, independent)
        dep_err, dep_count = def_param_names(df_sc, dependent)

        bin_width, limits, invert = get_variable_range(independent, source, dep_var=dependent, restrict=restrict, bounds=bounds, shift_centre=shift_centre)

        # Shift angular data to centre lies at +-180 rather than 0
        if independent=='B_clock' and shift_centre:
            df_omni = shift_angular_data(df_omni, independent)
            df_sc   = shift_angular_data(df_sc, dependent)

        df_ind = mask_df(df_omni, independent, limits)
        df_dep = mask_df(df_sc, dependent, limits if source=='sw' and independent!='B_clock' else None)
        df_dep = df_dep.sort_index()

        df_dep = df_dep.reindex(df_ind.index, method='nearest', tolerance='30s').dropna(how='all')
        df_ind = df_ind.loc[df_dep.index]

        kwargs['window_width'] = bin_width

        kwargs_source = kwargs.copy()
        kwargs_source['data1_name'] = create_label(f'{data_name}_omni')

        if independent!=dependent:
            data_name = dependent
        kwargs_source['data2_name'] = create_label(f'{data_name}_{source}')

        if kwargs['display']=='scatter':
            ind_err = None
            dep_err = None
            if f'sc_{source}' in df_dep:
                kwargs_source['df3'] = df_dep
                kwargs_source['col3'] = f'sc_{source}'
                kwargs_source['data_colour'] = 'spacecraft'

        _ = compare_dataframes(df_ind, df_dep, independent, dependent, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_err, col2_counts=dep_count, fig=fig, ax=ax, return_objs=True, **kwargs_source)

        ax.axline((limits[0],limits[0]), slope=1, c=black, ls=':')
        ax.grid(ls=':', c=grey, lw=0.5)

        if independent=='B_clock' and shift_centre:
            ax.axvline(x=np.pi, c=grey, ls=':')
            shifted_angle_ticks(ax, 'x')
            shifted_angle_ticks(ax, 'y')

        if independent.startswith('T_') and kwargs['display']!='heat':
            ax.set_xscale('log')
            ax.set_yscale('log')

        if invert:
            ax.invert_xaxis()
            ax.invert_yaxis()

    axs[0][0].text(0.02, 0.95, kwargs.get('region',''), transform=axs[0][0].transAxes, va='top', ha='left')

    plt.tight_layout()
    save_figure(fig, file_name=f'OMNI_vs_{source}_sc_{kwargs["display"]}', sub_directory='Pulkkinen')
    plt.show()
    plt.close()


# %% Compare_All

def plot_compare_sources(df_omni, df_sc, df_pc, ind_var, dep='PC', omni_colour=blue, contemp_colour=black, sc_colour=pink, restrict=True, shift_centre=True, contemp_omni=False, **kwargs):
    """
    2x2 grid comparing OMNI & in-situ as input and Index & Mag. as output
    Dep_var is either 'PC' for comparing PC and THL or 'AE' for comparing AE and SME
    """

    if df_pc is None:
        raise ValueError('Polar cap dataframe is none.')

    if df_sc is not None:
        sample_interval = df_sc.attrs.get('sample_interval','5min')
    elif df_omni is not None:
        sample_interval = df_omni.attrs.get('sample_interval','5min')
    data_type = 'mins' if sample_interval == '1min' else 'counts'

    kwargs['min_count'] = kwargs.get('min_count',minimum_counts[data_type])
    kwargs['display']   = kwargs.get('display','heat')
    kwargs['fit_type']  = kwargs.get('fit_type','saturation')
    if kwargs['display']=='rolling':
        kwargs['region'] = kwargs.get('region','sem')

    if 'data1_name' in kwargs:
        kwargs['data1_name'] = create_label(kwargs['data1_name'])

    if ind_var=='B_clock':
        shift_centre = True and shift_centre
    else:
        shift_centre = False

    dep_lags = {'PCN': 17, 'PCC': 17, 'AE': 53}
    lag = kwargs.get('lag',dep_lags.get(dep,0))

    dep_cols = {0: {'PCN': ['PCN','SMC_y_GSM'],
                    'PCC': ['PCC','SMC'],
                    'AE':  ['AE', 'SME']},
                1: {'PCN': [f'PCN_{lag}m',f'SMC_y_GSM_{lag}m'],
                    'PCC': [f'PCC_{lag}m',f'SMC_{lag}m'],
                    'AE':  [f'AE_{lag}m', f'SME_{lag}m']}}

    dep_cols = dep_cols.get(lag,dep_cols[1]) # Uses 1 for all implemented lags
    dep_vars = dep_cols.get(dep,None)
    if not dep_vars:
        raise Exception(f'"{dep}" not implemented.')

    ###----------PLOT GRIDS----------###
    n_rows, n_cols = 2, 2

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows), dpi=200, sharex='col', sharey='row')

    for i, (df, dep_var) in enumerate(it.product([df_omni,df_sc],dep_vars)):

        row, col = i % n_rows, i // n_rows
        ax = axs[row][col]

        if df is df_omni:
            if contemp_omni:
                overlap = df.index.intersection(df_sc.index)
                df = df.loc[overlap]
                colour = contemp_colour
            else:
                colour = omni_colour
        else:
            colour = sc_colour

        ind_err, ind_count = def_param_names(df, ind_var)
        dep_err, dep_count = def_param_names(df_pc, dep_var)

        bin_width, limits, invert = get_variable_range(ind_var, 'sw', dep_var=dep_var, restrict=restrict, shift_centre=shift_centre)

        if shift_centre:
            df = shift_angular_data(df, ind_var)

        df_ind = mask_df(df, ind_var, limits)
        df_dep = mask_df(df_pc, dep_var)

        intersect = df_ind.index.intersection(df_dep.index)
        df_ind = df_ind.loc[intersect]
        df_dep = df_dep.loc[intersect]

        if kwargs['display']=='heat':
            dep_bin_width = get_var_bin_width(dep_var, restrict)
            kwargs['bin_width'] = (bin_width,dep_bin_width)
            kwargs['fit_colour'] = 'cyan'
        elif kwargs['display']=='rolling':
            kwargs['window_width'] = bin_width
            kwargs['window_step']  = bin_width/10
        elif kwargs['display']=='scatter':
            kwargs['data_colour']  = colour
            kwargs['error_colour'] = colour
        kwargs['as_text'] = True

        if 'data_name_map' in kwargs:
            kwargs['data2_name'] = create_label(kwargs['data_name_map'].get(dep_var,dep_var))

        if df.attrs.get('units',{}).get(ind_var,'i')==df_pc.attrs.get('units',{}).get(dep_var,'d'):
            kwargs['reference_line'] = 'x'
        else:
            kwargs['reference_line'] = None

        # Rolling window
        objs = compare_dataframes(df_ind, df_dep, ind_var, dep_var, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_err, col2_counts=dep_count, fig=fig, ax=ax, return_objs=True, **kwargs)
        if len(objs)==3 and col!=n_cols-1: # indicates cbar present
            cbar = objs[-1]
            cbar.set_label(None)

        ###----------FORMATTING----------###
        if df is df_omni:
            title = f'OMNI (N={len(df_ind):,})'
        else:
            title = f'Spacecraft (N={len(df_ind):,})'

        # Formatting
        if shift_centre:
            ax.axvline(x=np.pi, c=grey, ls=':')
            if row==n_rows-1:
                shifted_angle_ticks(ax, 'x')

        if invert:
            ax.invert_xaxis()

        if row==0:
            ax.set_title(title)
        if row==n_rows-1:
            ax.tick_params(labelbottom=True)
        else:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel(None)

        if col!=0:
            ax.set_ylabel(None)

    if n_cols>1:
        fig.align_ylabels(axs[:,0])
    axs[0][0].text(0.02, 0.95, kwargs.get('region',''), transform=axs[0][0].transAxes, va='top', ha='left')

    file_name = f'Comparing_{ind_var}_{dep}_OMNI_sc_{lag}m'

    plt.tight_layout()
    save_figure(fig, file_name=file_name, sub_directory='Driver_Response')
    plt.show()
    plt.close()

def plot_compare_sources_with_lags(df_omni, df_sc, df_pc, ind_var, dep='PC', omni_colour=blue, contemp_colour=black, sc_colour=pink, restrict=True, shift_centre=True, contemp_omni=False, **kwargs):
    """
    2x2 grid comparing OMNI & in-situ as input and Index & Mag. as output
    Dep_var is either 'PC' for comparing PC and THL or 'AE' for comparing AE and SME
    This plots the fitted function parameters over different lags, for comparison
    """

    if df_pc is None:
        raise ValueError('Polar cap dataframe is none.')

    kwargs['fit_type']  = kwargs.get('fit_type','saturation')

    if 'data1_name' in kwargs:
        kwargs['data1_name'] = create_label(kwargs['data1_name'])


    dep_cols = {'PCN': ['PCN','SMC_y_GSM'],
                'PCC': ['PCC','SMC'],
                'AE':  ['AE', 'SME']}

    dep_vars = dep_cols.get(dep,None)
    if not dep_vars:
        raise Exception(f'"{dep}" not implemented.')


    ###----------PLOT GRIDS----------###
    n_rows, n_cols = 2, 2

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows), dpi=200, sharex='col')
    kwargs['fig'] = fig

    for i, ((df,ind_src), dep_var) in enumerate(it.product([(df_omni,'omni'),(df_sc,'sw')],dep_vars)):
        print(ind_src,dep_var)

        row, col = i % n_rows, i // n_rows
        ax = axs[row][col]

        if df is df_omni and contemp_omni:
            overlap = df.index.intersection(df_sc.index)
            df = df.loc[overlap]

        ind_err, ind_count = def_param_names(df, ind_var)
        dep_err, dep_count = def_param_names(df_pc, dep_var)

        _, limits, _ = get_variable_range(ind_var, ind_src, restrict=restrict)

        df_ind = mask_df(df, ind_var, limits)

        dep_lagged = get_lagged_columns(df_pc, dep_var)

        _, (ax, ax2) = plot_fit_params_against_z(df_ind, ind_var, dep_lagged, df_dep=df_pc, col1_err=ind_err, col2_err=dep_err, col1_counts=ind_count, col2_counts=dep_count, ax=ax, **kwargs)

        ###----------FORMATTING----------###
        ind_str = format_string(ind_var)
        dep_str = format_string(dep_var)
        ax.set_title(f'${dep_str}$ vs {ind_src.upper()} ${ind_str}$')

        if row==n_rows-1:
            ax.tick_params(labelbottom=True)
            ax.set_xlabel('Lags [mins]')
        else:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel(None)

        if col==0:
            ax2.set_ylabel(None)
        else:
            ax.set_ylabel(None)

        if i!=0:
            ax.get_legend().remove()

    if n_cols>1:
        fig.align_ylabels(axs[:,0])
    axs[0][0].text(0.02, 0.95, kwargs.get('region',''), transform=axs[0][0].transAxes, va='top', ha='left')

    file_name = f'Comparing_{ind_var}_{dep}_fit_{kwargs["fit_type"]}'

    plt.tight_layout()
    save_figure(fig, file_name=file_name, sub_directory='Driver_Response')
    plt.show()
    plt.close()