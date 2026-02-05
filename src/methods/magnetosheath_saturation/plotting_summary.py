# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:55:08 2025

@author: richarj2
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, MultipleLocator

from .plotting_utils import minimum_counts, def_param_names, get_variable_range, grp_param_splitting, shift_angular_data, mask_df

from ...plotting.space_time import plot_orbit_msh
from ...plotting.utils import save_figure, calculate_bins
from ...plotting.formatting import create_label, shifted_angle_ticks
from ...plotting.comparing.parameter import compare_dataframes
from ...plotting.config import black, blue, grey, pink, green



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
