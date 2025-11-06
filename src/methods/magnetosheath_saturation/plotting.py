# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:55:08 2025

@author: richarj2
"""
import warnings
import os

def short_warn_format(message, category, filename, lineno, line=None):
    # Get just the parent folder and filename, e.g. "magnetosheath_saturation/plotting.py"
    parent = os.path.basename(os.path.dirname(filename))
    base = os.path.basename(filename)
    short_path = f'{parent}/{base}'
    return f'{short_path}:{lineno}: {category.__name__}: {message}\n'

warnings.formatwarning = short_warn_format


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, MultipleLocator


from ...plotting.space_time import plot_orbit_msh
from ...plotting.utils import save_figure, calculate_bins
from ...plotting.formatting import data_string, create_label, add_legend
from ...plotting.comparing.parameter import compare_columns
from ...plotting.config import black, blue, grey, pink
#from ...plotting.distributions import plot_fit

minimum_counts = {'mins': 100, 'counts': 50}



# %%


imf_cols = ['B_avg', 'B_x_GSE', 'B_y_GSE', 'B_z_GSE', 'B_y_GSM', 'B_z_GSM', 'B_avg_rms', 'B_vec_rms', 'R_x_BSN', 'R_y_BSN', 'R_z_BSN', 'prop_time_s', 'E_y', 'M_A', 'beta']

plasma_cols = ['P_flow', 'n_p', 'T_p', 'na_np_ratio', 'V_flow', 'V_x_GSE', 'V_y_GSE', 'V_z_GSE', 'R_x_GSE', 'R_y_GSE', 'R_z_GSE', 'E_y', 'M_A', 'M_ms', 'beta', 'E_mag', 'E_x_GSM', 'E_y_GSM', 'E_z_GSM', 'S_mag', 'S_x_GSM', 'S_y_GSM', 'S_z_GSM', 'E_R']

def def_param_names(df, variable, source):

    ###----------PARAMETER NAMINGS----------###

    param       = '_'.join((variable,source))
    if source in ('sw','pc'):
        param_err = None # Need to include
    else:
        param_err   = '_'.join((variable,'unc',source))

    if param_err not in df:
        param_err = None

    var_count = None
    param_count = None

    if source in ('sw','pc'):
        if variable in plasma_cols:
            var_count   = 'plasma_counts'
            param_count = 'plasma_counts_sw'
        elif variable in imf_cols:
            var_count   = 'imf_counts'
            param_count = 'imf_counts_sw'

    elif '_GS' in variable:
        field, _, coords = variable.split('_')
        param_count = '_'.join((field,coords,'count',source))

    else:
        param_count = '_'.join((variable,'count',source))

    if param_count not in df:
        param_count = None

    return param, param_err, var_count, param_count

def ind_variable_range(ind_var, ind_src, dep_var=None, restrict=True, bounds=None):

    limits = [None, None]
    invert = False

    if ind_var=='B_avg':
        bin_step, limits[0] = 2, 0
        if restrict:
            bin_step, limits[1] = 1, 15

    elif ind_var=='B_parallel':
        bin_step, limits[0] = 2, 0
        if restrict:
            bin_step, limits[1] = 1, 80

    elif 'B_y' in ind_var:
        bin_step, limits[0] = 2, 0
        if restrict:
            bin_step, limits[1] = 1, 25

    elif 'B_' in ind_var:
        invert = True
        bin_step, limits[1] = 2, 0
        if restrict:
            bin_step = 1
            if ind_src=='msh':
                limits[0] = -80
            elif ind_var==dep_var:
                limits[0] = -20
            else:
                limits[0] = -40

    elif ind_var=='V_flow':
        bin_step, limits[0] = 50, 250
        if restrict:
            bin_step, limits[1] = 50, 800
            if ind_src=='msh':
                limits[1] = 750

    elif 'V_' in ind_var:
        invert = True
        bin_step, limits[1] = 50, 0

    elif ind_var=='E_parallel':
        bin_step, limits[0] = 2, 0
        if restrict:
            bin_step = 1
            limits[1] = 14

    elif 'E_' in ind_var:
        bin_step, limits[0] = 2, 0
        if restrict:
            bin_step = 1
            if ind_src=='msh':
                limits[1] = 20
            elif (dep_var is not None) and (
                    (ind_var==dep_var) or (dep_var.startswith('E_parallel'))):
                bin_step, limits[1] = 0.5, 10
            else:
                limits[1] = 20

    elif 'N_' in ind_var:
        bin_step, limits[0] = 5, 0
        if restrict:
            bin_step = 2
            if ind_src=='msh':
                limits[1] = 100
            elif ind_var==dep_var:
                limits[1] = 40
            else:
                limits[1] = 75

    elif ind_var=='S_perp':
        invert = True
        bin_step, limits[1] = 5, 0
        if restrict:
            limits[0] = -150

    elif 'S_' in ind_var:
        bin_step, limits[0] = 10, 0
        if restrict:
            bin_step, limits[1] = 5, 100
            if ind_src=='msh':
                bin_step, limits[1] = 20, 600

    elif 'M_A' in ind_var:
        bin_step, limits[0] = 5, 0
        if restrict:
            bin_step, limits[1] = 5, 50

    elif 'beta' in ind_var:
        bin_step, limits[0] = 1, 0
        if restrict:
            limits[1] = 30

    else:
        raise ValueError(f'"{ind_var} not implemented.')

    if bounds is not None:
        limits = bounds

    return bin_step, limits, invert

def grp_param_splitting(df, grp_var, grp_param, grp_unit, **kwargs):

    grp_split = kwargs.get('grp_split',None)
    quantiles = kwargs.get('quantiles',2)

    used_median = False

    if grp_unit in ('rad','deg','°'):
        z_unit_str = '°'
    elif grp_unit is not None and grp_unit not in ('1','NUM',''):
        z_unit_str = f' {grp_unit}'
    else:
        z_unit_str = ''

    grp_string = data_string(grp_var)
    grp_label = create_label(grp_var, '°' if grp_unit in ('rad','deg','°') else grp_unit)


    if grp_split is not None:
        edges = [grp_split]

    else:
        found_split = False

        if grp_var not in ('Delta B_theta','M_A'): # Using defined boundaries
            found_split = True
            if grp_var=='theta_Bn':
                edges = [np.pi/4]

            elif grp_var=='B_clock':
                edges = [0]

            elif grp_var=='Delta B_z':
                edges = [0]

            elif 'E_' in grp_var:
                edges = [0]

            elif 'V_' in grp_var:
                edges = [400]

            elif 'B_' in grp_var:
                edges = [0]

            elif 'beta' in grp_var:
                edges = [1]

            else:
                found_split = False

        if not found_split:
            used_median = True
            if grp_var not in ('Delta B_theta','M_A'):
                warnings.warn(f'Grouping parameter "{grp_param}" not implemented.')

            median = np.percentile(df[grp_param].dropna().to_numpy(),50)

            if quantiles==4:
                quar_1 = np.percentile(df[grp_param].dropna().to_numpy(),25)
                quar_3 = np.percentile(df[grp_param].dropna().to_numpy(),75)
                edges = [quar_1,median,quar_3]
            elif quantiles==3:
                tert_1 = np.percentile(df[grp_param].dropna().to_numpy(),100/3)
                tert_2 = np.percentile(df[grp_param].dropna().to_numpy(),200/3)
                edges = [tert_1,tert_2]
            else:
                edges = [median]

    if grp_var in ('Delta B_theta','theta_Bn','B_clock'):
        bin_width = np.pi/36

    elif grp_var=='Delta B_z':
        bin_width = 10

    elif grp_var=='M_A':
        bin_width = 1

    elif 'E_' in grp_var:
        bin_width = 0.5

    elif 'V_' in grp_var:
        bin_width = 50

    elif 'B_' in grp_var:
        bin_width = 1

    elif 'beta' in grp_var:
        bin_width = 0.1

    else:
        edges = [median]
        minimum = df[grp_param].dropna().min()
        maximum = df[grp_param].dropna().max()
        bin_width = int(np.log10(maximum-minimum))/20

    if grp_unit in ('rad','deg','°'):

        if len(edges)==1:
            z_labels = [f'${grp_string}$<{np.degrees(edges[0]):.1f}{z_unit_str}',
                        f'${grp_string}$$\\geq${np.degrees(edges[0]):.1f}{z_unit_str}',
                        f'${grp_string}$ = {np.degrees(edges[0]):.1f}{z_unit_str}']
        else:
            z_labels = []
            for edge in edges:
                z_labels.append(f'${grp_string}$={np.degrees(edge):.1f}{z_unit_str}')
    else:

        if len(edges)==1:
            z_labels = [f'${grp_string}$<{edges[0]:.1f}{z_unit_str}',
                        f'${grp_string}$$\\geq${edges[0]:.1f}{z_unit_str}',
                        f'${grp_string}$ = {edges[0]:.1f}{z_unit_str}']
        else:
            z_labels = []
            for edge in edges:
                z_labels.append(f'${grp_string}$={edge:.1f}{z_unit_str}')

    return edges, bin_width, grp_label, z_labels, used_median

def plot_saturation_overview(df_sw, df_msh, ind_var, dep_var, grp_var, ind_src='sw', dep_src='pc', grp_src='msh', data_type='counts', plot_test=False, same_var=False, invert_x=False, invert_y=False, **kwargs):

    if data_type=='mins':
        kwargs['min_count'] = 100
    elif not kwargs.get('min_count',None):
        kwargs['min_count'] = 50

    ind_param, ind_param_err, ind_var_count, ind_param_count = def_param_names(df_msh, ind_var, ind_src)
    dep_param, dep_param_err, dep_var_count, dep_param_count = def_param_names(df_msh, dep_var, dep_src)
    group_param = f'{grp_var}_{grp_src}'

    ###----------MASKS----------###

    mask      = np.ones(len(df_msh),dtype=bool)
    omni_mask = np.ones(len(df_sw),dtype=bool)

    bin_step, limits, invert = ind_variable_range(ind_var, ind_src, dep_var=dep_var)
    kwargs['window_width'] = bin_step

    # MSH Dataset
    if limits[0] is not None:
        mask  &= df_msh[ind_param] >= limits[0]
    if limits[-1] is not None:
        mask  &= df_msh[ind_param] <= limits[1]
    mask &= ~df_msh[[ind_param, dep_param, group_param]].isna().any(axis=1)

    df_masked  = df_msh.loc[mask]

    # OMNI dataset
    if ind_src != 'msh':
        if dep_var in df_sw:
            omni_mask &= ~df_sw[[ind_var, dep_var]].isna().any(axis=1)
        else:
            omni_mask &= ~df_sw[ind_var].isna()
        if limits[0] is not None:
            omni_mask &= df_sw[ind_var] >= limits[0]
        if limits[-1] is not None:
            omni_mask &= df_sw[ind_var] <= limits[1]

    df_omni_masked = df_sw.loc[omni_mask]

    ###----------GROUPING PARAMETER LABEL----------###

    z_unit = df_msh.attrs['units'].get(group_param,'')
    edges, bin_width, grp_label, z_labels, used_median = grp_param_splitting(df_masked, grp_var, group_param, z_unit, **kwargs)
    kwargs['zs_edges'] = edges

    if plot_test:

        fig, ax = plt.subplots()

        # Splits contemp MSH OMNI by grouping
        kwargs.update({'display': 'rolling_multiple', 'region': 'sem'})
        compare_columns(df_masked, ind_param, dep_param, col1_err=ind_param_err, col1_counts=ind_param_count, col2_err=dep_param_err, col2_counts=dep_param_count, col3=group_param, want_legend=True, ax=ax, return_objs=False, **kwargs)
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
        ax_bl2.hist(df_omni_masked[ind_var], bins=calculate_bins(df_omni_masked[ind_var],bin_step), color=omni_colour)

    if not (ind_src=='msh' or dep_src=='msh'):

        # All OMNI Driver vs Response
        _ = compare_columns(df_omni_masked, ind_var, dep_var, col1_counts=ind_var_count, col2_counts=dep_var_count, data_colour=omni_colour,  ax=ax_tl, **kwargs)

    kwargs.update({'col1_counts': ind_param_count, 'col1_err': ind_param_err, 'col2_counts': dep_param_count, 'col2_err': dep_param_err, 'col3': group_param})
    for i, (ax, ls, reg) in enumerate(zip((ax_tl,ax_bl),('-',':'),('sem','none'))):

        # OMNI with contemp. MSH times
        kwargs.update({'region': reg, 'ax': ax, 'line_style': ls})
        _ = compare_columns(df_masked, ind_param, dep_param, data_colour=full_colour, error_colour=full_colour,  **kwargs)

    # Splits contemp MSH OMNI by grouping
    kwargs.update({'display': 'rolling_multiple', 'region': 'sem', 'ax': ax_bl})
    _ = compare_columns(df_masked, ind_param, dep_param, want_legend=True, **kwargs)

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
        elif grp_var=='beta':
            ax_br.set_xscale('symlog', linthresh=1, linscale=1)
        ax_br.set_xlabel(grp_label)

        # Counts each ind_var bin
        df_msh_param = df_masked.loc[filter_mask,ind_param]
        ax_bl2.hist(df_msh_param, bins=calculate_bins(df_msh_param,bin_step), histtype='step', edgecolor=colour, linewidth=1.2)

        # Counts each year split by grouping
        df_msh_years = df_masked.loc[filter_mask].index.year.to_numpy()
        bins = calculate_bins(df_msh_years,1)
        counts, _ = np.histogram(df_msh_years, bins=bins)

        ax_br2.bar(bins[:-1] + (i-0.5)*0.5, counts, width=0.5, color=colour, label=f'{len(df_msh_years):,}{data_type}' if not used_median else None)

    ax_bl2.axhline(kwargs['min_count'], c='k', ls='-')
    ax_bl2.axhline(kwargs['min_count'], c='w', ls=':')

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

    if invert or invert_x:
        ax_bl.invert_xaxis()
    if (invert and ind_var == dep_var) or invert_y:
        ax_tl.invert_yaxis()
        ax_bl.invert_yaxis()


    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

# %% Compare_sw_msh_response

def plot_compare_responses(df_sw, df_msh, ind_var, dep_var, dep_src='pc', sw_colour=black, msh_colour=pink, bounds=None, restrict=True, data_type='counts', min_count=50, show_contemp=True, compare_sw_msh=False, compare_colour='purple', **kwargs):
    """
    Plots sw vs pc and msh vs pc and, if flagged, sw vs msh
    """

    msh_map = kwargs.get('msh_map',{})

    if data_type=='mins':
        min_count = 100

    kwargs['min_count'] = min_count
    kwargs['display']   = 'rolling'
    kwargs['region']    = 'sem'

    dep_param, dep_param_err, dep_var_count, dep_param_count = def_param_names(df_msh, dep_var, dep_src)

    ###----------PLOT GRIDS----------###

    n_cols = 2 + int(compare_sw_msh)
    fig, axs = plt.subplots(2, n_cols, figsize=(8*n_cols, 10), dpi=200, height_ratios=[3,2])

    # [row] [col]
    axs[0][0].sharex(axs[1][0])
    axs[0][1].sharex(axs[1][1])
    axs[0][0].sharey(axs[0][1])

    if compare_sw_msh:
        axs[0][2].sharex(axs[1][2])

    enumerator = zip((df_sw, df_msh), ('sw', 'msh'), (dep_var, dep_param), (sw_colour, msh_colour))
    if compare_sw_msh:
        enumerator = zip((df_sw, df_msh, df_msh), ('sw', 'msh', 'sw'), (dep_var, dep_param, ind_var), (sw_colour, msh_colour, compare_colour))

    for i, (df, source, dep, colour), in enumerate(enumerator):

        ax0 = axs[0][i]
        ax1 = axs[1][i]

        independent = ind_var
        if source=='msh':
            independent = msh_map.get(ind_var,ind_var)

        ind_param, ind_param_err, ind_var_count, ind_param_count = def_param_names(df, independent, source)
        bin_step, limits, invert = ind_variable_range(ind_var, source, dep_var=dep, restrict=restrict)
        if i==2:
            variable = msh_map.get(dep,dep)
            dep_param, dep_param_err, _, dep_param_count = def_param_names(df_msh, variable, 'msh')
            dep = dep_param

        if source=='sw' and i!=2:
            ind = ind_var
            ind_err = None
            ind_count = ind_var_count
        else:
            ind = ind_param
            ind_err = ind_param_err
            ind_count = ind_param_count

         # Overwrites with those passed in
        if bounds is not None:
            limits = bounds

        mask = ~df[[ind, dep]].isna().any(axis=1)

        if limits[0] is not None:
            mask &= df[ind] >= limits[0]
        if limits[-1] is not None:
            mask &= df[ind] <= limits[1]

        df_masked = df.loc[mask]

        kwargs['window_width'] = bin_step
        kwargs['data_colour']  = colour
        kwargs['error_colour'] = colour

        kwargs_source = kwargs.copy()
        if source=='msh':
            if ind_var in msh_map:
                kwargs_source['data1_name'] = independent

        if i==2:
            ax0.axline((limits[0],limits[0]), slope=1, c=black, ls=':')
            ax0.grid(ls=':', c=grey, lw=0.5)
            if ind_var in msh_map:
                independent = msh_map.get(ind_var,ind_var)
                kwargs_source['data1_name'] = independent
            kwargs_source['data2_name'] = create_label(f'{kwargs_source["data1_name"]}_msh')
            kwargs_source['data1_name'] = kwargs['data1_name']

        kwargs_source['data1_name'] = create_label(f'{kwargs_source["data1_name"]}_{source}')

        _ = compare_columns(df_masked, ind, dep, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_param_err, col2_counts=dep_param_count, fig=fig, ax=ax0, return_objs=True, **kwargs_source)

        # Counts
        ax1.axhline(min_count, c='k', ls='-')
        ax1.axhline(min_count, c='w', ls=':')
        ax1.hist(df_masked[ind], bins=calculate_bins(df_masked[ind],bin_step), color=colour)
        ax1.set_yscale('log')

        if invert:
            ax0.invert_xaxis()
            if i==2:
                ax0.invert_yaxis()

        if source=='sw' and show_contemp and i!=2:

            mask &= df.index.isin(df_msh.index)

            df_masked = df.loc[mask]

            kwargs_source['data_colour']  = blue
            kwargs_source['error_colour'] = blue

            _ = compare_columns(df_masked, ind, dep, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_param_err, col2_counts=dep_param_count, fig=fig, ax=ax0, return_objs=True, **kwargs_source)
            ax1.hist(df_masked[ind], bins=calculate_bins(df_masked[ind],bin_step), color=blue)

    axs[0][1].set_ylabel(None)
    axs[1][0].set_ylabel(data_type.capitalize())


    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

# %% Lags
def plot_different_lags(df, ind_var, dep_var='AE', ind_src='sw', dep_src='pc', df_type='omni', bounds=None, restrict=True, data_type='counts', min_count=50, **kwargs):

    if data_type=='mins':
        min_count = 100

    kwargs['min_count'] = min_count
    kwargs['display']   = 'rolling'
    kwargs['region']    = 'sem'

    ind_param, ind_param_err, ind_var_count, ind_param_count = def_param_names(df, ind_var, ind_src)
    if df_type=='omni':
        ind, ind_err, ind_count = ind_var, None, ind_var_count
    else:
        ind, ind_err, ind_count = ind_param, ind_param_err, ind_param_count

    bin_step, limits, invert = ind_variable_range(ind_var, ind_src, restrict=restrict)
    kwargs['window_width'] = bin_step

     # Overwrites with those passed in
    if bounds is not None:
        limits = bounds

    mask = ~df[ind].isna()
    if limits[0] is not None:
        mask &= df[ind] >= limits[0]
    if limits[-1] is not None:
        mask &= df[ind] <= limits[1]

    dep_cols = [col for col in df.columns if col.startswith(dep_var)]

    cmap = plt.get_cmap('cool')
    norm = plt.Normalize(vmin=0, vmax=len(dep_cols)-1)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)

    for i, col in enumerate(dep_cols):

        dep = col

        if df_type=='omni':
            try:
                _, lag = col.split('_')
            except:
                lag = '0m'
        else:
            try:
                _, lag, _ = col.split('_')
            except:
                lag = '0m'

        mask_dep = mask.copy()
        mask_dep &= ~df[dep].isna()

        df_masked = df.loc[mask_dep]

        colour = cmap(norm(i))
        kwargs['data_colour'] = colour
        kwargs['error_colour'] = colour

        _ = compare_columns(df_masked, ind, dep, col1_err=ind_err, col1_counts=ind_count, fig=fig, ax=ax, return_objs=True, **kwargs)
        ax.plot([], [], ls='-', color=colour, label=lag)


    ax.set_ylabel(create_label(dep_var,units=df.attrs['units']))

    add_legend(fig, ax)
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()



def plot_grouping_cause(df_merged, ind_var, dep_var, ind_src='sw', dep_src='msh', bounds=None, restrict=True, data_type='counts', min_count=50, **kwargs):

    if data_type=='mins':
        min_count = 100

    kwargs['min_count']    = min_count
    kwargs['display']      = 'rolling'
    kwargs['data_colour']  = black
    kwargs['error_colour'] = black


    ind_param, ind_param_err, _, ind_param_count = def_param_names(df_merged, ind_var, ind_src)
    dep_param, dep_param_err, _, dep_param_count = def_param_names(df_merged, dep_var, dep_src)

    bin_step, limits, invert = ind_variable_range(ind_var, ind_src, restrict)

     # Overwrites with those passed in
    if bounds is not None:
        limits = bounds

    mask = ~df_merged[[ind_param, dep_param]].isna().any(axis=1)

    if limits[0] is not None:
        mask &= df_merged[ind_param] >= limits[0]
    if limits[-1] is not None:
        mask &= df_merged[ind_param] <= limits[1]

    df_masked = df_merged.loc[mask]

    kwargs['window_width'] = bin_step


    compare_columns(df_masked, ind_param, dep_param, col1_err=ind_param_err, col1_counts=ind_param_count, col2_err=dep_param_err, col2_counts=dep_param_count, return_objs=False, **kwargs)

# %% Compare_OMNI

def plot_compare_sc_omni(df_omni, df_sc, ind_var, *dep_vars, dep_src='pc', omni_colour=black, contemp_colour=blue, sc_colour=pink, bounds=None, restrict=True, data_type='counts', min_count=None, **kwargs):

    if min_count is None:
        min_count = minimum_counts[data_type]

    kwargs['min_count'] = min_count
    kwargs['display']   = 'rolling'
    kwargs['region']    = 'sem'
    if 'data1_name' in kwargs:
        name = kwargs['data1_name']
        if df_sc is not None:
            name = f'{name}_sw'
        kwargs['data1_name'] = create_label(name)

    ###----------PLOT GRIDS----------###

    n_cols = len(dep_vars)
    fig, axs = plt.subplots(2, n_cols, figsize=(8*n_cols, 10), dpi=200, height_ratios=[3,2], sharex=True)
    groups = {}
    for i, ax in enumerate(axs[0]):
        prefix = dep_vars[i][:2]
        if prefix not in groups:
            groups[prefix] = ax
        else:
            axs[0][i].sharey(groups[prefix])

    for i, dep_var in enumerate(dep_vars):

        ax0 = axs[0][i]
        ax1 = axs[1][i]

        for df, source, colour in zip((df_omni,df_sc,df_sc), ('omni','sw','sc'), (omni_colour, contemp_colour, sc_colour)):

            if df is None:
                continue

            ind_param, ind_param_err, ind_var_count, ind_param_count = def_param_names(df, ind_var, source)
            dep_param, dep_param_err, dep_var_count, dep_param_count = def_param_names(df, dep_var, dep_src)

            if source=='omni':
                ind, ind_err, ind_count = ind_var, None, ind_var_count
                dep, dep_err, dep_count = dep_var, None, None

            else:
                ind, ind_err, ind_count = ind_param, ind_param_err, ind_param_count
                dep, dep_err, dep_count = dep_param, dep_param_err, dep_param_count

            bin_step, limits, invert = ind_variable_range(ind_var, source, dep_var=dep, restrict=restrict)

             # Overwrites with those passed in
            if bounds is not None:
                limits = bounds

            mask = ~df[[ind, dep]].isna().any(axis=1)

            if limits[0] is not None:
                mask &= df[ind] >= limits[0]
            if limits[-1] is not None:
                mask &= df[ind] <= limits[1]

            df_masked = df.loc[mask]

            kwargs['window_width'] = bin_step
            kwargs['data_colour']  = colour
            kwargs['error_colour'] = colour
            if 'data_name_map' in kwargs:
                name = kwargs['data_name_map'].get(dep_var,dep)
                kwargs['data2_name'] = create_label(name)


            _ = compare_columns(df_masked, ind, dep, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_err, col2_counts=dep_count, fig=fig, ax=ax0, return_objs=True, **kwargs)

            # Counts
            if source=='omni':
                ax1.hist(df_masked[ind], bins=calculate_bins(df_masked[ind],bin_step), color=colour)
            else:
                ax1.hist(df_masked[ind], bins=calculate_bins(df_masked[ind],bin_step), histtype='step', edgecolor=colour, linewidth=1.2)

        ax1.axhline(min_count, c='k', ls='-')
        ax1.axhline(min_count, c='w', ls=':')
        ax1.set_yscale('log')

        if invert:
            ax0.invert_xaxis()


    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()