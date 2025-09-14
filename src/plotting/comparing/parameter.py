# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:27:55 2025

@author: richarj2
"""

import numpy as np
import pandas as pd
import itertools as it

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import is_color_like
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from collections import Counter

from ..config import black, white, blue, scatter_markers
from ..formatting import add_legend, add_figure_title, create_label, dark_mode_fig, data_string
from ..utils import save_figure, calculate_bins, change_series_name
from ..distributions import plot_fit

from ...analysing.calculations import average_of_averages
from ...analysing.comparing import difference_columns

from ...processing.speasy.config import colour_dict, database_colour_dict

def is_marker_like(marker):
    try:
        MarkerStyle(marker)
        return True
    except ValueError:
        return False

def compare_columns(df, col1, col2, col3=None, col1_err=None, col2_err=None, col1_counts=None, col2_counts=None, delay_time=None, **kwargs):


    series1 = df.loc[:,col1]
    series2 = df.loc[:,col2]

    if col3 is not None:
        kwargs['zs'] = df.loc[:,col3]

    if col1_err is not None:
        kwargs['xs_unc'] = df.loc[:,col1_err]

    if col2_err is not None:
        kwargs['ys_unc'] = df.loc[:,col2_err]

    if col1_counts is not None:
        kwargs['xs_counts'] = df.loc[:,col1_counts]

    if col2_counts is not None:
        kwargs['ys_counts'] = df.loc[:,col2_counts]

    if delay_time is not None:
        delay = pd.Timedelta(minutes=delay_time)

        series2 = series2.shift(freq=delay)

        for y_param in ('ys_unc','ys_counts'):
            if y_param in kwargs:
                kwargs[y_param] = kwargs[y_param].shift(freq=delay)

        series1, series2 = series1.align(series2)

        for x_param in ('zs','xs_unc','xs_counts'):
            if x_param in kwargs:
                _, kwargs[x_param] = series1.align(kwargs[x_param])

    compare_series(series1, series2, **kwargs)


def investigate_difference(df, col1, col2, ind_col, **kwargs):

    diff_type = kwargs.get('diff_type','absolute')


    series1 = df.loc[:,ind_col]
    series2 = difference_columns(df, col2, col1, diff_type)

    kwargs['reference_line'] = 0
    if kwargs.get('ind_name_lat',None) is not None:
        change_series_name(series1, kwargs.get('ind_name_lat'))
    if kwargs.get('ind_name_str',None) is not None:
        kwargs['data1_name'] = kwargs.get('ind_name_str')

    compare_series(series1, series2, **kwargs)

def compare_series(series1, series2, **kwargs):

    display         = kwargs.get('display','scatter')
    bin_width       = kwargs.get('bin_width',None)
    want_legend     = kwargs.get('want_legend',True)
    reference_line  = kwargs.get('reference_line',None)
    add_count       = kwargs.get('add_count',False)

    data1_name    = kwargs.get('data1_name',None)
    data2_name    = kwargs.get('data2_name',None)
    brief_title   = kwargs.get('brief_title','')

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',False)

    is_heat = False
    if display == 'heat':
        is_heat = True

    ###---------------CONSTRUCT COLUMN LABELS---------------###

    data1_str = data_string(series1.name)
    data2_str = data_string(series2.name)
    unit1 = series1.attrs.get('units',{}).get(series1.name, None)
    unit2 = series2.attrs.get('units',{}).get(series2.name, None)

    data1_label = create_label(data1_str, unit=unit1, data_name=data1_name)
    data2_label = create_label(data2_str, unit=unit2, data_name=data2_name)
    brief_title = f'Comparing ${data1_str}$ and ${data2_str}$' if brief_title is None else brief_title


    ###---------------PLOTS MAIN SCATTER/HEAT DATA---------------###
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        kwargs['fig'] = fig
        kwargs['ax'] = ax

    if display == 'scatter_dict':
        try:
            _ = plot_scatter_with_dict(series1, series2, **kwargs)
        except:
            display = 'scatter'
            print('Using normal scatter.')

    if display == 'scatter':
        _ = plot_scatter(series1, series2, **kwargs)

    elif display == 'scatter_binned':
        _ = plot_scatter_binned(series1, series2, **kwargs)

    elif display == 'scatter_binned_multiple':
        _ = plot_scatter_binned_multiple(series1, series2, **kwargs)

    elif display == 'heat':
        if hasattr(bin_width, '__len__') and len(bin_width) == 2:
            n_bins = (calculate_bins(series1,bin_width[0]), calculate_bins(series2,bin_width[1]))
        else:
            n_bins = (calculate_bins(series1,bin_width), calculate_bins(series2,bin_width))
        h = ax.hist2d(series1, series2, bins=n_bins, norm=mpl.colors.LogNorm(), cmap='hot')

        cbar = fig.colorbar(h[3], ax=ax)
        cbar.ax.tick_params(colors=black)
        cbar.set_label('Number of Points', color=black)
        cbar.outline.set_edgecolor(black)

        line_color = 'w'

    elif display != 'scatter_dict':
        raise ValueError(f'"{display}" not valid display mode.')

    if reference_line is not None:
        if reference_line=='x':
            ax.axline((0, 0), slope=1, color=line_color, label='y=x', lw=2, ls=':')
        elif isinstance(reference_line, int) or isinstance(reference_line, int):
            ax.axhline(y=reference_line, color=line_color, label=f'y={reference_line}', lw=2, ls=':')

    fit_kwargs = kwargs.copy()
    fit_kwargs['fit_colour'] = black
    fit_kwargs['fit_style'] = '--'
    fit_kwargs['fit_width'] = 1.25

    _ = plot_fit(series1,series2,**fit_kwargs)

    ###---------------CONVERT TICKS TO DEGREES---------------###
    def rad2deg(x, pos):
            return f'{np.degrees(x):.0f}'

    if series1.attrs.get('units',{}).get(series1.name,None) == 'rad':

        ax.xaxis.set_major_formatter(FuncFormatter(rad2deg))
        ax.set_xticks(np.linspace(-np.pi, np.pi, 9))  # every 45°

        data1_label = create_label(data1_str, unit='°', data_name=data1_name)

    if series2.attrs.get('units',{}).get(series2.name,None) == 'rad':

        ax.yaxis.set_major_formatter(FuncFormatter(rad2deg))
        ax.set_yticks(np.linspace(-np.pi, np.pi, 9))

        data2_label = create_label(data2_str, unit='°', data_name=data2_name)

    ax.set_xlabel(data1_label, c=black)
    ax.set_ylabel(data2_label, c=black)

    if brief_title=='amount':
        brief_title = f'{len(series1):,}'
    elif brief_title != '' and add_count:
        brief_title += f', N={len(series1):,}'

    ###---------------LABELLING AND FINISHING TOUCHES---------------###
    add_legend(fig, ax, legend_on=want_legend, heat=is_heat)
    add_figure_title(fig, black, brief_title, ax=ax)
    dark_mode_fig(fig,black,white,is_heat)
    plt.tight_layout();

    if return_objs:
        try:
            return fig, ax, cbar
        except:
            return fig, ax

    save_figure(fig)
    plt.show()
    plt.close()

def plot_scatter(xs, ys, **kwargs):

    xs_unc       = kwargs.get('xs_unc',None)
    ys_unc       = kwargs.get('ys_unc',None)
    zs           = kwargs.get('zs',None)

    data_colour  = kwargs.get('data_colour',blue)
    error_colour = kwargs.get('error_colour','k')
    scat_size    = kwargs.get('scatter_size',0.5)
    scat_marker  = kwargs.get('scat_marker','o')
    zero_lines   = kwargs.get('zero_lines',False)

    z_min        = kwargs.get('z_min',None)
    z_max        = kwargs.get('z_max',None)

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)

    if fig is None or ax is None:
        fig, ax = plt.subplots()


    if xs_unc is not None or ys_unc is not None:
        ax.errorbar(xs, ys, xerr=xs_unc, yerr=ys_unc, fmt='.', ms=0, ecolor=error_colour, capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

    if not is_marker_like(scat_marker):
        scat_marker = 'o'

    if zs is not None:
        # Using a dictionary to map the colours
        if data_colour in ('spacecraft','detector','database'):

            if data_colour == 'database':
                sc_colours = database_colour_dict
            else:
                sc_colours = colour_dict

            spacecraft_counts = Counter(zs)
            colours = pd.Series(zs).map(sc_colours).fillna(black).to_numpy()

            scatter = ax.scatter(xs, ys, c=colours, marker=scat_marker, s=scat_size)

            legend_elements = [Line2D([0], [0], marker='o', color=colour, label=f'{label}: {spacecraft_counts.get(label, 0)}', markersize=1, linestyle='None') for label, colour in sc_colours.items() if spacecraft_counts.get(label, 0) > 0]

            ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

        else:
            # Using a z-value to apply a continuous colour map
            cmap = plt.cm.get_cmap('cool').copy()
            cmap.set_bad(color='black')
            scatter = ax.scatter(xs, ys, c=zs, cmap=cmap, vmin=z_min, vmax=z_max, marker=scat_marker, s=scat_size)

            z_str = data_string(zs.name)
            z_unit = zs.attrs.get('units',{}).get(zs.name, None)
            z_label = create_label(z_str, unit=z_unit)

            cbar = plt.colorbar(scatter)
            cbar.set_label(z_label)

    elif is_color_like(data_colour):
        ax.scatter(xs, ys, c=data_colour, marker=scat_marker, s=scat_size)

    else:
        ax.scatter(xs, ys, c=blue, marker=scat_marker, s=scat_size)

    if zero_lines:
        ax.axhline(0,c='grey',ls=':')
        ax.axvline(0,c='grey',ls=':')

    return fig, ax

def plot_scatter_binned(xs, ys, **kwargs):

    zs           = kwargs.get('zs',None)

    xs_unc       = kwargs.get('xs_unc',None)
    ys_unc       = kwargs.get('ys_unc',None)
    zs_unc       = kwargs.get('zs_unc',None)

    xs_counts    = kwargs.get('xs_counts',None)
    ys_counts    = kwargs.get('ys_counts',None)
    zs_counts    = kwargs.get('zs_counts',None)

    data_colour  = kwargs.get('data_colour',blue)
    error_colour = kwargs.get('error_colour','k')
    scat_size    = kwargs.get('scatter_size',0.5)
    scat_marker  = kwargs.get('scat_marker','o')

    bin_step     = kwargs.get('bin_step',1)
    z_min        = kwargs.get('z_min',None)
    z_max        = kwargs.get('z_max',None)

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)

    z_min = np.min(zs) if z_min is None else z_min
    z_max = np.max(zs) if z_max is None else z_max

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if not is_marker_like(scat_marker):
        scat_marker = 'o'

    for X_bin in calculate_bins(xs,bin_step):

        mask = (xs>=X_bin) & (xs<(X_bin+bin_step))

        if np.sum(mask)<2: # So a standard dev can be calculated
            continue

        x = average_of_averages(xs, xs_unc, xs_counts, mask)
        y = average_of_averages(ys, ys_unc, ys_counts, mask)

        if (isinstance(x, float) and np.isnan(x)) or (isinstance(y, float) and np.isnan(y)):
            continue

        ax.errorbar(x.n, y.n, xerr=x.s, yerr=y.s, fmt='.', ms=0, ecolor=error_colour, capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

        if zs is not None:

            z = average_of_averages(zs, zs_unc, zs_counts, mask)

            scatter = ax.scatter(x.n, y.n, c=z.n, cmap='cool', marker=scat_marker, vmin=z_min, vmax=z_max, s=scat_size)

        else:
            colour = data_colour if is_color_like(data_colour) else blue

            ax.scatter(x.n, y.n, c=colour, s=scat_size)

    if scatter:
        cbar = plt.colorbar(scatter)

        z_str = data_string(zs.name)
        z_unit = zs.attrs.get('units',{}).get(zs.name, None)
        z_label = create_label(z_str, unit=z_unit)

        cbar.set_label(z_label)

    return fig, ax, cbar

def plot_scatter_binned_multiple(xs, ys, **kwargs):

    zs           = kwargs.get('zs',None)
    zs_edges     = kwargs.get('zs_edges',None)

    xs_unc       = kwargs.get('xs_unc',None)
    ys_unc       = kwargs.get('ys_unc',None)

    xs_counts    = kwargs.get('xs_counts',None)
    ys_counts    = kwargs.get('ys_counts',None)

    data_colour  = kwargs.get('data_colour',blue)
    error_colour = kwargs.get('error_colour','k')
    scat_size    = kwargs.get('scatter_size',0.5)

    bin_step     = kwargs.get('bin_step',1)
    z_min        = kwargs.get('z_min',None)
    z_max        = kwargs.get('z_max',None)

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)


    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if zs_edges is not None and zs is not None:

        z_unit = zs.attrs.get('units',{}).get(zs.name, None)
        if z_unit is not None and z_unit not in ('1','NUM',''):
            z_unit = f' {z_unit}'
        else:
            z_unit = ''

        if len(zs_edges)==1:
            z_min = np.min(zs) if z_min is None else z_min
            z_max = np.max(zs) if z_max is None else z_max

            zs_bins  = [[np.min(zs),zs_edges[0]],[zs_edges[0],np.max(zs)]]
            z_labels = [f'{zs.name}<{zs_edges[0]}{z_unit}',f'{zs.name}$\\geq${zs_edges[0]}{z_unit}']
            z_vals   = [z_min, z_max]
        else:
            z_min = zs_edges[0]
            z_max = zs_edges[-1]

            zs_edges = np.array(zs_edges)
            zs_bins  = []
            z_labels = []
            z_vals   = []
            for z_i, _ in enumerate(zs_edges):

                if z_i==0:
                    lower = np.min(zs)
                else:
                    lower = (zs_edges[z_i]+zs_edges[z_i-1])/2

                if z_i==len(zs_edges)-1:
                    upper = np.max(zs)
                else:
                    upper = (zs_edges[z_i]+zs_edges[z_i+1])/2

                zs_bins.append([lower,upper])
                z_labels.append(f'{zs.name}={zs_edges[z_i]}{z_unit}')
                z_vals.append(zs_edges[z_i])


    for X_bin in calculate_bins(xs,bin_step):

        mask_X = (xs>=X_bin) & (xs<(X_bin+bin_step))

        if zs is not None:

            for z_i, (z_bin, z_val) in  enumerate(zip(zs_bins, z_vals)):

                mask = mask_X.copy()

                mask &= (zs>=z_bin[0]) & (zs<z_bin[1])

                if np.sum(mask)<2: # So a standard dev. can be calculated
                    continue

                x = average_of_averages(xs, xs_unc, xs_counts, mask)
                y = average_of_averages(ys, ys_unc, ys_counts, mask)

                if (isinstance(x, float) and np.isnan(x)) or (isinstance(y, float) and np.isnan(y)):
                    continue

                ax.errorbar(x.n, y.n, xerr=x.s, yerr=y.s, fmt='.', ms=0, ecolor=error_colour, capsize=0.8, capthick=0.4, lw=0.4, zorder=1)

                scatter = ax.scatter(x.n, y.n, c=z_val, cmap='cool', marker=scatter_markers[z_i], vmin=z_min, vmax=z_max, s=scat_size*(1+z_i/2))

        else:
            colour = data_colour if is_color_like(data_colour) else blue

            ax.scatter(x.n, y.n, c=colour, s=scat_size)

    if scatter:
        for z_i, (z_val, z_label) in enumerate(zip(z_vals, z_labels)):

            ax.scatter(x.n, y.n, c=z_val, cmap='cool', marker=scatter_markers[z_i], vmin=z_min, vmax=z_max, s=scat_size*(1+z_i/2), label=z_label)


    return fig, ax


def plot_scatter_with_dict(xs, ys, **kwargs):

    xs_unc       = kwargs.get('xs_unc',None)
    ys_unc       = kwargs.get('ys_unc',None)

    error_colour = kwargs.get('error_colour',black)
    scat_size    = kwargs.get('scatter_size',40)
    sc_ups       = kwargs.get('sc_ups',None)
    sc_dws       = kwargs.get('sc_dws',None)

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)

    if sc_ups is None:
        raise Exception('Missing upstream spacecraft.')
    elif sc_dws is None:
        raise Exception('Missing downstream spacecraft.')

    if fig is None or ax is None:
        fig, ax = plt.subplots()


    if xs_unc is not None or ys_unc is not None:
        ax.errorbar(xs, ys, xerr=xs_unc, yerr=ys_unc, fmt='.', ms=0, ecolor=error_colour, capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

    monitors = ('ACE', 'WIND', 'DSC', 'Cluster', 'THEMIS', 'OMNI')
    marker_dict = {'WIND': 'x', 'ACE': '+', 'DSC': '^', 'Cluster': 'o', 'THEMIS': 's'}

    for upstream, downstream in it.permutations(monitors, 2):
        sc_mask = np.all(len(xs))
        if upstream=='Cluster':
            sc_mask &= np.isin(sc_ups, ('C1','C2','C3','C4'))
        elif upstream=='THEMIS':
            sc_mask &= np.isin(sc_ups, ('THA','THB','THC','THD','THE'))
        else:
            sc_mask &= (sc_ups==upstream)

        if downstream=='Cluster':
            sc_mask &= np.isin(sc_dws, ('C1','C2','C3','C4'))
            sc_c=colour_dict['C1']
        elif downstream=='THEMIS':
            sc_mask &= np.isin(sc_dws, ('THA','THB','THC','THD','THE'))
            sc_c=colour_dict['THA']
        else:
            sc_mask &= (sc_dws==downstream)
            sc_c=colour_dict[downstream]

        count = np.sum(sc_mask)
        if count==0:
            continue
        ax.scatter(xs[sc_mask], ys[sc_mask], c=sc_c, s=scat_size, marker=marker_dict[upstream], label=f'{upstream} | {downstream}: {count}')



    ax.axhline(0,c='grey',ls=':')
    ax.axvline(0,c='grey',ls=':')


    return fig, ax
