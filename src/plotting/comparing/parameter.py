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
from matplotlib.colors import is_color_like
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from collections import Counter

from ..config import black, white, blue
from ..formatting import add_legend, add_figure_title, create_label, dark_mode_fig, data_string
from ..utils import save_figure, calculate_bins, change_series_name
from ..distributions import plot_fit

from ...analysing.comparing import difference_columns

from ...processing.speasy.config import colour_dict, database_colour_dict


def compare_columns(df, col1, col2, **kwargs):

    series1 = df.loc[:,col1]
    series2 = df.loc[:,col2]

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


def is_marker_like(marker):
    try:
        MarkerStyle(marker)
        return True
    except ValueError:
        return False

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


def plot_scatter(xs, ys, **kwargs):

    xs_unc       = kwargs.get('xs_unc',None)
    ys_unc       = kwargs.get('ys_unc',None)

    data_colour  = kwargs.get('data_colour',blue)
    error_colour = kwargs.get('error_colour','r')
    scat_size    = kwargs.get('scatter_size',0.5)
    scat_marker  = kwargs.get('scat_marker','o')

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)

    if fig is None or ax is None:
        fig, ax = plt.subplots()


    if xs_unc is not None or ys_unc is not None:
        ax.errorbar(xs, ys, xerr=xs_unc, yerr=ys_unc, fmt='.', ms=0, ecolor=error_colour, capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

    have_plotted = False

    if not is_marker_like(scat_marker):
        scat_marker = 'o'

    if is_color_like(data_colour):
        have_plotted = True
        ax.scatter(xs, ys, c=data_colour, marker=scat_marker, s=scat_size)


    # Using a z-value to apply a continuous colour map
    elif data_colour in ('coeff','sun_earth','angle'):
        colour_values = kwargs.get('colour_values',None)

        if colour_values is not None:
            have_plotted = True

            if data_colour == 'coeff':
                zmin   = np.min(colour_values)
                zmax   = 1
                zlabel = r'cross-corr $\rho$'
            elif data_colour == 'sun_earth':
                zmin   = 0
                zmax   = 100
                zlabel = r'$\sqrt{Y^2+Z^2}$ [$\mathrm{R_E}$]'
            elif data_colour == 'angle':
                zmin   = 0
                zmax   = 135
                zlabel = r'angle [$^\circ$]'
                colour_values = np.degrees(colour_values)

            scatter = ax.scatter(xs, ys, c=colour_values, cmap='plasma_r', vmin=zmin, vmax=zmax, marker=scat_marker, s=scat_size)

            cbar = plt.colorbar(scatter)
            cbar.set_label(zlabel)

    # Using a dictionary to map the colours
    elif data_colour in ('spacecraft','detector','database'):
        colour_values = kwargs.get('colour_values',None)

        if colour_values is not None:
            have_plotted = True

            if data_colour == 'database':
                sc_colours = database_colour_dict
            else:
                sc_colours = colour_dict

            spacecraft_counts = Counter(colour_values)
            colours = pd.Series(colour_values).map(sc_colours).fillna(black).to_numpy()

            scatter = ax.scatter(xs, ys, c=colours, marker=scat_marker, s=scat_size)

            legend_elements = [Line2D([0], [0], marker='o', color=colour, label=f'{label}: {spacecraft_counts.get(label, 0)}', markersize=1, linestyle='None') for label, colour in sc_colours.items() if spacecraft_counts.get(label, 0) > 0]

            ax.legend(handles=legend_elements, fontsize=8, loc='upper left')


    if not have_plotted:
        ax.scatter(xs, ys, c=blue, marker=scat_marker, s=scat_size)


    ax.axhline(0,c='grey',ls=':')
    ax.axvline(0,c='grey',ls=':')


    return fig, ax

