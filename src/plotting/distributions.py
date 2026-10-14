# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:38:41 2025

@author: richarj2
"""
import numpy as np
import pandas as pd

from uncertainties import UFloat

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib import ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .config import black, white
from .formatting import add_legend, add_figure_title, create_label, dark_mode_fig, data_string
from .utils import save_figure, calculate_bins

from .comparing.procedures import plot_fit

from ..analysing.fitting import fit_function
from ..analysing.calculations import calc_mean_error

def plot_freq_hist(series, **kwargs):


    data_name   = kwargs.get('data_name',None)
    fit_type    = kwargs.get('fit_type','mean')
    fit_err     = kwargs.get('fit_err',None)

    want_legend = kwargs.get('want_legend',True)
    brief_title = kwargs.get('brief_title','')
    add_count   = kwargs.get('add_count',False)
    orientation = kwargs.get('orientation','vertical')

    colour      = kwargs.get('colour',black)
    cmap        = kwargs.get('cmap',None)
    clipping    = kwargs.get('clipping',1)
    perc_low    = kwargs.get('perc_low',0)
    perc_high   = kwargs.get('perc_high',100)
    bin_width   = kwargs.get('bin_width',None)
    edge_colour = kwargs.get('edge_color','#444444')
    edge_width  = kwargs.get('edge_width',1)

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',False)

    data_str = data_string(series.name)
    unit     = series.attrs.get('units', {}).get(series.name, None)

    series = series.dropna()

    if unit == 'rad':
        series = np.degrees(series)
        unit = '°'
        series.attrs['units'][series.name] = unit

    series_0 = series.copy()
    series = series.to_numpy()

    data_label = create_label(data_str, unit=unit, data_name=data_name)

    ###-------------------PLOT HISTOGRAM-------------------###
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        kwargs['fig'] = fig
        kwargs['ax'] = ax

    n_bins = calculate_bins(series,bin_width)
    counts, bins, patches = ax.hist(series, bins=n_bins, alpha=1.0, color=colour, edgecolor=edge_colour, linewidth=edge_width, orientation=orientation)

    norm = mcolors.Normalize(vmin=min(counts), vmax=max(counts)*clipping)
    if cmap is not None:
        for count, patch in zip(counts, patches):
            colour = cmap(norm(count))
            patch.set_facecolor(colour)

    ###-------------------FITTING-------------------###
    if fit_type in ('mean','median'):
        plot_metric(series_0, metric=fit_type, **kwargs)

    else:
        mids = 0.5 * (bins[1:] + bins[:-1])
        bin_width = bins[1] - bins[0]
        xmin = mids[0] - 0.5 * bin_width
        xmax = mids[-1] + 0.5 * bin_width
        x_plot = np.linspace(xmin, xmax, 500)

        kwargs_fit = kwargs.copy()
        kwargs_fit['xs_unc'] = None
        if fit_err is None:
            kwargs_fit['ys_unc'] = None
        elif fit_err=='count':
            print('Using counts and uncertainties on fit')
            kwargs_fit['ys_unc'] = np.sqrt(counts+1) # +1 to avoid /0 errors

        fit_dict = plot_fit(mids, counts, x_range=x_plot, unit=unit, **kwargs_fit)

        if fit_type=='lognormal':
            yerrs = kwargs.get('ys_unc',None)
            peak = fit_dict['peaks'][0][0] # x position
            if yerrs is not None:
                try:
                    position = peak.n
                    label = f'${peak:L}$'
                except:
                    position = peak
                    label = f'$x\\simeq{peak:.3g}$ {unit}'
            else:
                position = peak.n if isinstance(peak, UFloat) else peak
                label = f'$x\\simeq{position:.3g}$ {unit}'
            ax.text(position+0.75, 0.9*ax.get_ylim()[1], s=label, family='monospace')

    perc_range = [None, None]
    if np.min(series)>=0: # If all positive data
        perc_range[0] = 0
    if perc_low>0:
        perc_range[0] = np.percentile(series, perc_low)
        ax.text(0.025, 0.075, f'$\\longleftarrow$\n{np.min(series):.1f}', transform=ax.transAxes, family='monospace')
    if perc_high<100:
        perc_range[1] = np.percentile(series, perc_high)
        ax.text(0.865, 0.075, f'$\\longrightarrow$\n{np.max(series):.1f}', transform=ax.transAxes, family='monospace')

    ###-------------------SET LABELS AND TITLE-------------------###
    if fit_type=='bimodal_offset':
        offset = fit_dict['params']['c']
        ax_lim = ax.get_ylim()[1]
        ax.set_ylim(0.75*offset.n, ax_lim+0.5*offset.n)

    if orientation=='vertical':
        ax.set_xlim(perc_range)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

        ax.set_xlabel(data_label, c=black)
        ax.set_ylabel('Counts', c=black)
    else:
        ax.set_ylim(perc_range)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))

        ax.set_ylabel(data_label, c=black)
        ax.set_xlabel('Counts', c=black)

    if brief_title=='amount':
        brief_title = f'{len(series):,}'
    elif brief_title != '' and add_count:
        brief_title += f', N={len(series):,}'

    loc = 'split' if fit_type in ('bimodal','bimodal_offset') else 'upper right'
    add_legend(fig, ax, loc=loc, edge_col=white, frame_on=False, legend_on=want_legend)
    add_figure_title(fig, brief_title, ax=ax)

    if return_objs:
        return fig, ax

    sub_dir   = kwargs.get('sub_directory',None)
    save_name = kwargs.get('file_name',None)

    plt.tight_layout();
    save_figure(fig, sub_directory=sub_dir, file_name=save_name)
    plt.show()
    plt.close()

def plot_counts(counts, **kwargs):
    """
    Plotting column chart of counts for discrete data
    """
    data_name   = kwargs.get('data_name',None)
    brief_title = kwargs.get('brief_title','')
    add_count   = kwargs.get('add_count',False)
    add_percs   = kwargs.get('add_percs',False)

    labels      = kwargs.get('labels',None)
    colour      = kwargs.get('colour',black)

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',False)

    data_label  = counts.name if data_name is None else data_name
    unit        = counts.attrs.get('units', {}).get(counts.name, 'Counts')

    ###-------------------PLOT HISTOGRAM-------------------###
    xs = counts.index
    if not pd.api.types.is_numeric_dtype(xs):
        xs = range(len(xs))

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12,6),dpi=400)
        kwargs['fig'] = fig
        kwargs['ax'] = ax

    bar_container = ax.bar(xs, counts.values, color=colour)
    if add_percs:
        total = sum(counts.values)
        ax.bar_label(bar_container, labels=[f'{v/total*100:.1f}%' for v in counts.values])
        total_unit = unit
        if unit=='Mins':
            total /= 525600
            total_unit = 'Years'
        ax.plot([], [], ' ', label=f'Total: {total:.3g} {total_unit}')
        add_legend(fig, ax, loc='upper right')

    if labels is not None:
        x_sorted = sorted(labels.keys())
        labels_sorted = [labels[k] for k in x_sorted]

    ###-------------------SET LABELS AND TITLE-------------------###
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))
    ax.set_xlabel(data_label)
    ax.set_ylabel(unit)

    ax.set_xticks(x_sorted)
    ax.set_xticklabels(labels_sorted)

    if brief_title=='amount':
        brief_title = f'{len(counts):,}'
    elif brief_title != '' and add_count:
        brief_title += f', N={len(counts):,}'

    if return_objs:
        return fig, ax

    plt.tight_layout();
    save_figure(fig)
    plt.show()
    plt.close()

def plot_q_q(series1, series2, **kwargs):

    series1_name   = kwargs.get('series1_name',None)
    series2_name   = kwargs.get('series2_name',None)
    series1_colour = kwargs.get('series1_colour',black) # for axis label only
    series2_colour = kwargs.get('series2_colour',black)

    perc_low    = kwargs.get('perc_low',0)
    perc_high   = kwargs.get('perc_high',100)
    brief_title = kwargs.get('brief_title','')

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',False)

    # Labels and units
    unit1 = series1.attrs.get('units', {}).get(series1.name, None)
    unit2 = series2.attrs.get('units', {}).get(series2.name, None)

    # Done in this order as attrs would be lost and need to convert to degrees from radians
    series_1 = series1.dropna().to_numpy()
    series_2 = series2.dropna().to_numpy()

    if unit1 == 'rad':
        series1 = np.degrees(series1)
        unit1 = '°'

    if unit2 == 'rad':
        series2 = np.degrees(series2)
        unit2 = '°'

    # Percentiles and deciles
    all_indices = np.arange(101)
    perc_mask = (all_indices >= perc_low) & (all_indices <= perc_high)

    percentiles_1 = np.percentile(series_1, all_indices)
    percentiles_2 = np.percentile(series_2, all_indices)

    decile_indices = np.arange(10, 100, 10)
    decile_mask = (decile_indices >= perc_low) & (decile_indices <= perc_high)

    deciles_1 = percentiles_1[decile_indices]
    deciles_2 = percentiles_2[decile_indices]

    min_1   = percentiles_1[0]
    min_2   = percentiles_2[0]
    max_1 = percentiles_1[-1]
    max_2 = percentiles_2[-1]

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    ax.axline(xy1=(0, 0), slope=1, lw=1.5, c=black, zorder=1)
    ax.scatter(percentiles_1[perc_mask], percentiles_2[perc_mask], c='darkgrey', edgecolors='grey', zorder=2, alpha=0.8)
    ax.scatter(deciles_1[decile_mask], deciles_2[decile_mask], c='r', edgecolors='k', zorder=3)

    if perc_low==0:
        ax.scatter(min_1, min_2, c='magenta', edgecolors='k', zorder=3)

    if perc_high==100:
        ax.scatter(max_1, max_2, c='magenta', edgecolors='k', zorder=3)

    if perc_low>0 or perc_high<100:

        # Inset plot (zoomed out, 0 to 100 percentiles)
        inset_ax = inset_axes(ax, width='30%', height='30%', loc='upper left')

        inset_ax.axline(xy1=(0, 0), slope=1, lw=1, c='k', zorder=1)
        inset_ax.scatter(percentiles_1, percentiles_2, c='darkgrey', s=20, edgecolors='grey', zorder=2, alpha=0.8)
        inset_ax.scatter(deciles_1, deciles_2, c='r', s=20, edgecolors='k', zorder=3, alpha=0.9)
        inset_ax.scatter((min_1,max_1), (min_2,max_2), c='magenta', s=20, edgecolors='k', zorder=3)

        inset_ax.set_xticks([np.floor(min_1), np.ceil(max_1)])
        inset_ax.set_yticks([np.floor(min_2), np.ceil(max_2)])
        inset_ax.yaxis.tick_right()
        inset_ax.tick_params(axis='x', which='both', labelsize=7)
        inset_ax.tick_params(axis='y', which='both', labelsize=7)

        #inset_ax.set_xticks([])  # Remove ticks for clarity
        #inset_ax.set_yticks([])


    ###-------------------LABELS AND TITLE-------------------###

    data1_str = data_string(series1.name)
    data2_str = data_string(series2.name)
    data1_label = create_label(data1_str, unit=unit1, data_name=series1_name)
    data2_label = create_label(data2_str, unit=unit2, data_name=series2_name)

    ax.set_xlabel(data1_label, c=series1_colour)
    ax.set_ylabel(data2_label, c=series2_colour)

    if brief_title=='slope':
        fit_dict = fit_function(percentiles_1, percentiles_2, fit_type='straight')
        m = fit_dict['m']
        brief_title = f'$m = {m:L}$'
    elif brief_title=='amount':
        brief_title = f'{len(series1):,} | {len(series2):,}'

    add_figure_title(fig, black, brief_title, ax=ax)
    dark_mode_fig(fig,black,white)

    if return_objs:
        return fig, ax

    plt.tight_layout();
    save_figure(fig)
    plt.show()
    plt.close()

def plot_metric(series, metric='mean', **kwargs):

    ax     = kwargs.get('ax',None)
    colour = kwargs.get('lc','r')
    ls     = kwargs.get('ls','-')
    unit   = kwargs.get('unit',None)

    if unit is None:
        unit = series.attrs.get('units',{}).get(series.name,'')

    if ax is not None:

        if metric=='mean':
            value = calc_mean_error(series)
            try:
                label = f'$\\mu={value:L}$ {unit}'
            except:
                label = f'$\\mu=${value:.3g} {unit}'
        elif metric=='median':
            value = np.median(series)
            try:
                label = f'$\\nu={value:L}$ {unit}'
            except:
                label = f'$\\nu=${value:.3g} {unit}'

        value = value.n if isinstance(value, UFloat) else value
        ax.axvline(value, lw=1, ls=ls, c=colour)
        ax.plot([], [], ' ', label=label)
