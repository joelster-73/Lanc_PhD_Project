# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:38:41 2025

@author: richarj2
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .config import black, white
from .formatting import add_legend, add_figure_title, create_label, dark_mode_fig, data_string
from .utils import save_figure, calculate_bins

from ..analysing.fitting import fit_function



def plot_freq_hist(series, **kwargs):


    data_name   = kwargs.get('data_name',None)
    fit_type    = kwargs.get('fit_type','mean')
    want_legend = kwargs.get('want_legend',True)
    brief_title = kwargs.get('brief_title','')

    lc          = kwargs.get('lc','r')
    ls          = kwargs.get('ls','-')

    colour      = kwargs.get('colour','k')
    perc_low    = kwargs.get('perc_low',0)
    perc_high   = kwargs.get('perc_high',100)
    bin_width   = kwargs.get('bin_width',None)

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',False)

    data_str = data_string(series.name)
    unit = series.attrs.get('units', {}).get(series.name, None)

    series = series.dropna().to_numpy()

    if unit == 'rad':
        series = np.degrees(series)
        unit = '°'

    data_label = create_label(data_str, unit=unit, data_name=data_name)

    ###-------------------PLOT HISTOGRAM-------------------###
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    n_bins = calculate_bins(series,bin_width)
    counts, bins, _ = ax.hist(series, bins=n_bins, alpha=1.0, color=colour, edgecolor='grey', linewidth=1)
    if fit_type in ('mean','median'):
        if fit_type=='mean':
            metric = np.mean(series)
            try:
                label = f'$\\mu={metric:L}$'
            except:
                label = f'$\\mu=${metric:.3g}'
        elif fit_type=='median':
            metric = np.median(series)
            try:
                label = f'$\\nu={metric:L}$'
            except:
                label = f'$\\nu=${metric:.3g}'
        ax.axvline(metric, lw=1, ls=ls, c=lc, label=label)
    else:
        mids = 0.5 * (bins[1:] + bins[:-1])
        bin_width = bins[1] - bins[0]
        xmin = mids[0] - 0.5 * bin_width
        xmax = mids[-1] + 0.5 * bin_width
        x_plot = np.linspace(xmin, xmax, 500)

        fit_dict = plot_fit(ax, mids, counts, fit_type, x_range=x_plot)

        if fit_type=='lognormal':
            peak = fit_dict['peaks'][0][0] # x position
            try:
                position = peak.n
                label = f'${peak:.1uL}$'
            except:
                position = peak
                label = f'{peak:.3g}'
            ax.text(x=position+0.75,y=0.9*ax.get_ylim()[1],s=label)

    perc_range = (np.percentile(series, perc_low), np.percentile(series, perc_high))
    if perc_low>0:
        ax.text(0.025, 0.075, f'$\\longleftarrow$\n{np.min(series):.1f}', transform=ax.transAxes)
    if perc_high<100:
        ax.text(0.865, 0.075, f'$\\longrightarrow$\n{np.max(series):.1f}', transform=ax.transAxes)

    ###-------------------SET LABELS AND TITLE-------------------###
    ax.set_xlim(perc_range)
    if fit_type=='bimodal_offset':
        offset = fit_dict['params']['c']
        ax_lim = ax.get_ylim()[1]
        ax.set_ylim(0.75*offset.n, ax_lim+0.5*offset.n)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

    ax.set_xlabel(data_label, c=black)
    ax.set_ylabel('Counts', c=black)
    if brief_title=='':
        brief_title = f'{len(series):,} mins'

    loc = 'split' if fit_type in ('bimodal','bimodal_offset') else 'upper right'
    add_legend(fig, ax, loc=loc, edge_col=white, frame_on=False, legend_on=want_legend)
    add_figure_title(fig, black, brief_title, ax=ax)
    dark_mode_fig(fig, black, white)

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
        brief_title = f'$m = {m:.1uL}$'
    else:
        brief_title = f'{len(series1):,} mins | {len(series2):,} mins'

    add_figure_title(fig, black, brief_title, ax=ax)
    dark_mode_fig(fig,black,white)

    if return_objs:
        return fig, ax

    plt.tight_layout();
    save_figure(fig)
    plt.show()
    plt.close()

def plot_fit(ax,xs,ys,fit_type,x_range=None,non_zero=True,plot_peaks=False,**kwargs):

    if fit_type is None:
        print('No fit type entered.')
        return {}

    inc_errs = kwargs.get('inc_errs',True)
    colour = kwargs.get('lc','r')
    ls     = kwargs.get('ls','-')

    if non_zero:
        xs = xs[ys > 0]
        ys = ys[ys > 0]

    fit_dict = fit_function(xs, ys, fit_type)
    func = fit_dict['func']
    popt = (v.n for v in fit_dict['params'].values())
    param_dict = fit_dict['params']

    if x_range is None:
        x_range = xs

    ax.plot(x_range, func(x_range, *popt), c=colour, ls=ls, linewidth=2)

    if fit_type in ('bimodal','bimodal_offset'):
        label1, label2 = bimodal_label(param_dict,inc_errs)
        ax.plot([],[],' ',label=label1)
        ax.plot([],[],' ',label=label2)
    else:
        if fit_type=='gaussian':
            label = gaussian_label(param_dict,inc_errs)
        elif fit_type=='lognormal':
            label = lognormal_label(param_dict,inc_errs)
        elif fit_type=='straight':
            label = straight_label(param_dict,inc_errs)
        else:
            label = None

        ax.plot([],[],' ',label=label)

    return fit_dict

# %% Labels

def straight_label(param_dict, detailed=True):

    m = param_dict['m']
    c = param_dict['c']

    if detailed:
        try:
            return f'$y=({m:.1uL})\\cdot x+{c:.1uL}$'
        except:
            return f'$y=$({m:.3g})$\\cdot x+${c:.3g}'

    return f'$y=$({m:.3g})$\\cdot x+${c:.3g}'


def gaussian_label(param_dict, detailed=True):

    mu  = param_dict['mu']
    std = param_dict['sigma']

    if detailed:
        try:
            return f'$\\mu$: ${mu:.1uL}$\n$\\sigma$: ${std:.1uL}$'
        except:
            return f'$\\mu$: {mu:.3g}\n$\\sigma$: {std:.3g}'

    return f'$\\mu$: {mu:.3g}\n$\\sigma$: {std:.3g}'

def bimodal_label(param_dict, detailed=True):

    mu1  = param_dict['mu1']
    std1 = param_dict['sigma1']
    mu2  = param_dict['mu2']
    std2 = param_dict['sigma2']

    if detailed:
        try:
            label1=f'$\\mu_1$: ${mu1:.1uL}$\n$\\sigma_1$: ${std1:.1uL}$'
            label2=f'$\\mu_2$: ${mu2:.1uL}$\n$\\sigma_2$: ${std2:.1uL}$'
        except:
            label1=f'$\\mu_1$: {mu1:.3g}\n$\\sigma_1$: {std1:.3g}'
            label2=f'$\\mu_2$: {mu2:.3g}\n$\\sigma_2$: {std2:.3g}'
    else:
        label1=f'$\\mu_1$: {mu1:.3g}\n$\\sigma_1$: {std1:.3g}'
        label2=f'$\\mu_2$: {mu2:.3g}\n$\\sigma_2$: {std2:.3g}'

    return label1, label2

def lognormal_label(param_dict, detailed=True):

    mu  = param_dict['mu']
    std = param_dict['sigma']

    if detailed:
        try:
            return f'$\\mu$: ${mu:.1uL}$\n$\\sigma$: ${std:.1uL}$'
        except:
            return f'$\\mu$: {mu:.3g}\n$\\sigma$: {std:.3g}'

    return f'$\\mu$: {mu:.3g}\n$\\sigma$: {std:.3g}'
