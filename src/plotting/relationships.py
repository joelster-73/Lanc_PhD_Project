# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 10:06:47 2025

@author: richarj2
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from uncertainties import UFloat
from .utils import save_figure
from .formatting import add_legend, create_label
from .distributions import plot_freq_hist
from .comparing.parameter import compare_series
from ..analysing.fitting import fit_series

def plot_fit_params_against_z(df_ind, ind_var, dep_vars, df_dep=None, **kwargs):

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',True)

    if not fig or not ax:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
    ax2 = ax.twinx()

    markers = ('v','x','s')
    colours = ('b','r','o')
    axes    = (ax,ax2)

    for i, (dep_val, dep_col) in enumerate(dep_vars.items()):

        param_dict = fit_series(df_ind, ind_var, dep_col, df2=df_dep, **kwargs)

        for j, (k, v) in enumerate(param_dict['params'].items()):

            print(k,v)
            marker = markers[j]
            colour = colours[j]
            axis   = axes[j]

            if isinstance(v,UFloat) and v.s<np.abs(v.n):
                axis.errorbar(dep_val, v.n, yerr=v.s, capsize=2, marker=marker, color=colour)

            else:
                axis.errorbar(dep_val, v.n, yerr=v.s, marker=marker, color=colour)


    for j, (k, v) in enumerate(param_dict['params'].items()):

        marker = markers[j]
        colour = colours[j]
        axis   = axes[j]

        label = create_label(k, unit=param_dict.get('units',{}).get(k,''))

        ax.scatter([], [], marker=marker, color=colour, label=k) # on same axis for legend
        axis.set_ylabel(label)


    add_legend(fig, ax, loc='upper center', cols=len(param_dict['params']))
    plt.tight_layout()

    if return_objs:
        return fig, (ax, ax2)

    save_figure(fig)
    plt.show()
    plt.close()


def plot_with_side_figs(series_1, series_2, bottom_panel=None, right_panel=None, **kwargs):

    panel_types = ('hist','rolling')

    if bottom_panel in panel_types and right_panel in panel_types:
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(4, 4)
        ax = fig.add_subplot(gs[0:3, 0:3])
        histx_ax = fig.add_subplot(gs[3, 0:3], sharex=ax)
        histy_ax = fig.add_subplot(gs[0:3, 3], sharey=ax)
        axs = (ax, histx_ax, histy_ax)
    elif bottom_panel in panel_types:
        fig = plt.figure(figsize=(8, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax = fig.add_subplot(gs[0, 0])
        histx_ax = fig.add_subplot(gs[1, 0], sharex=ax)
        axs = (ax, histx_ax)
    elif right_panel in panel_types:
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax = fig.add_subplot(gs[0, 0])
        histy_ax = fig.add_subplot(gs[0, 1], sharey=ax)
        axs = (ax, histy_ax)
    else:
        fig, ax = plt.subplots()
        axs = (ax,)

    kwargs_main = kwargs.copy()
    kwargs_main['fit_type'] = kwargs.get('main_fit','straight')
    kwargs_main['as_text'] = True
    kwargs_main['data_colour'] = kwargs['colouring']

    _, axis = compare_series(series_1, series_2, fig=fig, ax=ax, return_objs=True, **kwargs_main)
    #ylim = np.max(np.abs(ax.get_ylim()))
    #ax.set_ylim(-ylim,ylim)

    #####----------RIGHT PANEL----------#####


    kwargs_right = kwargs.copy()
    kwargs_right['fig'] = fig
    kwargs_right['ax'] = histy_ax
    kwargs_right['return_objs'] = True
    kwargs_right['as_text'] = True
    kwargs_right['brief_title'] = ''

    if right_panel == 'hist':
        kwargs_right['bin_width'] = 5
        kwargs_right['orientation'] = 'horizontal'
        kwargs_right['fit_type'] = kwargs.get('right_fit','bimodal')
        kwargs_right['fit_err'] = 'count'

        _, histy_ax = plot_freq_hist(series_2, **kwargs_right)
        histy_ax.set_ylabel(None)

    #####----------BOTTOM PANEL----------#####

    kwargs_bottom = kwargs.copy()
    kwargs_bottom['brief_title'] = ''
    kwargs_bottom['fig'] = fig
    kwargs_bottom['ax'] = histx_ax
    kwargs_bottom['return_objs'] = True

    if bottom_panel == 'hist':
        kwargs_bottom['bin_width'] = 5
        kwargs_bottom['orientation'] = 'vertical'
        kwargs_bottom['fit_type'] = kwargs.get('bottom_fit','bimodal')
        kwargs_bottom['simple_bounds'] = True
        kwargs_bottom['fit_err'] = 'count'
        kwargs_bottom['as_text'] = True

        _, histx_ax = plot_freq_hist(series_1, **kwargs_bottom)
        histx_ax.set_xlabel(None)

    elif bottom_panel == 'rolling':
        kwargs_bottom['display'] = 'rolling'
        kwargs_bottom['window_width'] = 10
        kwargs_bottom['window_step'] = 0.5
        kwargs_bottom['error_colour'] = 'red'

        _ = compare_series(series_1, series_2, **kwargs_bottom)
        histx_ax.set_xlabel(None)
        histx_ax.set_ylabel(None)

    if series_1.attrs.get('units',{}).get(series_1.name,'')!='mins':
        ax.invert_xaxis()

    return_objs = kwargs.get('return_objs',False)

    if return_objs:
        return fig, axs

    plt.tight_layout()
    save_figure(fig)

    plt.show()
    plt.close()

