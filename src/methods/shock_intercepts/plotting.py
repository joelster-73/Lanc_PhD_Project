# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:20:40 2025

@author: richarj2
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .statistics import get_diffs_with_OMNI, shock_compressions, get_shock_propagations

from ...plotting.config import black
from ...plotting.relationships import plot_with_side_figs
from ...plotting.distributions import plot_freq_hist
from ...plotting.comparing.parameter import compare_series
from ...plotting.utils import save_figure, change_series_name

def plot_propagations_both(shocks, **kwargs):

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20,7))

    _ = plot_shock_propagations(shocks, selection='all', fig=fig, ax=ax_left, return_objs=True, **kwargs)
    _ = plot_shock_propagations(shocks, selection='omni', fig=fig, ax=ax_right, return_objs=True, **kwargs)

    ###---------------LABELLING AND FINISHING TOUCHES---------------###
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

def plot_shock_propagations(shocks, **kwargs):

    selection      = kwargs.get('selection','all')
    x_axis         = kwargs.get('x_axis','delta_x')
    colouring      = kwargs.get('colouring','spacecraft')
    max_dist       = kwargs.get('max_dist',300)
    normals        = kwargs.get('shock_normals',None)

    if normals is None and x_axis in ('delta_n','delta_t'):
        print('Haven\'t provided normals; using x-distance instead.')
        x_axis = 'delta_x'

    fig = kwargs.get('fig',None)
    ax  = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',True)

    if fig is None or ax is None:
        fig, ax = plt.subplots()
        kwargs['fig'] = fig
        kwargs['ax'] = ax

    df = get_shock_propagations(shocks,normals)

    series_x = df.loc[:,x_axis]
    series_x_unc = df.loc[:,x_axis+'_unc']

    series_y = df.loc[:,'time']
    series_y_unc = df.loc[:,'time_unc']


    if 'earth' in selection:
        max_dist=100

    mask = series_x<max_dist
    mask &= series_x>-100 # Second condition just to restrict massive outlier

    if 'omni' in selection:
        new_y_name = 't_SC - t_OMNI'

        if x_axis=='delta_x':
            new_x_name = 'X_SC - X_BSN'
        elif x_axis=='delta_r':
            new_x_name = '|R_SC - R_BSN |'
        elif x_axis=='delta_n':
            new_x_name = 'n•(R_SC - R_BSN )'
        elif x_axis=='delta_t':
            new_x_name = 'n•(R_SC - R_BSN ) / v'

        mask &= df.loc[:,'interceptor'] == 'OMNI'
        title_info = 'OMNI and Detector'

    else:
        new_y_name = 't_SC1 - t_SC2'
        if x_axis=='delta_x':
            new_x_name = 'X_SC1 - X_SC2'
        elif x_axis=='delta_r':
            new_x_name = '|R_SC1 - R_SC2 |'
        elif x_axis=='delta_n':
            new_x_name = 'n•(R_SC1 - R_SC2 )'
        elif x_axis=='delta_t':
            new_x_name = 'n•(R_SC1 - R_SC2 ) / v'

        mask &= df.loc[:,'interceptor'] != 'OMNI'
        title_info = 'Two Shock Detectors'

    kwargs['brief_title'] = title_info

    change_series_name(series_x,new_x_name)
    change_series_name(series_y,new_y_name)

    xs = series_x[mask]
    ys = series_y[mask]/60
    ys.attrs['units'][ys.name] = 'mins'

    xs_unc = series_x_unc[mask]
    ys_unc = series_y_unc[mask]/60
    ys_unc.attrs['units'][ys_unc.name] = 'mins'

    if x_axis=='delta_t':
        xs /= 60
        xs_unc /= 60
        xs.attrs['units'][xs.name] = 'mins'
        xs_unc.attrs['units'][xs_unc.name] = 'mins'

    if colouring=='spacecraft':
        kwargs['sc_ups'] = df.loc[mask,'detector']
        kwargs['sc_dws'] = df.loc[mask,'interceptor']
        kwargs['error_colour'] = black
        kwargs['display'] = 'scatter_dict'

    kwargs['fit_type'] = 'straight'
    kwargs['as_text'] = True

    _ = compare_series(xs, ys, xs_unc=xs_unc, ys_unc=ys_unc, **kwargs)

    ax.invert_xaxis()
    if return_objs:
        return fig, ax

    ###---------------LABELLING AND FINISHING TOUCHES---------------###
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

def plot_time_differences(shocks, bottom_panel='hist', right_panel='hist', **kwargs):

    coeff_lim      = kwargs.get('coeff_lim',None)
    selection      = kwargs.get('selection','all')
    x_axis         = kwargs.get('x_axis','delta_x')
    colouring      = kwargs.get('colouring','spacecraft')
    max_dist       = kwargs.get('max_dist',300)
    normals        = kwargs.get('shock_normals',None)

    if normals is None and x_axis in ('delta_n','delta_t'):
        print('Haven\'t provided normals; using x-distance instead.')
        x_axis = 'delta_x'

    df = get_diffs_with_OMNI(shocks,normals)

    ###---------------MASKING---------------###
    if selection=='earth':
        max_dist=100
        bottom_panel = 'rolling' if bottom_panel=='hist' else bottom_panel
        kwargs['right_fit'] = 'gaussian'

    mask = np.ones(len(df), dtype=bool)

    mask &= (df.loc[:,'delta_x']<max_dist) & (df.loc[:,'delta_x']>-100) # Second condition just to restrict massive outlier
    if x_axis == 'delta_t':
        mask &= (df.loc[:,x_axis]<9000) # to restrict massive outlier
    if coeff_lim is not None and 'corr_coeff' in df:
        mask &= (df.loc[:,'coeff'] >= coeff_lim)

    mask &= (~np.isnan(df.loc[:,x_axis])) & (~np.isnan(df.loc[:,'time']))

    series_x = df.loc[mask,x_axis]
    series_x_unc = df.loc[mask,x_axis+'_unc']

    series_y = df.loc[mask,'time']/60
    series_y_unc = df.loc[mask,'time_unc']/60

    series_y.attrs['units'][series_y.name] = 'mins'
    series_y_unc.attrs['units'][series_y_unc.name] = 'mins'

    ###---------------LABELLING---------------###
    if x_axis=='delta_t':
        series_x /= 60
        series_x_unc /= 60
        series_x.attrs['units'][series_x.name] = 'mins'
        series_x_unc.attrs['units'][series_x_unc.name] = 'mins'

    if colouring=='spacecraft':
        kwargs['zs'] = df.loc[mask,'detector']
        kwargs['error_colour'] = black
        kwargs['display'] = 'scatter_dict'

        if x_axis=='delta_x':
            new_x_name = 'X_SC - X_BSN'
        elif x_axis=='delta_r':
            new_x_name = '|R_SC - R_BSN |'
        elif x_axis=='delta_n':
            new_x_name = 'n•(R_SC - R_BSN )'
        elif x_axis=='delta_t':
            new_x_name = 'n•(R_SC - R_BSN ) / v'

        change_series_name(series_x,new_x_name)
        change_series_name(series_y,'t_SC - t_OMNI')

    for series in (series_x,series_y,series_x_unc,series_y_unc):
        return_none = False
        if len(series.dropna())==0:
            return_none = True
            print(f'No "{series.name}" data.')
        if return_none:
            return

    plot_with_side_figs(series_x, series_y, bottom_panel, right_panel, xs_unc=series_x_unc, ys_unc=series_y_unc, **kwargs)


def plot_compressions_both(shocks, plot_type='hist', *kwargs):

    fig, (ax_left, ax_right) = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(16,6))

    _ = plot_compression(shocks, selection='all', fig=fig, ax=ax_left, return_objs=True, plot_type=plot_type)
    _ = plot_compression(shocks, selection='omni', fig=fig, ax=ax_right, return_objs=True, plot_type=plot_type)

    ax_right.set_ylabel(None)

    ###---------------LABELLING AND FINISHING TOUCHES---------------###
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

def plot_compression(shocks, selection='all', plot_type='hist', change='change_rel', **kwargs):

    fig = kwargs.get('fig',None)
    ax  = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',True)

    if fig is None or ax is None:
        fig, ax = plt.subplots()
        kwargs['fig'] = fig
        kwargs['ax'] = ax

    compressions = shock_compressions(shocks)

    title_info = 'All spacecraft'
    if selection=='omni':
        title_info = 'Spacecraft in OMNI'

    sc_mask = np.all(len(compressions))

    if selection=='omni':
        sc_mask &= (compressions['sc_dw']=='OMNI')
        x_name = 'r_B (sc)'
        y_name = 'r_B (OMNI)'
        count_name = 'r_B_OMNI / r_B_sc'
        count_label = 'Compression Change (OMNI / sc)'
        title_info = 'OMNI and Detector'
    else:
        sc_mask &= (compressions['sc_dw']!='OMNI')
        x_name = 'r_B (sc #1)'
        y_name = 'r_B (sc #2)'
        count_name = 'r_B_2 / r_B_1'
        count_label = 'Compression Change (sc2 / sc1)'
        title_info = 'Earliest and Latest Spacecraft'

    kwargs['brief_title'] = title_info

    clipping = 1.3 if selection=='omni' else 1.2
    if selection=='omni':
        cmap = plt.colormaps['Oranges']
    else:
        cmap = ListedColormap(plt.colormaps['PiYG'](np.linspace(0, 0.45, 128))[::-1])

    if plot_type=='hist':

        bin_width = 0.05 if change=='change_rel' else 0.1

        xs = compressions.loc[sc_mask,change].apply(lambda x: x.nominal_value)
        change_series_name(xs, count_name)

        _ = plot_freq_hist(xs, cmap=cmap, clipping=clipping, bin_width=bin_width, fit_type='gaussian', simple_bounds=True, fit_colour=black, data_name=count_label, fit_err='count', **kwargs)

    elif plot_type=='scatter':
        x_lim = (1,5)
        y_lim = (1,5)

        sc_mask &= (compressions['comp_up']>=x_lim[0]) & (compressions['comp_up']<=x_lim[1])
        sc_mask &= (compressions['comp_dw']>=y_lim[0]) & (compressions['comp_dw']<=y_lim[1])


        if 'time' in plot_type:
            xs     = compressions.loc[sc_mask,'time'].apply(lambda x: x.nominal_value/60)
            xs_unc = compressions.loc[sc_mask,'time'].apply(lambda x: x.std_dev/60)

            ys = compressions.loc[sc_mask,change].apply(lambda x: x.nominal_value)
            ys = compressions.loc[sc_mask,change].apply(lambda x: x.std_dev)

            change_series_name(ys, count_name)

        else:
            xs = compressions.loc[sc_mask,'comp_up'].apply(lambda x: x.nominal_value)
            xs_unc = compressions.loc[sc_mask,'comp_up'].apply(lambda x: x.std_dev)

            ys = compressions.loc[sc_mask,'comp_dw'].apply(lambda x: x.nominal_value)
            ys_unc = compressions.loc[sc_mask,'comp_dw'].apply(lambda x: x.std_dev)

            change_series_name(xs, x_name)
            change_series_name(ys, y_name)

        sc_ups = compressions.loc[sc_mask,'sc_up']
        sc_dws = compressions.loc[sc_mask,'sc_dw']

        _ = compare_series(xs, ys, xs_unc=xs_unc, ys_unc=ys_unc, display='scatter_dict', sc_ups=sc_ups, sc_dws=sc_dws, fit_type='straight', as_text=True, **kwargs)


    ###---------------LABELLING AND FINISHING TOUCHES---------------###
    if return_objs:
        return fig, ax

    save_figure(fig)
    plt.show()
    plt.close()

def plot_omni_compressions(shocks, plot_type='hist', change='change_rel', **kwargs):

    fig = kwargs.get('fig',None)
    ax  = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',False)

    if fig is None or ax is None:
        fig, ax = plt.subplots()
        kwargs['fig'] = fig
        kwargs['ax'] = ax

    compressions = shock_compressions(shocks)

    sign = '/' if change=='change_rel' else '-'

    x_name      = 'r_B (sc)'
    y_name      = 'r_B (OMNI)'
    count_name  = f'r_B_OMNI {sign} r_B_sc'
    count_label = f'Compression Difference (OMNI {sign} sc)'


    clipping = 1.3
    cmap = plt.colormaps['Oranges']
    kwargs_plot = kwargs.copy()
    kwargs_plot['return_objs'] = True

    if plot_type=='hist':

        bin_width = 0.05 if change=='change_rel' else 0.1

        xs = compressions.loc[:,change].apply(lambda x: x.nominal_value)
        change_series_name(xs, count_name)

        _ = plot_freq_hist(xs, cmap=cmap, clipping=clipping, bin_width=bin_width, fit_type='gaussian', simple_bounds=True, fit_colour=black, data_name=count_label, fit_err='count', **kwargs_plot)

    elif plot_type=='scatter':

        xs = compressions.loc[:,'comp_up'].apply(lambda x: x.nominal_value)
        xs_unc = compressions.loc[:,'comp_up'].apply(lambda x: x.std_dev)

        ys = compressions.loc[:,'comp_dw'].apply(lambda x: x.nominal_value)
        ys_unc = compressions.loc[:,'comp_dw'].apply(lambda x: x.std_dev)

        change_series_name(xs, x_name)
        change_series_name(ys, y_name)

        sc_ups = compressions.loc[:,'sc_up']
        sc_dws = compressions.loc[:,'sc_dw']

        _ = compare_series(xs, ys, xs_unc=xs_unc, ys_unc=ys_unc, display='scatter_dict', sc_ups=sc_ups, sc_dws=sc_dws, fit_type='straight', as_text=True, **kwargs_plot)


    ###---------------LABELLING AND FINISHING TOUCHES---------------###
    if return_objs:
        return fig, ax

    save_figure(fig)
    plt.show()
    plt.close()



