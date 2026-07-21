# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:55:08 2025

@author: richarj2
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from sc_delay_time import merge_with_lag
from .plotting_utils import minimum_counts, def_param_names, get_variable_range, shift_angular_data, mask_df

from ...plotting.utils import save_figure, calculate_bins, get_grid_shape
from ...plotting.formatting import create_label, shifted_angle_ticks
from ...plotting.comparing.parameter import compare_dataframes
from ...plotting.config import black, blue, grey, pink, green
from ...plotting.distributions import plot_freq_hist

from ...processing.reading import import_processed_data
from ...processing.mag.indices import import_processed_index


def plot_driver_multi_responses(ind_var, *dep_vars, lags=None, show_omni=True, spacecraft=None, resolution='5min', region='sw', omni_colour=black, contemp_colour=blue, sc_colour=pink, bounds=None, restrict=True, shift_centre=True, bottom_axis='scatter', **kwargs):
    """
    Look at OMNI and in-situ data in driver-response
    The same ind_var is used for the omni and sc data, then a variety of dep_vars are shown in separate columns
    """

    ###----------IMPORTS----------###
    df_omni = None
    if show_omni:
        df_omni = import_processed_data('omni', resolution=resolution)

    df_sc = None
    if spacecraft is not None:
        df_sc = import_processed_data(region, dtype='plasma', resolution=resolution, file_name=f'{region}_times_{spacecraft}')

    kwargs['min_count'] = kwargs.get('min_count',minimum_counts[resolution])
    kwargs['display']   = kwargs.get('display','rolling')

    if 'data1_name' in kwargs:
        kwargs['data1_name'] = create_label(kwargs['data1_name'])

    if ind_var=='B_clock':
        shift_centre = True and shift_centre
    else:
        shift_centre = False

    ###----------PLOT GRIDS----------###
    n_rows, n_cols = 2, len(dep_vars)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows), dpi=200, height_ratios=[3,2], sharex='col')

    for i, dep_var in enumerate(dep_vars):

        df_pc = import_processed_index(dep_var, resolution=resolution, return_series=False)

        ax0 = axs[0][i]
        ax1 = axs[1][i]

        if df_sc is None:
            omni_j = 0
            enumerator = ((df_omni,omni_colour),)
        else:
            omni_j = 0
            overlap = df_omni.index.intersection(df_sc.index)
            enumerator = zip((df_omni,df_omni.loc[overlap],df_sc),(omni_colour,contemp_colour,sc_colour))

        for j, (df, colour) in enumerate(enumerator):

            if len(df)==0:
                print('df is empty')

            ind_err, ind_count = def_param_names(df, ind_var)
            dep_err, dep_count = def_param_names(df, dep_var)

            bin_width, limits, invert = get_variable_range(ind_var, 'sw', dep_var=dep_var, restrict=restrict, bounds=bounds, shift_centre=shift_centre)

            # Masks and slicing
            df_ind = mask_df(df, ind_var, limits)
            df_dep = mask_df(df_pc, dep_var)

            if shift_centre:
                shift_angular_data(df_ind, ind_var)

            df_ind, df_dep = merge_with_lag(df_ind, df_dep, lags[i], resolution)

            # Config
            kwargs['window_width'] = bin_width
            kwargs['window_step']  = bin_width/10
            kwargs['data_colour']  = colour
            kwargs['error_colour'] = colour

            if 'data_name_map' in kwargs:
                kwargs['data2_name'] = create_label(kwargs['data_name_map'].get(dep_var,dep_var))

            # Rolling window
            _ = compare_dataframes(df_ind, df_dep, ind_var, dep_var, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_err, col2_counts=dep_count, fig=fig, ax=ax0, return_objs=True, **kwargs)

            if bottom_axis=='heat' and j==0:
                bins_x = calculate_bins(df_ind[ind_var], bin_width)
                bins_y = calculate_bins(df_dep[dep_var], bin_width)

                ax1.hist2d(df_ind[ind_var], df_dep[dep_var], bins=[bins_x, bins_y], cmap='hot', norm=mpl.colors.LogNorm())
                ax1.set_facecolor('k')

                ax1.axline((limits[0],limits[0]), slope=1, c='w', ls=':')

            elif bottom_axis=='hist':
                hist_type = 'bar' if j==omni_j else 'step'
                ax1.hist(df_ind[ind_var], bins=calculate_bins(df_ind[ind_var],bin_width), color=colour, histtype=hist_type)

                ax1.axhline(kwargs['min_count'], c='k', ls='-')
                ax1.axhline(kwargs['min_count'], c='w', ls=':')
                ax1.set_yscale('log')

            elif bottom_axis=='scatter':
                # Scatter
                kwargs_copy = kwargs.copy()
                kwargs_copy['display'] = 'scatter'
                ind_err = None
                dep_err = None

                _ = compare_dataframes(df_ind, df_dep, ind_var, dep_var, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_err, col2_counts=dep_count, fig=fig, ax=ax1, return_objs=True, **kwargs_copy)

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

def plot_pulkkinen_grid(*params, ind_src='sw', dep_src='msh', resolution='5min', bounds=None, restrict=True, shift_centre=True, compare_colour=green, **kwargs):
    """
    Pulkkinen comparisons

    So plots OMNI vs sc (sw or msh) for range of parameters
    """

    ###----------IMPORTS----------###

    if ind_src=='omni':
        df1 = import_processed_data('omni', resolution=resolution)
    elif ind_src in ('sw','msh'):
        df1 = import_processed_data(ind_src, dtype='plasma', resolution=resolution, file_name=f'{ind_src}_times_combined')
    else:
        raise ValueError(f'"{ind_src}" is not a valid source.')

    if dep_src=='omni':
        df2 = import_processed_data('omni', resolution=resolution)
    elif dep_src in ('sw','msh'):
        df2 = import_processed_data(dep_src, dtype='plasma', resolution=resolution, file_name=f'{dep_src}_times_combined')
    else:
        raise ValueError(f'"{dep_src}" is not a valid source.')

    kwargs['data_colour']  = compare_colour
    kwargs['error_colour'] = compare_colour

    enumerator = [p if isinstance(p,tuple) else (p,p) for p in params]

    n_rows, n_cols = get_grid_shape(len(enumerator))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows), dpi=400)

    ###----------PLOT GRIDS----------###

    for i, (independent, dependent), in enumerate(enumerator):

        col, row = i % n_cols, i // n_cols

        ax = axs[row][col]

        ind_err, ind_count = def_param_names(df1, independent)
        dep_err, dep_count = def_param_names(df2, dependent)

        bin_width, limits, invert = get_variable_range(independent, ind_src, dep_var=dependent, restrict=restrict, bounds=bounds, shift_centre=shift_centre)

        # Shift angular data to centre lies at +-180 rather than 0
        if independent=='B_clock' and shift_centre:
            shift_angular_data(df1, independent)
            shift_angular_data(df2, dependent)

        df_ind = mask_df(df1, independent, limits)
        df_dep = mask_df(df2, dependent)

        df_ind, df_dep = merge_with_lag(df_ind, df_dep, 0, resolution)

        # Config
        kwargs['window_width'] = bin_width
        kwargs['data1_name'] = create_label(f'{independent}_{ind_src}')
        kwargs['data2_name'] = create_label(f'{dependent}_{dep_src}')

        if kwargs['display']=='scatter':
            ind_err = None
            dep_err = None
            if f'sc_{dep_src}' in df_dep:
                kwargs['df3'] = df_dep
                kwargs['col3'] = f'sc_{dep_src}'
                kwargs['data_colour'] = 'spacecraft'

        _ = compare_dataframes(df_ind, df_dep, independent, dependent, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_err, col2_counts=dep_count, fig=fig, ax=ax, return_objs=True, **kwargs)

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
    save_figure(fig, file_name=f'{ind_src}_vs_{dep_src}_sc_{kwargs["display"]}', sub_directory='Pulkkinen')
    plt.show()
    plt.close()

# %% Delay Histograms

def plot_delay_hists(sc, region, data_pop='plasma', sample_interval='5min'):

    df = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_{sc}')

    plot_freq_hist(df['prop_time_s'], bin_width=60, data_name=f'Lag ({sc.upper()} to BS) [s]', brief_title=region.upper(), sub_directory='prop_hists', file_name=f'{region}_{sc}_{sample_interval}')

