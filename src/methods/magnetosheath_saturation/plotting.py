# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:55:08 2025

@author: richarj2
"""

import numpy as np
import itertools as it

import matplotlib as mpl
import matplotlib.pyplot as plt

from .plotting_utils import minimum_counts, def_param_names, get_var_bin_width, get_variable_range, shift_angular_data, mask_df

from ...plotting.utils import save_figure, calculate_bins
from ...plotting.formatting import create_label, shifted_angle_ticks
from ...plotting.comparing.parameter import compare_dataframes
from ...plotting.config import black, blue, grey, pink, green


def plot_driver_multi_responses(df_omni, df_sc, df_pc, ind_var, *dep_vars, ind_src='sw', dep_src='pc', omni_colour=black, contemp_colour=blue, sc_colour=pink, bounds=None, restrict=True, shift_centre=True, bottom_axis='scatter', **kwargs):
    """
    Look at OMNI and in-situ data in driver-response
    The same ind_var is used for the omni and sc data, then a variety of dep_vars are shown in separate columns
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
    kwargs['print_text'] = True
    kwargs['show_error'] = True
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

    dep_cols = {0: {#'PCN': ['PCN','SMC_y_GSM'],
                    'PCN': ['PCN','SMC'], # temporary
                    'PCC': ['PCC','SMC'],
                    'AE':  ['AE', 'SME']},
                1: {#'PCN': [f'PCN_{lag}m',f'SMC_y_GSM_{lag}m'],
                    'PCN': [f'PCN_{lag}m',f'SMC_{lag}m'],
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

        # Setup

        row, col = i % n_rows, i // n_rows
        ax = axs[row][col]

        if df is df_omni:
            print('OMNI',dep_var)
            if contemp_omni:
                overlap = df.index.intersection(df_sc.index)
                df = df.loc[overlap]
                colour = contemp_colour
            else:
                colour = omni_colour
        else:
            print('SC',dep_var)
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

        # Kwargs

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

        objs = compare_dataframes(df_ind, df_dep, ind_var, dep_var, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_err, col2_counts=dep_count, fig=fig, ax=ax, return_objs=True, **kwargs)
        if len(objs)==3: # indicates cbar present
            cbar = objs[-1]
            if col!=n_cols-1:
                cbar.set_label(None)
            else:
                cbar.set_label(data_type.capitalize())

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

        if dep_var.startswith('PCN'): # Nicer formatting
            ax.set_ylim(-10)

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

    file_name = f'Comparing_{ind_var}_{dep}_fit_{kwargs["fit_type"]}_{lag}m'

    plt.tight_layout()
    save_figure(fig, file_name=file_name, sub_directory='Driver_Response')
    plt.show()
    plt.close()


    plt.close()