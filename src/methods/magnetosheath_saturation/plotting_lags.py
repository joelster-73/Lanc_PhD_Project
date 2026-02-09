# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:55:08 2025

@author: richarj2
"""

import itertools as it

import matplotlib.pyplot as plt

from .plotting_utils import minimum_counts, def_param_names, get_variable_range, get_lagged_columns, mask_df

from ...plotting.utils import save_figure
from ...plotting.formatting import create_label, add_legend, format_string
from ...plotting.comparing.parameter import compare_dataframes
from ...plotting.relationships import plot_fit_params_against_z
from ...plotting.config import black, blue, pink



def plot_different_lags_saturation(df, df_pc, ind_var, dep_var, ind_src='sw', bounds=None, restrict=True, skip_zero=False, **kwargs):
    """
    Plots driver-response on one set of axes for many lag times
    To see if a particular lag time shows stronger saturation than others
    """

    kwargs['min_count'] = kwargs.get('min_count',minimum_counts['counts'])
    kwargs['display']   = kwargs.get('display','rolling')
    if kwargs['display']=='rolling':
        kwargs['region'] = kwargs.get('region','sem')

    ind_err, ind_count = def_param_names(df, ind_var)

    bin_width, limits, invert = get_variable_range(ind_var, ind_src, restrict=restrict, bounds=bounds)
    kwargs['window_width'] = bin_width

    df_ind = mask_df(df, ind_var, limits)

    dep_cols = [col for col in df_pc.columns if col.startswith(dep_var)]

    if skip_zero and dep_var in dep_cols:
        dep_cols.remove(dep_var)

    cmap = plt.get_cmap('autumn_r')
    norm = plt.Normalize(vmin=0, vmax=len(dep_cols)-1)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

    for i, col in enumerate(dep_cols):

        dep = col
        num_parts = len(col.split('_'))
        if num_parts==3:
            lag = col.split('_')[1] + r' + $\delta t$'
        elif num_parts==2:
            lag = col.split('_')[1]
        else:
            lag = '0m'

        df_dep = mask_df(df_pc, dep_var)
        intersect = df_ind.index.intersection(df_dep.index)
        df_ind_masked = df_ind.loc[intersect]
        df_dep_masked = df_dep.loc[intersect]

        colour = cmap(norm(i))
        kwargs['data_colour'] = colour
        kwargs['error_colour'] = colour

        _ = compare_dataframes(df_ind_masked, df_dep_masked, ind_var, dep, col1_err=ind_err, col1_counts=ind_count, fig=fig, ax=ax, return_objs=True, **kwargs)

        ax.plot([], [], ls='-', color=colour, label=lag)

    ax.set_ylabel(create_label(dep_var,units=df_pc.attrs['units']))

    add_legend(fig, ax)
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

def plot_different_lags_fits(df, df_pc, ind_var, dep_var, ind_src='sw', bounds=None, restrict=True, skip_zero=True, **kwargs):
    """
    Fits a model to driver-reponse data and retrieves the fitted parameters
    The parameters are then plotted as a function of lag time, to see if there's any best value
    Flat lines indicates the saturation is not sensitive to choice of static lag time
    """

    kwargs['fit_type'] = kwargs.get('fit_type','saturation')

    ind_err, ind_count = def_param_names(df, ind_var)
    dep_err, dep_count = def_param_names(df_pc, dep_var)

    _, limits, _ = get_variable_range(ind_var, ind_src, restrict=restrict, bounds=bounds)

    df_ind = mask_df(df, ind_var, limits)

    dep_vars = get_lagged_columns(df_pc, dep_var, skip_zero)

    fig, (ax, ax2) = plot_fit_params_against_z(df_ind, ind_var, dep_vars, df_dep=df_pc, col1_err=ind_err, col2_err=dep_err, col1_counts=ind_count, col2_counts=dep_count, **kwargs)
    ax.set_xlabel('Lag [mins]')

    ind_str = format_string(ind_var)
    dep_str = format_string(dep_var)
    ax.set_title(f'Fitting ${dep_str}$ against {ind_src.upper()} ${ind_str}$')

    file_name = f'Fitting_{kwargs["fit_type"]}_of_{dep_var}_to_{ind_src}_{ind_var}_lags'
    save_figure(fig, file_name=file_name, sub_directory='Comparing_Lags')
    plt.show()
    plt.close()

def plot_compare_sources_with_lags(df_omni, df_sc, df_pc, ind_var, dep='PC', omni_colour=blue, contemp_colour=black, sc_colour=pink, restrict=True, shift_centre=True, contemp_omni=False, **kwargs):
    """
    2x2 grid comparing OMNI & in-situ as input and Index & Mag. as output
    Dep_var is either 'PC' for comparing PC and THL or 'AE' for comparing AE and SME
    This plots the fitted function parameters over different lags, for comparison
    """

    if df_pc is None:
        raise ValueError('Polar cap dataframe is none.')

    kwargs['fit_type']  = kwargs.get('fit_type','saturation')

    if 'data1_name' in kwargs:
        kwargs['data1_name'] = create_label(kwargs['data1_name'])


    dep_cols = {'PCN': ['PCN','SMC_y_GSM'],
                'PCNC': ['PCN','SMC'],
                'PCC': ['PCC','SMC'],
                'AE':  ['AE', 'SME']}

    dep_vars = dep_cols.get(dep,None)
    if not dep_vars:
        raise Exception(f'"{dep}" not implemented.')


    ###----------PLOT GRIDS----------###
    n_rows, n_cols = 2, 2

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows), dpi=200, sharex='col')
    kwargs['fig'] = fig

    for i, ((df,ind_src), dep_var) in enumerate(it.product([(df_omni,'omni'),(df_sc,'sw')],dep_vars)):
        print(ind_src,dep_var)

        row, col = i % n_rows, i // n_rows
        ax = axs[row][col]

        if df is df_omni and contemp_omni:
            overlap = df.index.intersection(df_sc.index)
            df = df.loc[overlap]

        ind_err, ind_count = def_param_names(df, ind_var)
        dep_err, dep_count = def_param_names(df_pc, dep_var)

        _, limits, _ = get_variable_range(ind_var, ind_src, restrict=restrict)

        df_ind = mask_df(df, ind_var, limits)

        dep_lagged = get_lagged_columns(df_pc, dep_var)

        _, (ax, ax2) = plot_fit_params_against_z(df_ind, ind_var, dep_lagged, df_dep=df_pc, col1_err=ind_err, col2_err=dep_err, col1_counts=ind_count, col2_counts=dep_count, ax=ax, **kwargs)

        ###----------FORMATTING----------###
        ind_str = format_string(ind_var)
        dep_str = format_string(dep_var)
        ax.set_title(f'${dep_str}$ vs {ind_src.upper()} ${ind_str}$')

        if row==n_rows-1:
            ax.tick_params(labelbottom=True)
            ax.set_xlabel('Lags [mins]')
        else:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel(None)

        if col==0:
            ax2.set_ylabel(None)
        else:
            ax.set_ylabel(None)

        if i!=0:
            ax.get_legend().remove()

    if n_cols>1:
        fig.align_ylabels(axs[:,0])
    axs[0][0].text(0.02, 0.95, kwargs.get('region',''), transform=axs[0][0].transAxes, va='top', ha='left')

    file_name = f'Comparing_{ind_var}_{dep}_fit_{kwargs["fit_type"]}'

    plt.tight_layout()
    save_figure(fig, file_name=file_name, sub_directory='Driver_Response/Lags')
    plt.show()
    plt.close()