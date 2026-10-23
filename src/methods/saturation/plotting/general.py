# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:55:08 2025

@author: richarj2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

from .utils import minimum_counts, def_param_names, get_variable_range, shift_angular_data, mask_df

from ..delay import merge_with_lag, get_response_delay

from ....analysing.fitting import fit_bootstrap, get_confidence_range
from ....analysing.calculations import calc_mean_error

from ....plotting.utils import save_figure, calculate_bins
from ....plotting.formatting import shifted_angle_ticks, format_string
from ....plotting.comparing.parameter import compare_dataframes
from ....plotting.config import black, blue, grey, green, red

from ....processing.reading import import_processed_data
from ....processing.mag.indices import import_processed_index

def plot_driver_response(ind_var, dep_var, df_sc=None, df_pc=None, spacecraft='combined', resolution='5min', region='sw', restrict=True, second_x=False, fig=None, ax=None, return_objs=False, **kwargs):
    """
    Plots a driver-response function, and fits a model to the data.
    """

    ###----------IMPORTS----------###

    if df_sc is None:
        df_sc = import_processed_data(region, dtype='plasma', resolution=resolution, file_name=f'{region}_times_{spacecraft}')

    if df_pc is None:
        df_pc = import_processed_index(dep_var, resolution=resolution, return_series=False, return_full=True)

    print(f'{region.upper()} {ind_var} | {dep_var}')

    _, ind_count = def_param_names(df_sc, ind_var) # region not in name
    _, dep_count = def_param_names(df_pc, dep_var)

    shift_centre = ind_var=='B_clock'

    bin_width, limits, invert = get_variable_range(ind_var, region, dep_var=dep_var, restrict=restrict, shift_centre=shift_centre)

    sc_colour = {'sw': blue, 'msh': red, 'msh20': green}.get(region)

    kwargs.setdefault('min_count', minimum_counts[resolution])
    kwargs.setdefault('display', 'scatter_binned')
    kwargs.setdefault('data_colour', sc_colour)
    kwargs.setdefault('error_colour', sc_colour)
    kwargs.setdefault('fit_type', 'saturation')
    kwargs.setdefault('fit_colour', black)
    kwargs.setdefault('iterations', 2500)

    kwargs['bin_step'] = bin_width

    ###----------CONFIG----------###

    condition = kwargs.get('condition')
    cond_col  = kwargs.get('cond_col')

    if ind_var in ('E_R', 'B_z_GSM'):
        print(f'Condition doesn\'t apply to {ind_var}.')
        condition = None

    # Masks and slicing
    df_ind = mask_df(df_sc, ind_var, limits, ind_count, resolution, condition, cond_col, invert) # removes low sampling counts too
    df_dep = mask_df(df_pc, dep_var)

    if shift_centre:
        shift_angular_data(df_ind, ind_var)

    if invert:
        ind_var = f'-{ind_var}'

    lag = get_response_delay(dep_var)

    df_ind, df_dep = merge_with_lag(df_ind, df_dep, lag, resolution)

    ###----------PLOT ----------###

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6.5 if second_x else 5), dpi=400)

    x_range = np.linspace(np.min(df_ind[ind_var]), np.max(df_ind[ind_var]), 100)

    # Binned scatter
    fit_dict = compare_dataframes(df_ind, df_dep, ind_var, dep_var, fig=fig, ax=ax, return_objs=True, return_fit=True, **kwargs)[-1]

    if fit_dict['params']['x0']>np.max(df_ind[ind_var]): # suggests no saturation in range

        print('No saturation in range.')
        ax.clear()
        kwargs = kwargs.copy()
        kwargs['fit_type'] = 'straight'
        kwargs['x_range'] = x_range

        _ = compare_dataframes(df_ind, df_dep, ind_var, dep_var, fig=fig, ax=ax, return_objs=True, return_fit=True, **kwargs)

    ###----------BOOTSTRAP----------###

    num_iters = kwargs['iterations']
    if num_iters>0:

        df_boot = fit_bootstrap(df_ind, ind_var, dep_var, df_dep, **kwargs)

        confidence = 95
        param_intervals, (lower, upper) = get_confidence_range(df_boot, x_range, kwargs['fit_type'], confidence)

        ax.fill_between(x_range, lower, upper, color=black, alpha=0.2)

        print(f'\nConfidence intervals ({confidence}%)')
        for key, val in param_intervals.items():
            print(f'{key}:\t{val[0]:.4g}\t|\t{val[1]:.4g}')
        print()

    ###----------FORMATTING----------###

    if ind_var.startswith('-'):
        ind_var = ind_var[1:]

    if shift_centre:
        ax.axvline(x=np.pi, c=grey, ls=':')
        shifted_angle_ticks(ax, 'x')
        ax.invert_xaxis()

    if second_x:

        Vx = -400 if region=='sw' else -200 # km/s

        if ind_var.startswith('E_y'):

            factor = -1000/Vx
            functions = (lambda E: E * factor, lambda B: B / factor)

        elif ind_var.startswith('S_x'):
            mu0 = 4 * np.pi * 1e-7

            # Using -S in figure
            def S_to_B(S):
                return 1e9 * np.sqrt(-S * 1e-6 * mu0 / (Vx * 1e3))

            def B_to_S(B):
                return -(B * 1e-9)**2 * Vx * 1e3 / mu0 * 1e6

            functions = (S_to_B, B_to_S)

        else:
            second_x = False

    if second_x:

        ax.set_xlim(0)

        ax2 = ax.secondary_xaxis(-0.2, functions=functions)
        ax2.set_xlabel(f'$-B_z$ [nT] ($V_x$={Vx} km/s)', fontsize=12)

    if return_objs:
        return fig, ax

    file_name = f'Coupling_{region}_{ind_var}_driving_{dep_var}_{num_iters}'
    if kwargs.get('file_suffix') is not None:
        file_name += f'_{kwargs.get("file_suffix")}'

    plt.tight_layout()
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()

    #return df_boot


def plot_multi_driver_responses(ind_vars, dep_vars, spacecraft='combined', resolution='5min', region='sw', restrict=True, **kwargs):
    """
    Look at OMNI and in-situ data in driver-response
    The same ind_var is used for the omni and sc data, then a variety of dep_vars are shown in separate columns
    """

    ###----------PLOT GRIDS----------###
    n_rows, n_cols = len(dep_vars), len(ind_vars)

    if n_cols==1: # one driver

        n_cols = len(dep_vars)
        n_rows = 1

        sharex = 'col'

    else:

        sharex = 'row'

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows + 1.5), dpi=300, sharex=sharex, sharey='row', squeeze=False) # squeeze keeps axs as 2D even if n_cols or n_rows = 1

    if n_rows == 1: # for iteration below
        axs = axs.T

    df_sc = import_processed_data(region, dtype='plasma', resolution=resolution, file_name=f'{region}_times_{spacecraft}')

    for row, dep_var in enumerate(dep_vars):

        df_pc = import_processed_index(dep_var, resolution=resolution, return_series=False, return_full=True)

        for col, ind_var in enumerate(ind_vars):

            ax = axs[row, col]

            _ = plot_driver_response(ind_var, dep_var, df_sc, df_pc, spacecraft, resolution, region, restrict, fig=fig, ax=ax, return_objs=True, **kwargs)

            # Formatting
            if row != (n_rows-1) and n_rows!=1:
                ax.set_xlabel(None)
            if col != 0:
                ax.set_ylabel(None)

    file_name = f'Multi_Driver_Response_for_{region}'

    plt.tight_layout()
    save_figure(fig, file_name=file_name)
    plt.show()
    plt.close()

def plot_compare_systems(ind_vars, dep_vars, spacecraft='combined', resolution='5min', region='sw', restrict=True, **kwargs):
    """
    Look at OMNI and in-situ data in driver-response
    The same ind_var is used for the omni and sc data, then a variety of dep_vars are shown in separate columns
    """

    condition = kwargs.get('condition')
    cond_col  = kwargs.get('cond_col')

    if condition is None:
        raise ValueError('Haven\'t passed in a condition.')

    ###----------PLOT GRIDS----------###
    if len(dep_vars)>1 and len(ind_vars)>1:
        raise Exception('Can\'t have more than one independent and dependent variable.')

    elif len(dep_vars)==1:
        n_rows     = len(ind_vars)
        dep_var    = dep_vars[0]
        enumerator = ind_vars
        var_type   = 'ind'

    elif len(ind_vars)==1:
        n_rows     = len(dep_vars)
        ind_var    = ind_vars[0]
        enumerator = dep_vars
        var_type   = 'dep'

    n_cols = 2

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows), dpi=300, sharex='row', sharey='row', squeeze=False)
    # squeeze keeps axs as 2D even if n_cols or n_rows = 1

    df_sc = import_processed_data(region, dtype='plasma', resolution=resolution, file_name=f'{region}_times_{spacecraft}')

    if var_type=='ind': # to prevent importing in the loop
        df_pc = import_processed_index(dep_var, resolution=resolution, return_series=False, return_full=True)

    for row, var in enumerate(enumerator):

        if var_type=='ind':
            ind_var = var
        else:
            dep_var = var

            df_pc = import_processed_index(dep_var, resolution=resolution, return_series=False, return_full=True)

        ###-----WITHOUT-----###
        ax = axs[row, 0]
        kwargs_copy = kwargs.copy()
        kwargs_copy['condition'] = None

        _ = plot_driver_response(ind_var, dep_var, df_sc, df_pc, spacecraft, resolution, region, restrict, fig=fig, ax=ax, return_objs=True, **kwargs_copy)

        ###-----WITH-----###
        ax = axs[row, 1]

        _ = plot_driver_response(ind_var, dep_var, df_sc, df_pc, spacecraft, resolution, region, restrict, fig=fig, ax=ax, return_objs=True, **kwargs)

        ax.set_ylabel(None)

    axs[0,0].set_title('Unrestricted')
    axs[0,1].set_title(f'${format_string(cond_col)}${condition}')

    file_name = f'Comparing_{dep_var}_responses_with_restriction_' + '_'.join(ind_vars)

    plt.tight_layout()
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()


def plot_compare_regions(ind_vars, dep_vars, spacecraft='combined', resolution='5min', regions=('sw','msh'), restrict=True, **kwargs):
    """
    Look at OMNI and in-situ data in driver-response
    The same ind_var is used for the omni and sc data, then a variety of dep_vars are shown in separate columns
    """

    ###----------PLOT GRIDS----------###
    if len(dep_vars)>1 and len(ind_vars)>1:
        raise Exception('Can\'t have more than one independent and dependent variable.')

    elif len(dep_vars)==1:
        n_rows     = len(ind_vars)
        dep_var    = dep_vars[0]
        enumerator = ind_vars
        var_type   = 'ind'

    elif len(ind_vars)==1:
        n_rows     = len(dep_vars)
        ind_var    = ind_vars[0]
        enumerator = dep_vars
        var_type   = 'dep'

    n_cols = len(regions)

    second_x = kwargs.get('second_x',False)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, (5.5 if second_x else 5)*n_rows), dpi=300, sharex='col', sharey='row', squeeze=False)
    # squeeze keeps axs as 2D even if n_cols or n_rows = 1

    dfs = {}
    for region in regions:
        df = import_processed_data(region, dtype='plasma', resolution=resolution, file_name=f'{region}_times_{spacecraft}')
        dfs[region] = df

    if var_type=='ind': # to prevent importing in the loop
        df_pc = import_processed_index(dep_var, resolution=resolution, return_series=False, return_full=True)

    for row, var in enumerate(enumerator):

        if var_type=='ind':
            ind_var = var
        else:
            dep_var = var

            df_pc = import_processed_index(dep_var, resolution=resolution, return_series=False, return_full=True)

        ###-----REGIONS-----###
        for col, (region, df) in enumerate(dfs.items()):
            ax = axs[row, col]

            _ = plot_driver_response(ind_var, dep_var, df, df_pc, spacecraft, resolution, region, restrict, fig=fig, ax=ax, return_objs=True, **kwargs)

            if col!=0:
                ax.set_ylabel(None)
            if row==0:
                ax.text(0.025, 0.95, region, transform=ax.transAxes, ha='left', va='top', fontsize=8)

    file_name = f'Comparing_{dep_var}_{ind_var}_for_regions_' + '_'.join(regions)

    plt.tight_layout()
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()

# %% Delay Histograms

def plot_delay_hists(sc, regions=('sw','msh'), data_pop='plasma', resolution='5min', close=True):

    limit = 30
    if close: # mainly to exclude artemis
        limit = 15

    fig, ax = plt.subplots(figsize=(8,6))

    for region in regions:

        df = import_processed_data(region, dtype=data_pop, resolution=resolution, file_name=f'{region}_times_{sc}')
        if close:# mainly to exclude artemis
            df = df.loc[df['r_mag']<35]

        lags = df.loc[~df['prop_time_s'].isna(),'prop_time_s'] / 60
        unit = 'mins'
        lags.attrs['units']['prop_time_s'] = unit

        colour = {'sw': blue, 'msh': red, 'msh20': green}.get(region)
        location = {'sw': 'upper left', 'msh': 'upper right', 'msh20': 'right'}.get(region)

        bins = calculate_bins(lags,1)
        counts, _ = np.histogram(lags, bins=bins)
        bins = bins[:-1]

        mean = calc_mean_error(lags)
        std = np.std(lags, ddof=1)
        label = f'{region}: {len(lags):,}\n$\\mu={mean:L}$ {unit}\n$\\sigma = {std:.3g}$ mins'

        ax.bar(bins, counts, width=1, color=colour, alpha=0.7, label=label)

        legend = ax.legend(handles=[ax.containers[-1]], labels=[label], loc=location, fontsize=10)
        ax.add_artist(legend)

    ax.invert_xaxis() # so positive goes to the left resembling physical system
    ax.set_xlim(limit,-limit)
    ax.set_xlabel('Lag to BSN [mins]')

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))
    ax.set_ylabel('Counts', c=black)

    file_name = f'lags_{sc}_{resolution}'


    plt.tight_layout();
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()
