# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:55:08 2025

@author: richarj2
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from datetime import datetime

from .utils import minimum_counts, def_param_names, get_variable_range, shift_angular_data, mask_df, get_variable_bin_width

from ..delay import merge_with_lag, get_response_delay

from ....analysing.fitting import fit_bootstrap, get_confidence_range
from ....analysing.calculations import calc_mean_error

from ....plotting.utils import save_figure, calculate_bins, save_text
from ....plotting.formatting import shifted_angle_ticks, format_string
from ....plotting.comparing.parameter import compare_dataframes
from ....plotting.config import black, blue, grey, green, red

from ....processing.reading import import_processed_data, import_processed_index

def plot_driver_response(ind_var, dep_var, df_sc=None, df_pc=None, spacecraft='combined', resolution='5min', region='sw', restrict=True, second_x=False, plot_histograms=False, fig=None, ax=None, return_objs=False, **kwargs):
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

    limits, invert = get_variable_range(ind_var, region, dep_var=dep_var, restrict=restrict, shift_centre=shift_centre)

    sc_colour = {'sw': blue, 'msh': red, 'msh20': green}.get(region)

    kwargs.setdefault('min_count', minimum_counts[resolution])
    kwargs.setdefault('display', 'scatter_binned')
    kwargs.setdefault('data_colour', sc_colour)
    kwargs.setdefault('error_colour', sc_colour)
    kwargs.setdefault('fit_type', 'saturation')
    kwargs.setdefault('fit_colour', black)
    kwargs.setdefault('iterations', 2500)

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

    delay = get_response_delay(dep_var)

    df_ind, df_dep = merge_with_lag(df_ind, df_dep, delay, resolution)

    ###----------PLOT ----------###

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6.5 if second_x else 5), dpi=400)

    x_min, x_max = np.min(df_ind[ind_var]), np.max(df_ind[ind_var])
    x_range = np.linspace(x_min, x_max, 100)

    # Binned scatter
    kwargs['bin_step'] = get_variable_bin_width(df_ind[ind_var])
    fit_dict = compare_dataframes(df_ind, df_dep, ind_var, dep_var, fig=fig, ax=ax, return_objs=True, return_fit=True, **kwargs)[-1]

    x0 = fit_dict['params']['x0']
    if x0 > np.max(df_ind[ind_var]): # suggests no saturation in range

        print(f'No saturation in range (x0 = {x0:.4g}).')
        ax.clear()
        kwargs = kwargs.copy()
        kwargs['fit_type'] = 'straight'
        kwargs['x_range'] = x_range

        fit_dict = compare_dataframes(df_ind, df_dep, ind_var, dep_var, fig=fig, ax=ax, return_objs=True, return_fit=True, **kwargs)[-1]

    ###----------BOOTSTRAP----------###

    num_iters = kwargs['iterations']
    if num_iters>0:

        df_boot = fit_bootstrap(df_ind, ind_var, dep_var, df_dep, **kwargs)

        confidence = 95
        param_intervals, (y_lower, y_upper) = get_confidence_range(df_boot, x_range, kwargs['fit_type'], confidence)

        ax.fill_between(x_range, y_lower, y_upper, color=black, alpha=0.2)

        print(f'\nConfidence intervals ({confidence}%)')
        for param, quantiles in param_intervals.items():
            print(f'{param}:\t{quantiles["low"]:.4g}\t|\t{quantiles["high"]:.4g}')
        print()

        # Save to txt file

        if kwargs.get('save_bootstrap',True):

            txt_file_name, string = bootstrap_string(df_boot, ind_var, dep_var, region, fit_dict, cond_col, condition, param_intervals, confidence, num_iters, x_min, x_max)

            save_text(string, sub_directory='Bootstraps', file_name=txt_file_name)

    ###----------FORMATTING----------###

    if ind_var.startswith('-'):
        ind_var = ind_var[1:]

    if shift_centre:
        ax.axvline(x=np.pi, c=grey, ls=':')
        shifted_angle_ticks(ax, 'x')
        ax.invert_xaxis()

    second_x, functions, Vx = convert_axis(second_x, ind_var, region)

    if second_x:

        ax.set_xlim(0)

        ax2 = ax.secondary_xaxis(-0.2, functions=functions)
        B_label = r'|\boldsymbol{B}|'
        ax2.set_xlabel(f'${B_label}$ [nT] ($V_x=${Vx} km/s)', fontsize=12)

    if return_objs:
        return fig, ax

    ###----------SAVING----------###

    file_name = f'Coupling_{region}_{ind_var}_driving_{dep_var}_{num_iters}'
    if kwargs['display'] != 'scatter_binned':
        file_name += f'_{kwargs["display"]}'
    if kwargs.get('file_suffix') is not None:
        file_name += f'_{kwargs.get("file_suffix")}'

    plt.tight_layout()
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()

    #return df_boot

def bootstrap_string(df_boot, ind_var, dep_var, region, fit_dict, cond_col, condition, param_intervals, confidence, num_iters, x_min, x_max):

    time = datetime.now().strftime('%H%M%S')
    file_name = f'Bootstrap_{dep_var}_vs_{ind_var}_{region}_{time}'

    n_samples = df_boot.attrs.get('samples')
    n_days    = df_boot.attrs.get('days')

    string  = f'Fitting to {region} {ind_var} data for {dep_var} response.\n'
    string += '--------------------\n\n'

    string += f'Fitted over {ind_var} in [{x_min:.4g}, {x_max:.4g}].\n'
    string += ('No condition/restriction applied.\n\n') if condition is None else (f'Condition applied: {cond_col} {condition}.\n\n')

    string += f'Bootstrapping N = {num_iters}.\n'
    string += f'{n_samples:,} samples | {n_days:,} days.\n'
    string += '--------------------\n'

    string += '\nParameters from all data:\n'
    for param, val in fit_dict['params'].items():
        unit = fit_dict['units'][param]
        string += f'{param}:\t {val.n:.5g} [{unit}]\n'

    string += '\nMedian Values:\n'
    for param, quantiles in param_intervals.items():
        unit = fit_dict['units'][param]
        string += f'{param}:\t{quantiles["med"]:.5g}\n'

    string += f'\nConfidence intervals ({confidence}%):\n'
    for param, quantiles in param_intervals.items():
        unit = fit_dict['units'][param]
        string += f'{param}:\t{quantiles["low"]:.5g}\t|\t{quantiles["high"]:.5g}\n'

    string += '--------------------\n'

    return file_name, string

def convert_axis(second_x, var, region):

    functions = []

    Vx = -400 if region=='sw' else -150 # km/s

    if second_x:

        if var.startswith('E_y'):

            B_of_E = lambda E: E / (-Vx * 1e-3) # B = B(E)
            E_of_B = lambda B: B * (-Vx * 1e-3) # E = E(B)

            functions = (B_of_E, E_of_B)

        elif var.startswith('S_x'):

            mu0 = 4 * np.pi * 1e-7

            # Using -S in figure
            # factor of 3 to account for cross terms
            def B_of_S(S): # B = B(S)
                return 1e9 * np.sqrt(-S * 1e-6 * mu0 / (Vx * 1e3)) * np.sqrt(3)

            def S_of_B(B): # S = S(B)
                return -(B * 1e-9)**2 * (Vx * 1e3) /(mu0 * 1e-6) / np.sqrt(3)

            functions = (B_of_S, S_of_B)

        else:
            second_x = False

    return second_x, functions, Vx

# %% grids

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

    height = (6 if kwargs.get('second_x',False) else 5)*n_rows

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, height), dpi=400, sharex=sharex, sharey='row', squeeze=False) # squeeze keeps axs as 2D even if n_cols or n_rows = 1

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

    file_name = f'Multi_Driver_Response_for_{region}_'
    file_name += '_'.join(ind_vars) + '_'
    file_name += '_'.join(dep_vars)

    plt.tight_layout()
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()

def plot_compare_systems(ind_vars, dep_vars, spacecraft='combined', resolution='5min', region='sw', restrict=True, **kwargs):
    """
    Look at OMNI and in-situ data in driver-response
    The same ind_var is used for the omni and sc data, then a variety of dep_vars are shown in separate columns
    """

    kwargs.setdefault('display', 'scatter_binned')
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

    height = (6 if kwargs.get('second_x',False) else 5)*n_rows + 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, height), dpi=400, sharex='row', sharey='row', squeeze=False)
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
    if condition=='sine':
        title = f'$\\sin^2\\left({format_string(cond_col)}/2\\right)$'
    else:
        title = f'${format_string(cond_col)}${condition}'
    axs[0,1].set_title(title)

    file_name = f'Comparing_{dep_var}_responses_with_restriction_{cond_col}' + '_'.join(ind_vars)
    if kwargs['display'] != 'scatter_binned':
        file_name += f'_{kwargs["display"]}'

    plt.tight_layout()
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()


def plot_compare_regions(ind_vars, dep_vars, spacecraft='combined', resolution='5min', regions=('sw','msh'), restrict=True, **kwargs):
    """
    Look at OMNI and in-situ data in driver-response
    The same ind_var is used for the omni and sc data, then a variety of dep_vars are shown in separate columns
    """

    kwargs.setdefault('display', 'scatter_binned')

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

    height = (6 if kwargs.get('second_x',False) else 5)*n_rows + 1

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, height), dpi=400, sharex='col', sharey='row', squeeze=False)
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
    if kwargs['display'] != 'scatter_binned':
        file_name += f'_{kwargs["display"]}'

    plt.tight_layout()
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()

# %% Delay Histograms

def plot_delay_hists(sc, regions=('sw','msh'), data_pop='plasma', resolution='5min', close=True):

    fig, ax = plt.subplots(figsize=(10,6))

    for region in regions:

        df = import_processed_data(region, dtype=data_pop, resolution=resolution, file_name=f'{region}_times_{sc}')
        if close: # to exclude artemis
            df = df.loc[df['r_x_GSE']<30]

        lags = df.loc[~df['prop_time_s'].isna(),'prop_time_s'] / 60
        unit = 'mins'
        lags.attrs['units']['prop_time_s'] = unit

        colour = {'sw': blue, 'msh': red, 'msh20': green}.get(region)
        location = {'sw': 'upper left', 'msh': 'upper right', 'msh20': 'right'}.get(region)

        bins = calculate_bins(lags,1)
        counts, _ = np.histogram(lags, bins=bins)
        bins = bins[:-1] + 0.5

        mean = calc_mean_error(lags)
        std = np.std(lags, ddof=1)
        label = f'{region}: {len(lags):,}\n$\\mu={mean:L}$ {unit}\n$\\sigma = {std:.3g}$ mins'

        ax.bar(bins, counts, width=1, color=colour, alpha=0.7, label=label)

        legend = ax.legend(handles=[ax.containers[-1]], labels=[label], loc=location, fontsize=10)
        ax.add_artist(legend)

    ax.invert_xaxis() # so positive goes to the left resembling physical system
    ax.set_xlabel('Lag to BSN [mins]')

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))
    ax.set_ylabel('Counts', c=black)

    file_name = f'lags_{sc}_{resolution}'

    plt.tight_layout();
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()
