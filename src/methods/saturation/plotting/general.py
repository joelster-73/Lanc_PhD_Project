# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:55:08 2025

@author: richarj2
"""

import numpy as np
import matplotlib.pyplot as plt

from .utils import minimum_counts, def_param_names, get_variable_range, shift_angular_data, mask_df

from ..delay import merge_with_lag, get_response_delay

from ....analysing.fitting import fit_bootstrap, get_confidence_range

from ....plotting.utils import save_figure, get_grid_shape
from ....plotting.formatting import create_label, shifted_angle_ticks
from ....plotting.comparing.parameter import compare_dataframes
from ....plotting.config import black, blue, grey, green, white, red
from ....plotting.distributions import plot_freq_hist

from ....processing.reading import import_processed_data, import_updated_omni
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
    kwargs.setdefault('iterations', 2500)

    kwargs['bin_step'] = bin_width

    if 'data1_name' in kwargs:
        kwargs['data1_name'] = create_label(kwargs['data1_name'])
    if 'data_name_map' in kwargs:
        kwargs['data2_name'] = create_label(kwargs['data_name_map'].get(dep_var,dep_var))

    ###----------CONFIG----------###

    condition = kwargs.get('condition')
    cond_col  = kwargs.get('cond_col')

    # Masks and slicing
    df_ind = mask_df(df_sc, ind_var, limits, ind_count, resolution, condition, cond_col) # removes low sampling counts too
    df_dep = mask_df(df_pc, dep_var)

    if shift_centre:
        shift_angular_data(df_ind, ind_var)

    if invert:
        df_ind[ind_var] *= -1
        df_ind.rename(columns={ind_var: f'-{ind_var}'}, inplace=True) ##### I THINK I HAVE FUNCTON FOR THIS
        df_ind.attrs.get('units',{})[f'-{ind_var}'] = df_ind.attrs.get('units',{}).get(ind_var)
        ind_var = f'-{ind_var}'

    lag = get_response_delay(dep_var)

    df_ind, df_dep = merge_with_lag(df_ind, df_dep, lag, resolution)

    ###----------PLOT ----------###

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6 if second_x else 5), dpi=400)

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

    if False:

        df_boot = fit_bootstrap(df_ind, ind_var, dep_var, df_dep, **kwargs)

        confidence = 95
        param_intervals, (lower, upper) = get_confidence_range(df_boot, x_range, kwargs['fit_type'], confidence)

        ax.fill_between(x_range, lower, upper, color=black, alpha=0.2)

        print(f'\nConfidence intervals ({confidence}%)')
        for key, val in param_intervals.items():
            print(f'{key}:\t{val[0]:.4g}\t| {val[1]:.4g}')
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

    file_name = f'Response_from_{region}_{ind_var}_on_{dep_var}'
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

    #-----
    # try importing all here, if too much memory, then keep import in each loop
    #-----

    ###----------PLOT GRIDS----------###
    n_rows, n_cols = len(dep_vars), len(ind_vars)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows), dpi=300, sharex='col', sharey='row', squeeze=False)
    # squeeze keeps axs as 2D even if n_cols or n_rows = 1

    df_sc = import_processed_data(region, dtype='plasma', resolution=resolution, file_name=f'{region}_times_{spacecraft}')

    for row, dep_var in enumerate(dep_vars):

        df_pc = import_processed_index(dep_var, resolution=resolution, return_series=False, return_full=True)

        for col, ind_var in enumerate(ind_vars):

            ax = axs[row, col]

            plot_driver_response(ind_var, dep_var, df_sc, df_pc, spacecraft, resolution, region, restrict, fig=fig, ax=ax, return_objs=True, **kwargs)

            # Formatting
            if row != (n_rows-1):
                ax.set_xlabel(None)
            if col != 0:
                ax.set_ylabel(None)

    file_name = f'Multi_Driver_Response_for_{region}'

    plt.tight_layout()
    save_figure(fig, file_name=file_name)
    plt.show()
    plt.close()

# %% Pulk

def plot_pulkkinen_grid(*params, ind_reg='sw', dep_reg='msh', resolution='5min', bounds=None, restrict=True, shift_centre=True, compare_colour=green, ind_src='combined', dep_src='combined', **kwargs):
    """
    Pulkkinen comparisons

    So plots OMNI vs sc (sw or msh) for range of parameters
    """

    ###----------IMPORTS----------###

    ind_suff = ind_reg
    if ind_reg=='omni':
        df1 = import_updated_omni(resolution=resolution)
    elif ind_reg in ('sw','msh'):
        df1 = import_processed_data(ind_reg, dtype='plasma', resolution=resolution, file_name=f'{ind_reg}_times_{ind_src}')
        ind_suff += f'_{ind_src}' if ind_src!='combined' else ''
    else:
        raise ValueError(f'"{ind_reg}" is not a valid source.')

    dep_suff = dep_reg
    if dep_reg=='omni':
        df2 = import_updated_omni(resolution=resolution)
    elif dep_reg in ('sw','msh'):
        df2 = import_processed_data(dep_reg, dtype='plasma', resolution=resolution, file_name=f'{dep_reg}_times_{dep_src}')
        dep_suff += f'_{dep_src}' if dep_src!='combined' else ''
    else:
        raise ValueError(f'"{dep_reg}" is not a valid source.')

    kwargs['data_colour']  = compare_colour
    kwargs['error_colour'] = compare_colour

    enumerator = [p if isinstance(p,tuple) else (p,p) for p in params]

    n_rows, n_cols = get_grid_shape(len(enumerator))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 5*n_rows), dpi=400)

    ###----------PLOT GRIDS----------###

    for i, (independent, dependent), in enumerate(enumerator):

        col, row = i % n_cols, i // n_cols

        if len(enumerator)>1:
            ax = axs[row][col]
        else:
            ax = axs

        ind_err, ind_count = def_param_names(df1, independent)
        dep_err, dep_count = def_param_names(df2, dependent)

        bin_width, limits, invert = get_variable_range(independent, ind_reg, dep_var=dependent, restrict=restrict, bounds=bounds, shift_centre=shift_centre)

        # Shift angular data to centre lies at +-180 rather than 0
        if independent=='B_clock' and shift_centre:
            shift_angular_data(df1, independent)
            shift_angular_data(df2, dependent)

        df_ind = mask_df(df1, independent, limits)
        df_dep = mask_df(df2, dependent)

        if ind_reg in ('sw','msh'):
            df_ind, df_dep = merge_with_lag(df_ind, df_dep, 0, resolution)
        elif dep_reg in ('sw','msh') and ind_reg=='omni':
            df_dep, df_ind = merge_with_lag(df_dep, df_ind, 0, resolution) # the spacecraft is the dependent variable
        else:
            raise ValueError('Not a valid combination.')

        # Config
        kwargs['window_width'] = bin_width
        kwargs['data1_name'] = create_label(f'{independent}_{ind_suff}')
        kwargs['data2_name'] = create_label(f'{dependent}_{dep_suff}')

        if kwargs['display']=='scatter':
            ind_err = None
            dep_err = None
            if f'sc_{dep_reg}' in df_dep:
                kwargs['df3'] = df_dep
                kwargs['col3'] = f'sc_{dep_reg}'
                kwargs['data_colour'] = 'spacecraft'

        objs = compare_dataframes(df_ind, df_dep, independent, dependent, col1_err=ind_err, col1_counts=ind_count, col2_err=dep_err, col2_counts=dep_count, fig=fig, ax=ax, return_objs=True, **kwargs)

        ax.axline((limits[0],limits[0]), slope=1, c=black, ls='-')
        ax.axline((limits[0],limits[0]), slope=1, c=white, ls=':')
        ax.grid(ls=':', c=grey, lw=0.5)

        if independent=='B_clock' and shift_centre:
            ax.axvline(x=np.pi, c=grey, ls=':')
            shifted_angle_ticks(ax, 'x')
            shifted_angle_ticks(ax, 'y')
            ax.set_xlim(limits[0], limits[1])
            ax.set_ylim(limits[0], limits[1])

        if independent.startswith('T_') and kwargs['display']!='heat':
            ax.set_xscale('log')
            ax.set_yscale('log')

        if invert:
            ax.invert_xaxis()
            ax.invert_yaxis()

        if len(objs)==3 and not (col==(n_cols-1) and row==0):
            objs[2].set_label(None)


    if len(enumerator)>1:
        ax0 = axs[0][0]
    else:
        ax0 = axs

    ax0.text(0.02, 0.95, kwargs.get('region',''), transform=ax0.transAxes, va='top', ha='left')

    plt.tight_layout();
    save_figure(fig, file_name=f'{ind_suff}_vs_{dep_suff}_{kwargs["display"]}', sub_directory='Pulkkinen', overwrite=True)
    plt.show()
    plt.close()

# %% Delay Histograms

def plot_delay_hists(sc, region, data_pop='plasma', resolution='5min', close=True):

    df = import_processed_data(region, dtype=data_pop, resolution=resolution, file_name=f'{region}_times_{sc}')

    limit = 30
    if close: # mainly to exclude artemis
        df = df.loc[df['r_mag']<35]
        limit = 15

    lags = df['prop_time_s'] / 60
    lags.attrs['units']['prop_time_s'] = 'mins'

    fig, ax = plot_freq_hist(lags, bin_width=1, data_name='Lag to BSN', brief_title=region.upper(), return_objs=True)

    ax.invert_xaxis() # so positive goes to the left resembling physical system
    ax.set_xlim(limit,-limit)

    file_name = f'lags_{sc}_{region}_{resolution}'

    std = np.std(lags, ddof=1)

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.get_legend()
    legend.get_texts()[0].set_text(f'{legend.get_texts()[0].get_text()}\n$\\sigma = {std:.3g}$ mins')

    plt.tight_layout();
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()
