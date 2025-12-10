# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:55:08 2025

@author: richarj2
"""
import warnings
import os

def short_warn_format(message, category, filename, lineno, line=None):
    # Get just the parent folder and filename, e.g. "magnetosheath_saturation/plotting.py"
    parent = os.path.basename(os.path.dirname(filename))
    base = os.path.basename(filename)
    short_path = f'{parent}/{base}'
    return f'{short_path}:{lineno}: {category.__name__}: {message}\n'

warnings.formatwarning = short_warn_format


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgba

from ...processing.reading import import_processed_data

from ...plotting.space_time import plot_orbit_msh
from ...plotting.utils import save_figure, calculate_bins
from ...plotting.formatting import create_label
from ...plotting.config import colour_dict
from ...plotting.config import black, bar_hatches
#from ...plotting.distributions import plot_fit

minimum_counts = {'mins': 100, 'counts': 50}


def plot_sc_orbits(sc_dir, sc_keys=None, data_type='mins', region='msh'):

    if sc_keys is None:
        sc_keys = ('c1','mms1','tha','thc','thd','the')

    n_cols = min(3,len(sc_keys))
    n_rows = round(len(sc_keys)/n_cols)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4*(n_cols+1),6*(n_rows+1)), dpi=400)

    for i, sc_key in enumerate(sc_keys):

        df_sc = import_processed_data(sc_dir, f'{region}_times_{sc_key}.cdf')

        row = i % n_rows
        col = i // n_rows

        if len(sc_keys)==1:
            ax = axs
        elif n_rows==1:
            ax = axs[col]
        else:
            ax = axs[row,col]

        title = f'{sc_key}: {len(df_sc[f"B_avg_{sc_key}"].dropna()):,} {data_type}'

        _, _, cbar, _ = plot_orbit_msh(df_sc, sc_keys=sc_key, title=title, region=region, fig=fig, ax=ax, return_objs=True)

        if col!=0:
            ax.set_ylabel(None)
        if col!=(n_cols-1):
            cbar.set_label(None)
        if row!=(n_rows-1) and n_rows!=1:
            ax.set_xlabel(None)

    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

def plot_sc_sw_msh(sw_dir, msh_dir, sw_keys=None, msh_keys=None, data_type='mins'):


    if sw_keys is None:
        sw_keys = ('c1','mms1','thb')

    if msh_keys is None:
        msh_keys = ('c1','mms1','the')

    fig, axs = plt.subplots(2, 1, figsize=(12,8), dpi=400, sharex=True)

    for ax, direc, region, keys, label in zip(axs, (sw_dir,msh_dir), ('sw','msh'), (sw_keys,msh_keys), ('Solar Wind','Magnetosheath')):
        plot_sc_years(direc, region=region, sc_keys=keys, combined=True, data_type=data_type, fig=fig, ax=ax, return_objs=True)
        ax.legend(loc='upper left')
        ax.set_ylabel(label)

    axs[0].set_xlabel(None)

    # Tight format
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.canvas.draw()
    xticks = ax.get_xticks()
    xticks = xticks[(xticks >= ax.get_xlim()[0]) & (xticks <= ax.get_xlim()[1])]
    for ax in axs:
        for x in xticks:
            ax.axvline(x, color=to_rgba(black,0.9), linestyle=':', linewidth=0.5, zorder=20)

    save_figure(fig)
    plt.show()
    plt.close()


def plot_sc_years(sample_interval='1min', region='msh', data_pop='plasma', sc_keys=None, combined=True, **kwargs):

    """
    Combined flag: show all years on one axis, rather than split per spacecraft
    """

    data_type = 'mins' if sample_interval == '1min' else 'counts'


    fig          = kwargs.get('fig',None)
    axs          = kwargs.get('ax',None)
    return_objs  = kwargs.get('return_objs',False)

    if sc_keys is None:
        sc_keys = ('c1','mms1','tha','thb','thc','thd','the')
        if combined:
            sc_keys = ('c1','mms1','th')

    n_rows = len(sc_keys)
    n_cols = 1
    width  = 1
    fig_h  = 2*(n_rows+1)
    fig_w  = 4.5*(n_cols+1)
    if combined:
        n_rows = 1
        width  = 1/len(sc_keys)

    if n_rows==1:
        fig_h = 2*(n_rows+1.5)

    if fig is None or axs is None:

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_w,fig_h), dpi=400, sharex=True)

    unique_indices = set()

    for i, sc_key in enumerate(sc_keys):

        if sc_key=='th':
            years = []
            for sc in [f'th{x}' for x in ('a','b','c','d','e')]:
                try:
                    df_sc = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_{sc}')
                except:
                    print(f'{sc} data not found in directory')
                    continue
                years.append(df_sc['B_avg'].dropna().index.year.to_numpy())
            years = np.concatenate(years)
        else:
            try:
                df_sc = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_{sc_key}')
            except:
                print(f'{sc_key} data not found in directory')
                continue
            years = df_sc['B_avg'].dropna().index.year.to_numpy()

        if len(years)==0:
            continue

        unique_indices.update(df_sc.index)

        if n_rows==1:
            ax = axs
        else:
            ax = axs[i]

        bins = calculate_bins(years,1)
        counts, _ = np.histogram(years, bins=bins)

        label = f'{sc_key}: {len(years):,} {data_type}'

        offset = 0.5
        hatch = None
        if combined:
            offset = (i+0.5)*width
            hatch = bar_hatches[i]

        colour = colour_dict.get(sc_key.upper(),'k')
        edge_colour = to_rgba('k', alpha=0.2)
        ax.bar(bins[:-1]+offset, counts, width=width, color=colour, hatch=hatch, edgecolor=edge_colour, label=label)

        ax.legend(loc='upper right', framealpha=1)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f'{val:,.0f}'))
        ax.set_ylabel(data_type.capitalize())

    print(f'{len(unique_indices):,} unique {data_type} of {region} data')

    if n_rows==1:
        the_ax = axs
    else:
        the_ax = axs[-1]

    the_ax.set_xlabel('Year')
    if kwargs.get('year_range',None):
        the_ax.set_xlim(kwargs['year_range'])

    if n_rows>1:
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.canvas.draw()
        xticks = ax.get_xticks()
        xticks = xticks[(xticks >= ax.get_xlim()[0]) & (xticks <= ax.get_xlim()[1])]
        for ax in axs:
            for x in xticks:
                ax.axvline(x, color=to_rgba(black,0.9), linestyle=':', linewidth=0.5, zorder=20)
    else:
        plt.tight_layout()

    if return_objs:
        return fig, ax

    file_name = '_'.join(sc_keys)+f'_in_{region}'
    if combined:
        file_name += '_combined'
    save_figure(fig, file_name=file_name)
    plt.show()
    plt.close()


def plot_bias_over_years(df_sw, df_msh, sw_col='AE'):

    # Shows driver for full OMNI and for times when contemp. MSH

    fig, ax = plt.subplots(figsize=(10,6), dpi=400, sharex=True)

    for i, (colour, label) in enumerate(zip(('orange','blue'),('All OMNI','Contemp MSH'))):
        mask = np.ones(len(df_sw),dtype=bool)

        if i==1:
            mask = df_msh.index

        monthly_max = df_sw.loc[mask,sw_col].resample('ME').max()
        rolling_max = monthly_max.rolling(window=6, min_periods=1).mean()

        month_datetimes = monthly_max.index
        fractional_years = month_datetimes.year + (month_datetimes.month - 1)/12

        ax.plot(fractional_years, rolling_max.values, linestyle='-', color=colour, label=label)

    y_label = create_label(sw_col, units=df_sw.attrs['units'])

    ax.set_ylabel(y_label)
    ax.legend(loc='upper left')

    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()


def plot_bias_in_parameters(df_sw, df_msh, params=None):

    if params is None:
        parameters_to_plot = ('B_avg','B_z_GSM','V_flow','V_x_GSE','E_R','E_y_GSM','n_p','P_flow','M_A','AE')
    param_width = {'B_avg': 1, 'B_z_GSM': 1, 'V_flow': 50, 'V_x_GSE': 50, 'n_p': 1, 'P_flow': 1, 'MA': 2, 'AE': 50, 'E_mag': 1, 'S_mag': 2}


    n_params = len(parameters_to_plot)
    n_cols = 2
    n_rows = round(n_params/n_cols)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(6*(n_cols+1),4*(n_rows+1)), dpi=400)

    for i, param in enumerate(parameters_to_plot):

        col = i % 2
        row = i // 2

        ax = axs[row,col]

        for j in range(2):
            if j==0:
                mask = np.ones(len(df_sw),dtype=bool)
                colour = 'orange'
                axis = ax
            else:
                mask = df_msh.index
                colour = 'blue'
                axis = ax.twinx()

            series = df_sw.loc[mask,param].dropna()
            if param=='MA':
                series = series.loc[series<200]

            axis.hist(series, bins=calculate_bins(series,param_width.get(param)), histtype='step', edgecolor=colour)
            axis.set_yscale('log')

        x_label = create_label(param, units=df_sw.attrs.get('units',{}))
        ax.set_xlabel(x_label)

    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()
