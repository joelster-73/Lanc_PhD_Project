# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 10:55:08 2025

@author: richarj2
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgba

from ....processing.reading import import_processed_data

from ....plotting.space_time import plot_orbit_msh
from ....plotting.utils import save_figure, calculate_bins
from ....plotting.formatting import add_figure_title
from ....plotting.config import colour_dict, black, bar_hatches, colour_dict_simple
#from ...plotting.distributions import plot_fit

from ....config import INSTRUMENTS, CONSTELLATION

# %% space

def plot_sc_orbits(sample_interval='1min', data_pop='plasma', region='msh', sc_keys=None, **kwargs):

    data_type = 'mins' if sample_interval == '1min' else 'counts'
    column_check = kwargs.get('column_check','B_avg') # column check non-nan data for counts

    if sc_keys is None:
        sc_keys = ('c1','mms1','tha','thc','thd','the')

    n_cols = len(sc_keys)
    n_rows = round(len(sc_keys)/n_cols)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3.5*(n_cols+0.5),6*(n_rows+1)), dpi=400)

    for i, sc_key in enumerate(sc_keys):

        df_sc = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_{sc_key}.cdf')

        row = i % n_rows
        col = i // n_rows

        if len(sc_keys)==1:
            ax = axs
        elif n_rows==1:
            ax = axs[col]
        else:
            ax = axs[row,col]

        length = len(df_sc[column_check].dropna())
        title = f'{sc_key}: {length:,} {data_type}'

        _, _, cbar, _ = plot_orbit_msh(df_sc, sc_keys=None, title=title, region=region, fig=fig, ax=ax, return_objs=True)

        if col!=0:
            ax.set_ylabel(None)
        if row!=(n_rows-1) and n_rows!=1:
            ax.set_xlabel(None)

    file_name = f'orbits_{region}_{data_pop}_{sample_interval}_'
    file_name += '_'.join(sc_keys)

    plt.tight_layout()
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()

def plot_sc_sw_msh(sample_interval='1min', data_pop='plasma', sw_keys=None, msh_keys=None, **kwargs):

    if sw_keys is None:
        sw_keys = ('c1','mms1','thb')

    if msh_keys is None:
        msh_keys = ('c1','mms1','the')

    fig, axs = plt.subplots(2, 1, figsize=(12,8), dpi=400, sharex=True)

    for ax, region, keys, label in zip(axs,  ('sw','msh'), (sw_keys,msh_keys), ('Solar Wind','Magnetosheath')):
        plot_sc_years(sample_interval, data_pop, region, keys, combined=True, fig=fig, ax=ax, return_objs=True, **kwargs)
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

    file_name = f'sw_msh_sc_combined_{data_pop}_{sample_interval}'

    plt.tight_layout()
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()

# %% time

def plot_sc_years(sample_interval='1min', data_pop='plasma', region='msh', sc_keys=None, combined=True, **kwargs):

    """
    Combined flag: show all years on one axis, rather than split per spacecraft
    """

    data_type = 'mins' if sample_interval == '1min' else 'counts'
    column_check = kwargs.get('column_check','r_x_GSE') # column check non-nan data for counts

    fig          = kwargs.get('fig',None)
    axs          = kwargs.get('ax',None)
    return_objs  = kwargs.get('return_objs',False)

    if sc_keys is None:
        sc_keys = ('c1','mms1','tha','thb','thc','thd','the')
        if combined:
            sc_keys = ('c1','mms1','th')

    n_rows = 1 if combined else len(sc_keys)
    n_cols = 1
    width  = 1/len(sc_keys) if combined else 1

    dh = 1.5 if n_rows==1 else 1
    fig_h  = 2*(n_rows+dh)
    fig_w  = 4.5*(n_cols+1)


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
                years.append(df_sc[column_check].dropna().index.year.to_numpy())
            years = np.concatenate(years)
        else:
            try:
                df_sc = import_processed_data(region, dtype=data_pop, resolution=sample_interval, file_name=f'{region}_times_{sc_key}')
            except:
                print(f'{sc_key} data not found in directory')
                continue
            years = df_sc[column_check].dropna().index.year.to_numpy()

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

    the_ax = axs if n_rows==1 else axs[-1]

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
        return fig, axs

    first_ax = axs if n_rows==1 else axs[0]

    regions = {'sw': 'Solar Wind', 'msh': 'Magnetosheath'}
    add_figure_title(fig, title=regions.get(region,''),ax=first_ax)

    file_name = '_'.join(sc_keys)+f'_in_{region}'
    if combined:
        file_name += '_combined'
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()

def plot_data_inventory(*spacecraft, region='msh', display='yearly', **kwargs):

    row = 0
    row_sep = 0.4
    row_labels = []

    rows = len(spacecraft) * (2 if region=='all' else 1)
    width = 8 * (2 if region=='all' else 1)
    height = (2*rows + 1)*row_sep if region=='all' else 10*row_sep
    ys = np.arange(0, rows*row_sep-0.1, row_sep)

    field_col = 'B_avg'
    plasma_col = 'V_flow'

    def _draw_bar(ax, row, start, end, group, colour, row_sep, alpha_min=0.2, bar_scale=0.75):
        alpha = len(group) / (end - start).total_seconds() * 60
        alpha = alpha * (1 - alpha_min) + alpha_min
        ax.barh(row, end - start, left=start, height=row_sep*bar_scale, color=colour, alpha=alpha)


    fig, ax = plt.subplots(figsize=(width, height))

    for sc in spacecraft:
        print(sc)

        if region=='all': # all spacecraft before region filtering

            constellation = CONSTELLATION.get(sc)
            instruments = INSTRUMENTS.get(constellation)

            field = instruments.get('field')
            plasma = instruments.get('plasma')

            print(field,plasma)

            df_field = import_processed_data(sc, dtype=field, resolution='1min')
            df_plasma = import_processed_data(sc, dtype=plasma, resolution='spin')

            df_field = df_field.loc[df_field[field_col].notna(),[field_col]]
            df_plasma = df_plasma.loc[df_plasma[plasma_col].notna(),[plasma_col]]

            plasma_times = df_plasma.index.floor('min').drop_duplicates()

            times_dict = {f'{sc} {field.lower()}': pd.to_datetime(df_field.index), f'{sc} {plasma.lower()}': pd.to_datetime(plasma_times)}

        else:

            df = import_processed_data(region, dtype='plasma', resolution='1min', file_name=f'{region}_times_{sc}', )

            times_dict = {sc: pd.to_datetime(df.index)}

        colour = colour_dict_simple.get(sc.upper(),black)

        for label, times in times_dict.items():

            times = pd.DatetimeIndex(times).floor('min').unique()

            if display=='scatter':

                ax.scatter(times, np.full(len(times), row), marker='|', s=300, linewidths=0.5, facecolors=colour, alpha=0.1)

            elif display == 'yearly':
                times = pd.Series(times)
                time_groups = times.groupby(times.dt.year)
                for year, group in time_groups:
                    start = pd.Timestamp(f'{year}-01-01')
                    end = start + pd.DateOffset(years=1)
                    _draw_bar(ax, row, start, end, group, colour, row_sep)

            elif display == 'monthly':
                times = pd.Series(times)
                time_groups = times.groupby([times.dt.year, times.dt.month])
                for (year, month), group in time_groups:
                    start = pd.Timestamp(f'{year}-{month:02d}-01')
                    end = start + pd.DateOffset(months=1)
                    _draw_bar(ax, row, start, end, group, colour, row_sep)

            else:
                continue

            row_labels.append(label)
            row += row_sep

    ax.set_ylim(-row_sep, rows*row_sep)
    ax.invert_yaxis()
    ax.set_yticks(ys)
    ax.set_yticklabels(row_labels)

    if region=='all':
        # left-align all ticklabels
        for ticklabel in ax.get_yticklabels():
            ticklabel.set_horizontalalignment('left')
        ax.tick_params('y', pad=90)

    ax.set_title(region)

    plt.tight_layout()
    save_figure(fig, file_name=f'Data_Inventory_{region}', overwrite=True)

    plt.show()
    plt.close()
