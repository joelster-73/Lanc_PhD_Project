# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:53:12 2026

@author: richarj2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import def_param_names, mask_df, get_variable_bin_width

from ....plotting.utils import save_figure, calculate_bins
from ....plotting.formatting import create_label, format_comma_integers
from ....plotting.config import blue, black, white, red, green

from ....processing.reading import import_processed_data
from ....config import INSTRUMENTS, CONSTELLATION, SPACECRAFT, THEMIS_REGIONS

def plot_calibration(df, ind_col, good_col, total_col, buffer):

    independent     = df[ind_col]
    good_percentage = df[good_col]
    total_counts    = df[total_col]

    fig, ax = plt.subplots(figsize=(12,8), dpi=300)

    ax.plot(independent, total_counts, marker='s', c=black)
    ax.set_xlabel(create_label(ind_col, units=df.attrs.get('units',{})))
    ax.set_ylabel('Total Data')
    format_comma_integers(ax, 'y')

    ax2 = ax.twinx()

    ax2.plot(independent, good_percentage, marker='s', c=blue, ls='--')
    ax2.set_ylabel('overlap [%]', color=blue)

    # highest percentage

    position = good_percentage.idxmax()
    param    = independent.loc[position]
    count    = int(total_counts.loc[position])
    percent  = good_percentage.max()

    ax2.scatter(param, percent, s=100, facecolors='none', edgecolors=red, label=f'Highest overlap\n({param:.2f},{percent:.2f} %)\n{count:,}')
    ax2.legend()

    file_name = f'Varying_{ind_col}_{buffer}'

    plt.tight_layout()
    save_figure(fig, file_name=file_name, sub_directory='Calibration', overwrite=True)
    plt.show()
    plt.close()

# %% averaging

def plot_avg_counts(sc, *params, resolution='5min', data_pop='plasma', region='sw', cut_off=25, dfs=None, fig=None, axs=None, return_objs=False):

    if sc in CONSTELLATION:
        spacecraft = CONSTELLATION.get(sc)
    else:
        spacecraft = sc

    if fig is None:
        n_cols = len(params)
        fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(n_cols*6, 3), dpi=800)

    if dfs is None:
        df = import_processed_data(region, dtype=data_pop, resolution=resolution, file_name=f'{region}_times_{sc}')
        dfs = tuple(df for _ in range(len(params)))

    for i, (ax, param, df) in enumerate(zip(axs, params, dfs)):

        if param.startswith('B_'):
            parameter = 'field'
        elif param.startswith('V_'):
            parameter = 'plasma'

        ###----------PLOTTING----------###

        series = df[f'{param}_count'].to_numpy()

        bins = calculate_bins(series,1)
        counts, _ = np.histogram(series, bins=bins)

        maximum = int(bins[np.argmax(counts)])

        if (bins[-1]-bins[0])>1000:

            mask   = counts > 100
            bins   = bins[:-1][mask].astype(int)
            counts = counts[mask]
            ticks  = range(len(counts))

            ax.bar(ticks, counts, width=0.75, color=black, edgecolor=white)
            ax.set_xticks(ticks, labels=bins)
            ax.tick_params(axis='x', which='major', labelsize=9)

        else:

            mask   = counts > 10
            bins   = bins[:-1][mask] # left edge of each remaining bin
            counts = counts[mask]

            if (bins[-1]-bins[0])>100:
                mask = np.ones_like(bins, dtype=bool)
                if maximum - bins[0] > 60:
                    mask &= bins > maximum - 60
                if bins[-1] - maximum > 40:
                    mask &= bins < maximum + 40
                bins = bins[mask]
                counts = counts[mask]

            ax.bar(bins, counts, width=1, color=black, edgecolor=white)

            if (cut_off > ax.get_xlim()[0]):
                ax.axvline(x=cut_off, ls='--', c='r', lw=0.5)

        ###----------FORMATTING----------###

        ax.set_xlabel(r'$N$ for '+f'{create_label(param)}')
        ax.set_yscale('log')

        label = f'$N_{{\\mathrm{{max}}}}$ = {maximum}'
        ax.text(0.03, 0.96, label, transform=ax.transAxes, ha='left', va='top', fontsize=12)

        num_below  = np.sum(series<cut_off)
        frac_below = np.mean(series<cut_off)
        label = f'{frac_below * 100:.1f} %'
        if (frac_below>0.0005):
            ax.text(0.75, 0.96, label, transform=ax.transAxes, ha='left', va='top', fontsize=12, color='r')

        ax.set_title(INSTRUMENTS.get(spacecraft,{}).get(parameter,'').upper())

        ###----------PRINTING----------###

        print(f'{sc} {parameter} data:\n')
        print(f'{len(series):,} rows of {resolution} data.')
        print(f'Count bins range between [{int(bins[0])}, {int(bins[-1])}].\n')
        print(f'Maximum bin count N = {maximum}.')
        print(f'Amount below cut-off N = {cut_off}: {num_below:,} | {frac_below * 100:.1f} %\n\n')


    fig.align_xlabels(axs)
    axs[0].set_ylabel(f'{spacecraft.upper()}')

    if return_objs:

        return fig, axs

    plt.tight_layout()
    file_name = f'Counts_per_{resolution}_avg_{sc}_{region}'
    save_figure(fig, file_name=file_name, sub_directory='Quality', overwrite=True)

    plt.show()
    plt.close()

def plot_avg_counts_grid(*spacecraft, resolution='5min', cut_off=25):
    """
    Combines field and plasma data from each mission into a df each across all probes.
    """

    n_rows = len(spacecraft)
    fig, axs = plt.subplots(nrows=n_rows, ncols=2, figsize=(12, 3*n_rows), dpi=500, squeeze=False)

    for i, (spacecraft, ax_row) in enumerate(zip(spacecraft, axs)): # likely wrong due to squeeze

        field_dfs = []
        plasma_dfs = []

        field_instrument  = INSTRUMENTS.get(spacecraft,{}).get('field')
        plasma_instrument = INSTRUMENTS.get(spacecraft,{}).get('plasma')

        for sc in SPACECRAFT.get(spacecraft,[]):

            df = import_processed_data(sc, field_instrument, resolution)
            field_dfs.append(df)

            if sc in SPACECRAFT.get('mms'):
                df = import_processed_data(sc, plasma_instrument, resolution)
                plasma_dfs.append(df)

            else:
                region_dfs = []
                for region in ('sw','msh'):
                    if spacecraft=='themis' and sc not in THEMIS_REGIONS.get(region):
                        continue
                    df = import_processed_data(sc, region, resolution)
                    region_dfs.append(df)
                df = pd.concat(region_dfs)
                plasma_dfs.append(df)

        field_df = pd.concat(field_dfs)
        plasma_df = pd.concat(plasma_dfs)
        dfs = (field_df, plasma_df)

        _ = plot_avg_counts(spacecraft, 'B_avg', 'V_mag', resolution=resolution, cut_off=cut_off, dfs=dfs, fig=fig, axs=ax_row, return_objs=True)

        del dfs

        for ax in ax_row:
            ax.set_title(None)

        if i!=(n_rows-1):
            for ax in ax_row:
                ax.set_xlabel(None)

    fig.align_ylabels(axs[:,0])
    plt.tight_layout()
    save_figure(fig, file_name=f'Counts_each_{resolution}_average', sub_directory='Quality', overwrite=True)

    plt.show()
    plt.close()


# %% distributions

def plot_avg_distributions(spacecraft, *params, data_pop='plasma', resolutions=('1min','5min'), regions=('sw','msh'), restrict=False, display='lines'):

    suffix = ''
    if params==('data',):
        params = 'V_x_GSE', 'V_y_GSM', 'V_z_GSM', 'N_tot', 'B_x_GSE', 'B_y_GSM', 'B_z_GSM'
        suffix = '_data'
    elif params==('drivers',):
        params = 'B_z_GSM', 'E_yz', 'E_R', 'S_x_GSE'
        suffix = '_drivers'

    n_rows = len(params)
    n_cols = len(regions)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12*n_cols, 4*(n_rows+1)), dpi=500, squeeze=False)

    if isinstance(spacecraft, tuple) and len(spacecraft)==len(regions): # iterate through spacecraft and regions
        enumerator = zip(spacecraft, regions)
        iterable = 'both'
    else:
        enumerator = regions
        iterable = 'region'

    for i, enum in enumerate(enumerator): # column

        if iterable=='region':
            region = enum
            sc = spacecraft

        elif iterable =='both':
            sc, region = enum

        region_colour = {'sw': blue, 'msh': red, 'msh20': green}.get(region)

        for j, (resolution, colour, alpha) in enumerate(zip(resolutions,(black,region_colour),(0.7,0.8))): # dataset

            df = import_processed_data(region, dtype=data_pop, resolution=resolution, file_name=f'{region}_times_{sc}')

            for k, param in enumerate(params):

                ax = axs[k][i]

                ###----------COUNTS----------###

                _, ind_count = def_param_names(df, param)
                series = mask_df(df, param, count_col=ind_count, resolution=resolution)[param] # remove avgs with low sample size

                if restrict:
                    lower, upper = series.quantile([0.005, 0.995]) # 0.5% and 99.5%
                    series = series[(series>=lower) & (series<=upper)]

                bin_width = get_variable_bin_width(series, -1)

                bins = calculate_bins(series, bin_width)
                counts, _ = np.histogram(series, bins=bins)

                mask = counts > 5
                bins = bins[:-1][mask]

                ###----------STYLE----------###

                if display == 'bars':
                    bins += bin_width/2 # left edge is aligned with ticks

                counts = counts[mask]

                if display == 'bars':
                    ax.bar(bins, counts, width=bin_width, color=colour, alpha=alpha, label=resolution)
                elif display == 'lines':
                    ax.plot(bins, counts, color=colour, lw=3, label=resolution)
                    ax.axvline(np.median(series), color=colour, lw=3, ls='--')

                ax.set_xlabel(create_label(param, units=df.attrs.get('units')))
                ax.set_yscale('log')

                if k==0:
                    title = region if sc=='combined' else f'{sc} {region}'
                    ax.legend(title=title)

                del series, counts, bins

    ###----------FORMATTING----------###

    fig.align_ylabels(axs[:,0])

    plt.tight_layout()
    file_name  = 'Compare_distributions_'
    file_name += ('_'.join(spacecraft) + '_') if iterable=='both' else f'_{spacecraft}_'
    file_name += '_'.join(resolutions) + suffix
    file_name += '_restricted' if restrict else ''
    file_name += f'_{display}' if display != 'lines' else ''
    save_figure(fig, file_name=file_name, sub_directory='Quality', overwrite=True)

    plt.show()
    plt.close()

