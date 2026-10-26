# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:53:12 2026

@author: richarj2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import def_param_names, mask_df


from ....plotting.utils import save_figure, calculate_bins
from ....plotting.formatting import create_label, format_comma_integers
from ....plotting.config import blue, black, white, red, green

from ....processing.reading import import_processed_data
from ....config import INSTRUMENTS, CONSTELLATION, SPACECRAFT, THEMIS_REGIONS

def plot_calibration(df, ind_col, good_col, total_col):

    independent     = df[ind_col]
    good_percentage = df[good_col]
    total_counts    = df[total_col]

    fig, ax = plt.subplots(figsize=(12,8), dpi=300)

    ax.plot(independent, total_counts, c=black)
    ax.set_xlabel(create_label(ind_col, units=df.attrs.get('units',{})))
    ax.set_ylabel('Total Data')
    format_comma_integers(ax, 'y')

    ax2 = ax.twinx()

    ax2.plot(independent, good_percentage, c=blue, ls='--')
    ax2.set_ylabel('overlap [%]', color=blue)


    file_name = f'Varying_{ind_col}'

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

    # change so doesn't import filtered data but instrument data
    # and if the data is separated sw/msh, combine
    # should really combined spacecraft into missions

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

        series = df[f'{param}_count'].copy()

        bins = calculate_bins(series,1)
        counts, _ = np.histogram(series, bins=bins)

        maximum = int(bins[np.argmax(counts)])

        if (bins[-1]-bins[0])>100:
            mask = ((bins[:-1] >= max(maximum-50,np.min(bins))) & (bins[:-1] <= min(maximum+50,np.max(bins))))
            bins = bins[:-1][mask]    # left edge of each remaining bin
            counts = counts[mask]
            bins += 0.5

        else:
            mask = counts>1
            bins = bins[:-1][mask] # left edge
            counts = counts[mask]

        ax.bar(bins, counts, width=1, color=black, edgecolor=white)
        ax.set_xlabel(r'$N$ for '+f'{create_label(param)}')
        ax.set_yscale('log')

        if ax.get_xlim()[0] < cut_off < ax.get_xlim()[1]:
            ax.axvline(x=cut_off, ls='--', c='r', lw=0.5)

        label = f'$N_{{\\mathrm{{max}}}}$ = {maximum}'
        ax.text(0.03, 0.96, label, transform=ax.transAxes, ha='left', va='top', fontsize=12)

        frac_below = (series<cut_off).mean() * 100
        label = f'{frac_below:.1f} %'
        ax.text(0.3, 0.96, label, transform=ax.transAxes, ha='left', va='top', fontsize=12, color='r')

        ax.set_title(INSTRUMENTS.get(spacecraft,{}).get(parameter,'').upper())

    axs[0].set_ylabel(f'{spacecraft.capitalize()} Counts')

    if return_objs:

        return fig, axs

    plt.tight_layout()
    save_figure(fig, file_name=f'Counts_per_{resolution}_avg_{sc}_{region}', overwrite=True)

    plt.show()
    plt.close()

def plot_avg_counts_grid(*spacecraft, resolution='5min', cut_off=25):

    n_rows = len(spacecraft)
    fig, axs = plt.subplots(nrows=n_rows, ncols=2, figsize=(12, 3*n_rows), dpi=500, squeeze=False)

    for i, (spacecraft, ax_row) in enumerate(zip(spacecraft, axs)): # likely wrong due to squeeze

        field_dfs = []
        plasma_dfs = []

        field_instrument  = INSTRUMENTS.get(spacecraft,{}).get('field')
        plasma_instrument = INSTRUMENTS.get(spacecraft,{}).get('plasma')

        for sc in SPACECRAFT.get(spacecraft,[]):
            print(sc)
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

        _ = plot_avg_counts(spacecraft, 'B_avg', 'V_flow', resolution=resolution, cut_off=cut_off, dfs=dfs, fig=fig, axs=ax_row, return_objs=True)

        for ax in ax_row:
            ax.set_title(None)

        if i!=(n_rows-1):
            for ax in ax_row:
                ax.set_xlabel(None)

    fig.align_ylabels(axs[:,0])
    plt.tight_layout()
    save_figure(fig, file_name=f'Counts_each_{resolution}_average', overwrite=True)

    plt.show()
    plt.close()


# %% distributions

def plot_avg_distributions(sc, *params, data_pop='plasma', resolutions=('1min','5min'), regions=('sw','msh'), restrict=False):

    n_rows = len(params)
    n_cols = len(regions)

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12*n_cols, 4*(n_rows+1)), dpi=800)

    for i, region in enumerate(regions): # column

        region_colour = {'sw': blue, 'msh': red, 'msh20': green}.get(region)
        for j, (resolution, colour, alpha) in enumerate(zip(resolutions,(black,region_colour),(0.7,0.8))): # dataset

            df = import_processed_data(region, dtype=data_pop, resolution=resolution, file_name=f'{region}_times_{sc}')

            for k, param in enumerate(params):

                ax = axs[k][i]

                _, ind_count = def_param_names(df, param)
                df_ind = mask_df(df, param, count_col=ind_count, resolution=resolution) # remove avgs with low sample size
                series = df_ind[param]

                if restrict:
                    lower, upper = series.quantile([0.005, 0.995]) # 0.5% and 99.5%
                    series = series[(series>=lower) & (series<=upper)]

                data_range = (series.max()-series.min())
                if data_range > 400:
                    bin_width = 10
                elif data_range > 250:
                    bin_width = 5
                elif data_range > 50:
                    bin_width = 2
                else:
                    bin_width = 1

                bins = calculate_bins(series, bin_width)
                counts, _ = np.histogram(series, bins=bins)

                mask = counts > 10
                bins = bins[:-1][mask] # left edge
                counts = counts[mask]

                ax.bar(bins, counts, width=bin_width, color=colour, alpha=alpha, label=resolution)
                ax.set_xlabel(create_label(param, units=df.attrs.get('units')))
                ax.set_yscale('log')

                if k==0:
                    ax.legend(title=region)

    fig.align_ylabels(axs[:,0])

    plt.tight_layout()
    filename = f'Compare_distributions_{sc}_'
    filename += '_'.join(resolutions)
    filename += '_restricted' if restrict else ''
    save_figure(fig, file_name=filename, overwrite=True)

    plt.show()
    plt.close()

