# -*- coding: utf-8 -*-
'''
Created on Sat Feb 28 14:22:13 2026

@author: richarj2
'''

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from config import DIRECTORIES
from stauning_imports import import_data, import_true_pcn
from stauning_plots import print_coeffs_monthly_ut, plot_coeff, plot_pcn_uncertainty
from stauning_compares import counts_above_levels

def coefficients_overview(source):

    print(f'----------{source}----------\n')

    source_dir = DIRECTORIES.get('analysis')
    source_dir = os.path.join(source_dir, source)
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(os.path.join(source_dir,'years'), exist_ok=True)

    df = import_data('coeff', '', source=source)

    for column in df:

        if column.endswith('covar'):
            continue

        print(f'-----{column}-----')

        df_column = df[[column]]

        if column.endswith('_var'):

            df[column.replace('_var','_unc')] = np.sqrt(df_column)

            df_column = df[[column.replace('_var','_unc')]]

        df_2d = print_coeffs_monthly_ut(df_column)
        df_2d.to_csv(os.path.join(source_dir,f'{column}.txt'), sep='\t', index=True, float_format='%.3g')

        fig = plot_coeff(coeff_data=df_column)
        fig.savefig(os.path.join(source_dir,f'{column}.png'), dpi=300, bbox_inches='tight')


    source2='Original'
    if source=='original':
        return
        source2 = 'true'
        pcn = import_true_pcn()

    for year in range(1997,2022):
        year = str(year)

        df_pcn = import_data('pcn', year, source)

        if source=='original':
            df_original = pcn.loc[pcn.index.year==int(year)]
        else:
            df_original = import_data('pcn', year, 'original')

        fig, _ = plot_pcn_uncertainty(df_pcn, df_original, source=source, source2=source2)
        fig.savefig(os.path.join(source_dir,'years',f'{year}.png'), dpi=300, bbox_inches='tight')

def total_uncertainty(source):

    print(f'\n----------{source}----------\n')

    df_stats = None
    for year in range(1997, 2022):
        df_pcn = import_data('pcn', str(year), source)
        df_stats = counts_above_levels(df_pcn, df_stats)

    # totals row
    totals = df_stats.sum(numeric_only=True).to_frame().T
    totals.index = ['Total']
    df_stats = pd.concat([df_stats, totals])

    # percentage row (excludes the totals row from denominator)
    n_years = len(df_stats) - 1
    pct = (100 * df_stats.loc['Total'] / (n_years * 365.25)).to_frame().T
    pct.index = ['%']
    df_stats = pd.concat([df_stats, pct])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_stats.round(1))

    df_plot = df_stats.iloc[:-2]

    top_cols = ['Quiet', 'Moderate', 'Strong', 'Severe']
    bot_cols = ['Small', 'Trouble', 'Large']

    markers = ['+','s','x','o']
    cmap = plt.colormaps['Reds'].resampled(max(len(bot_cols),len(top_cols)) + 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=200, sharex=True)
    fig.subplots_adjust(hspace=0)
    ax1.tick_params(labelbottom=False)

    for ax, cols, name in zip((ax1,ax2),(top_cols,bot_cols), ('PCN','Unc')):

        for i, col in enumerate(cols):
            colour = cmap(i)
            ax.plot(df_plot[col], color=colour, marker=markers[i], label=col)
            if i!=3:
                ax.annotate(f'{df_stats[col].iloc[-1]:.1f}%', xy=(df_plot.index[-1], df_plot[col].iloc[-1]), xytext=(0, 5), textcoords='offset points', color=colour, va='bottom')

        ax.legend(loc='upper left',ncols=len(cols))
        ax.set_ylabel(f'{name} # Days')

    plt.show()


if __name__ == '__main__':


    if False:
        coefficients_overview('original')
        coefficients_overview('staun_proj')
        coefficients_overview('staun_phi')
        coefficients_overview('recreated_phi')

    total_uncertainty('staun_phi')
    total_uncertainty('recreated_phi')


