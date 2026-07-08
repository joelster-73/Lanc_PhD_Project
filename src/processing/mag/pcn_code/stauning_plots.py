# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:44:09 2026

@author: richarj2
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from .config import DIRECTORIES, PLOT_LABELS_SHORT, PLOT_LABELS_LONG, dark_mode, PLOT_LABELS_SCALED
from .stauning_imports import import_phi, import_ab, import_coeff, import_data
from .stauning_compares import counts_above_levels


def monthly_phi_plot(smooth_R, best_indices, year, month, save_dir):
    """
    In MATLAB, this is done by makerr directly.

    Plot smoothed correlation R and best phi direction ff.
    """
    fig, ax = plt.subplots(figsize=(10,8), dpi=300)

    # Contour plot
    c = ax.contourf(smooth_R, 10, cmap='bwr_r', vmin=-1, vmax=1)

    # Overlay best direction
    ax.plot(best_indices, np.arange(288), c='green', linewidth=2.5)

    # Time ticks every hour
    time_ticks = np.arange(0, 288+1, 12)
    ax.set_yticks(time_ticks)
    ax.set_yticklabels([str(t//12) for t in time_ticks])

    # Phi ticks every 30 degrees
    ax.set_xticks(np.arange(0, 72, 6))  # every 6 indices → 30 degrees
    ax.set_xticklabels([str(i*5 - 180) for i in np.arange(0, 72, 6)])

    ax.set_title(f'Correlation contour ({year}-{month:02d})')
    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Time (hours)')

    plt.colorbar(c, ax=ax, label='R')
    plt.savefig(os.path.join(save_dir, f'{year}_{month:02d}.png'), dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    return fig


def plot_phi(var='2d', source='original', phi_data=None):
    """
    Load Phi data (.npz or .mat) and generate a contour plot.
    var: 'Phi_2d' or 'Phi_year'
    """
    if phi_data is None:
        phi_data = import_phi(var=var, source=source)['phi']

    # For Phi_year, reshape to 12x288 for plotting
    if var == 'year':
        phi_data = phi_data.reshape(-1, 288)

    fig, ax = plt.subplots(figsize=(10,6), dpi=300)
    cb = plt.contourf(phi_data, 12, cmap='rainbow')
    _ = plt.contour(phi_data, 12, colors='black', linewidths=0.5)
    cbar = plt.colorbar(cb)
    cbar.ax.set_ylabel('Phi')

    ax.set_title(f'{var.capitalize()} distribution ({source})')
    ax.set_xlabel('UT hour')
    ax.set_ylabel('Month')

    # Time ticks every hour
    time_ticks = np.arange(0, phi_data.shape[1]+1, 12)
    ax.set_xticks(time_ticks)
    ax.set_xticklabels([str(t//12) for t in time_ticks])

    # Time ticks every hour
    if var == 'year':
        time_ticks = np.arange(0, phi_data.shape[0]+1, phi_data.shape[0]//12)
        ax.set_yticks(time_ticks)
        ax.set_yticklabels([str(t//30) for t in time_ticks])

    if source=='original':
        save_dir = DIRECTORIES.get('phi')
    else:
        save_dir = DIRECTORIES.get(source)
    plt.savefig(os.path.join(save_dir, f'phi_{var}_{source}.png'), dpi=300, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    return fig

def plot_ab(var, source='npz', coeff_data=None):
    """
    Load ab data (.npz or .mat) and generate a contour plot.
    """
    if coeff_data is None:
        coeff_data = import_ab(var, source=source)

    coeff_data = {'a': coeff_data['a'], 'b': coeff_data['b']}
    nrows = len(coeff_data)

    fig, axs = plt.subplots(nrows=nrows, figsize=(10,4), sharex=True, sharey=True, dpi=300)

    if nrows>1:

        for r, ax in enumerate(axs):

            coeff  = list(coeff_data.keys())[r]
            data   = coeff_data[coeff]

            # For Phi_year, reshape to 12x288 for plotting
            if var == 'year':
                data = data.reshape(-1, 288)

            print(data.shape)

            cb = ax.contourf(data, 12, cmap='rainbow')
            _ = ax.contour(data, 12, colors='black', linewidths=0.5)

            # Time ticks every three months
            time_ticks = np.arange(0, data.shape[0]+1, data.shape[0]//4)
            ax.set_yticks(time_ticks)
            if data.shape[0]>12:
                ax.set_yticklabels([str(t//30) for t in time_ticks])
            ax.set_ylabel('Month')
            ax.set_ylim(0, data.shape[0] - 1)

            cbar = plt.colorbar(cb)
            cbar.ax.set_ylabel(PLOT_LABELS_SHORT[coeff])
            cbar.ax.yaxis.set_ticks_position('left')

        # Time ticks every hour
        time_ticks = np.arange(0, data.shape[1]+1, 12)
        axs[-1].set_xticks(time_ticks)
        axs[-1].set_xticklabels([str(t//12) for t in time_ticks])
        axs[-1].set_xlabel('UT hour')
        axs[-1].set_xlim(0, data.shape[1] - 1)

        axs[0].set_title(f'{var.capitalize()} distribution ({source})')

    if source=='original':
        save_dir = DIRECTORIES.get('ab')
    else:
        save_dir = DIRECTORIES.get(source)
    plt.savefig(os.path.join(save_dir, f'ab_{var}_{source}.png'), dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    return fig

def plot_coeff(coeff_data=None, source='original', coeff=None, scale_data=None):
    """
    Load Phi data (.npz or .mat) and generate a contour plot.
    var: 'Phi_2d' or 'Phi_year'
    """
    if coeff_data is None:
        coeff_data = import_coeff(source=source)

    if coeff is not None:
        nrows = 1
    else:
        nrows = len(coeff_data.columns)

    fig, axs = plt.subplots(nrows=nrows, figsize=(10,2*nrows+0.5), sharex=True, sharey=True, dpi=600)

    if nrows==1:
        axs = [axs]

    if coeff is None:
        coeffs = list(coeff_data.columns)
    else:
        coeffs = [coeff]

    plot_label_dict = PLOT_LABELS_SHORT


    for r, ax in enumerate(axs):

        coeff = coeffs[r]
        data = coeff_data[coeff].to_numpy()

        n_days = len(data) // 1440
        data = data[:n_days * 1440]
        daily = data.reshape(n_days, 1440)  # (n_days, 1440)

        if coeff in ('b','covar'):
            daily *= -1

        if scale_data is not None:

            if isinstance(scale_data[coeff],int):
                plot_label_dict = PLOT_LABELS_SHORT

            else:
                plot_label_dict = PLOT_LABELS_SCALED
                scale = scale_data[coeff][:n_days * 1440]
                daily /= scale.reshape(n_days, 1440)
                daily = np.abs(daily*100)

        print(f'{coeff}: mean = {np.mean(daily):.3g}')

        if coeff.endswith(('_unc','var')):
            cmap   = 'plasma'
            cb     = ax.pcolormesh(np.arange(1440), np.arange(n_days), daily, cmap=cmap)
            limits = None

        else:
            cmap      = 'viridis'
            ncontours = 20
            #limits    = PLOT_RANGE_CBAR.get(coeff)
            limits    = None

            if limits:
                levels = np.linspace(limits[0], limits[1], ncontours)
            else:
                levels = ncontours

            cb = ax.contourf(np.arange(1440), np.arange(n_days), daily, levels, cmap=cmap)
            _  = ax.contour(np.arange(1440), np.arange(n_days), daily, levels, colors='black', linewidths=0.5)

        for x in (4,8,12,16,20):
            ax.axvline(60*x, c='grey', ls=':', lw=0.5, alpha=0.5)

        for y in (3,6,9):
            ax.axhline(30*y, c='grey', ls=':', lw=0.5, alpha=0.5)

        ax.set_xticks(np.arange(0, 1440+1, 60))
        ax.set_xticklabels(np.arange(0, 24+1))

        ax.set_yticks(np.arange(0, n_days+1, 90))
        ax.set_yticklabels(np.arange(0, 12+1, 3))

        cbar = plt.colorbar(cb)
        if limits:
            cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10, prune='both'))

        cbar.ax.yaxis.set_ticks_position('left')
        if nrows>1:
            cbar.ax.set_ylabel(plot_label_dict.get(coeff,''))

        ax.set_ylabel('Month')

    axs[-1].set_xlabel('UT hour')

    if nrows==1:
        ax.set_title(PLOT_LABELS_LONG.get(coeff,''))

    plt.tight_layout(h_pad=0)
    plt.show()
    plt.close(fig)

    return fig


def plot_pcn_uncertainty(df, df_original, pcn_col='pcn', unc_col='pcn_unc', source='Reconstructed', source2='Original', threshold=0.5):
    """
    Plot PCN comparison, difference, and uncertainty.
    Panel 1: Reconstructed vs reference PCN with ±1σ band
    Panel 2: Difference with ±1σ band
    Panel 3: Absolute and relative uncertainty (relative masked where |PCN| < threshold)
    """
    if unc_col in df:
        fig, axs = plt.subplots(nrows=3, figsize=(14, 10), sharex=True, dpi=600)
        ax1, ax2, ax3 = axs
    else:
        fig, axs = plt.subplots(nrows=2, figsize=(14, 6), sharex=True, dpi=600)
        ax1, ax2 = axs
    fig.subplots_adjust(hspace=0)

    if dark_mode:
        red    = 'tomato'
        blue   = 'dodgerblue'
        black  = 'w'
        green  = 'limegreen'
        orange = 'orange'
    else:
        red    = 'r'
        blue   = 'b'
        black  = 'k'
        green  = 'g'
        orange = 'darkorange'

    # Panel 1: PCN comparison
    ax1.axhline(0, c='grey', lw=0.5, ls='--')
    ax1.plot(df_original.index, df_original[pcn_col], lw=0.7, c=green, label=source2, alpha=1)
    ax1.plot(df.index, df[pcn_col], lw=0.7, c=blue,  label=source.capitalize().replace('_',' '), alpha=0.75)

    ax1.set_ylabel(r'PCN [$\mathrm{mV\,m^{-1}}$]')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.tick_params(labelbottom=False)

    # Panel 2: Difference
    ax2.axhline(0, c='grey', lw=0.5, ls='--')
    diff = df[pcn_col] - df_original[pcn_col].reindex(df.index)
    diff_smooth = diff.rolling(window=2880, center=True, min_periods=1).mean()
    ax2.plot(df.index, diff_smooth, lw=0.8, c=black, alpha=0.8, label='Smoothed')
    ax2.text(0.01, 0.95, f'$\\langle \\Delta \\rangle$ = {diff.mean():.2g} $\\mathrm{{mV\\,m^{{-1}}}}$',
         transform=ax2.transAxes, fontsize=9, va='top', color=black, alpha=0.9, family='monospace')

    ax2.set_ylabel(r'$\Delta$PCN [$\mathrm{mV\,m^{-1}}$]')
    ax2.legend(fontsize=8, loc='upper right')

    if unc_col in df:
        ax2.tick_params(labelbottom=False)
        #ax2.fill_between(df.index, diff - df[unc_col], diff + df[unc_col], alpha=0.1, color=red)


        # Panel 3: Uncertainty
        abs_unc = df[unc_col]
        rel_unc = np.where(np.abs(df[pcn_col]) > threshold, df[unc_col] / np.abs(df[pcn_col]) * 100, np.nan)
        rel_unc_smooth = pd.Series(rel_unc, index=df.index).rolling(window=2880, center=True, min_periods=1).mean()

        ax4 = ax3.twinx()
        ax3.plot(df.index, abs_unc, lw=0.8, c=red, label=r'Abs. $\sigma$')
        ax4.plot(df.index, rel_unc_smooth, lw=0.8, c=orange, alpha=0.9, label=r'Rel. $\sigma$ (smoothed)')

        ax3.text(0.01, 0.95,
             f'med. abs. = {abs_unc.median():.3g} $\\mathrm{{mV\\,m^{{-1}}}}$  |  '
             f'med. rel. = {np.nanmedian(rel_unc):.3g}%  |  '
             f'max. abs. = {abs_unc.max():.3g} $\\mathrm{{mV\\,m^{{-1}}}}$',
             transform=ax3.transAxes, fontsize=9, va='top', color=black, alpha=0.9, family='monospace')

        ax3.set_ylim(0)
        ax4.set_ylim(0)

        ax3.set_ylabel(r'Abs. $\sigma$ [$\mathrm{mV\,m^{-1}}$]', c=red)
        ax4.set_ylabel(r'Rel. $\sigma$ [%]',   c=orange)

        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax4.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

    fig.align_ylabels(axs)

    #fig.suptitle(f'{list(df.index.year)[0]} ({source})', fontsize=12)
    plt.tight_layout(h_pad=0)
    plt.show()
    plt.close(fig)

    return fig, axs

def plot_yearly_uncertainty(source, years=range(1997, 2024)):
    """
    Plot PCN index and uncertainty according to Stauning intervals.
    Panel 1: PCN index as an indicator of substorm
    Panel 2: Uncertainty as a measure of quality control
    """

    source_dir = DIRECTORIES.get('analysis')
    source_dir = os.path.join(source_dir, source)
    os.makedirs(source_dir, exist_ok=True)

    print(f'\n----------{source}----------\n')

    df_stats = None
    for year in years:
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
    top_colours = ['#fdcc8a', '#fc8d59', '#e34a33', '#b30000']
    bot_colours = ['#fdcc8a', '#fc8d59', '#b30000']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=200, sharex=True)
    fig.subplots_adjust(hspace=0)
    ax1.tick_params(labelbottom=False)

    for ax, cols, colours, name in zip((ax1,ax2), (top_cols,bot_cols), (top_colours,bot_colours), ('PCN','Uncertainty')):

        for i, col in enumerate(cols):
            colour = colours[i]
            ax.plot(df_plot[col], color=colour, marker=markers[i], label=col)
            if i!=3:
                ax.annotate(f'{df_stats[col].iloc[-1]:.1f}%', xy=(df_plot.index[-1], df_plot[col].iloc[-1]), xytext=(0, 5), textcoords='offset points', color=colour, va='bottom', family='monospace')

        ax.legend(loc='upper left',ncols=len(cols))
        ax.set_ylabel(f'{name} # Days')
        ax.set_ylim(top=350)

    plt.show()
    fig.savefig(os.path.join(source_dir,'yearly.png'), dpi=200, bbox_inches='tight')
    plt.close()


def plot_unc_vs_pcn(source, years=range(1997, 2010), deg=2):
    """
    Have to use data from 1997 to 2010, as that is the range over which the coefficients are calculated.
    """
    source_dir = DIRECTORIES.get('analysis')
    source_dir = os.path.join(source_dir, source)
    os.makedirs(source_dir, exist_ok=True)

    df_pcn = pd.concat([import_data('pcn', str(year), source) for year in years])
    threshold = df_pcn['pcn_unc'].quantile(0.999)
    mask = (
        np.isfinite(df_pcn['pcn']) &
        np.isfinite(df_pcn['pcn_unc']) &
        (df_pcn['pcn_unc'] > 0) &
        (df_pcn['pcn_unc'] < threshold)
    )

    fig, ax = plt.subplots(dpi=300)
    ax.scatter(df_pcn.loc[mask,'pcn'], df_pcn.loc[mask,'pcn_unc'], s=0.025, c='k', alpha=0.01, linewidths=0)

    popt = np.polyfit(df_pcn.loc[mask,'pcn'], df_pcn.loc[mask,'pcn_unc']**2, deg)
    err_func = np.poly1d(popt)
    x_range = np.linspace(-30, 30, 200)
    ax.plot(x_range, np.sqrt(err_func(x_range)), c='r', ls=':', lw=0.5)

    residuals = df_pcn.loc[mask,'pcn_unc']**2 - err_func(df_pcn.loc[mask,'pcn'])
    ss_res = np.sum(residuals**2)

    chi2 = ss_res / np.var(df_pcn.loc[mask,'pcn_unc']**2)
    dof = len(df_pcn.loc[mask]) - 3
    chi2_red = chi2 / dof

    ax.text(0.05, 0.95,
            f'$\\sigma^2$ = {poly_to_latex(popt)} $(\\mathrm{{mV\\,m^{{-1}}}})^2$\n'
            f'$\\chi^2_\\nu$ = {chi2_red:.3f}  ($\\nu$ = {dof:,})',
            transform=ax.transAxes, va='top', fontsize=9, family='monospace')

    ax.set_xlabel(r'PCN [$\mathrm{mV\,m^{-1}}$]')
    ax.set_ylabel(r'$\sigma_\mathrm{PCN}$ [$\mathrm{mV \,m^{-1}}$]')

    ax.set_xlim(-10, 10)
    ax.set_ylim(0, int(df_pcn.loc[mask,'pcn_unc'].max())+1)

    plt.show()
    fig.savefig(os.path.join(source_dir,'var_function.png'), dpi=300, bbox_inches='tight')
    plt.close()

def poly_to_latex(popt, ind='x'):
    deg = len(popt) - 1
    terms = []
    for i, coef in enumerate(popt):
        power = deg - i
        abs_coef = abs(coef)
        coef_str = f'{abs_coef:.3f}'
        if power == 0:
            term = coef_str
        elif power == 1:
            term = f'{coef_str}{ind}'
        else:
            term = f'{coef_str}{ind}^{{{power}}}'
        terms.append((coef, term))

    result = ''
    for i, (coef, term) in enumerate(terms):
        if i == 0:
            result += f'-{term}' if coef < 0 else term
        else:
            sign = '-' if coef < 0 else '+'
            result += f' {sign} {term}'

    return f'${result}$'