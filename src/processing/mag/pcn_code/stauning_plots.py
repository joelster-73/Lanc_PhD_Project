# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:44:09 2026

@author: richarj2
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from config import DIRECTORIES, PLOT_LABELS_SHORT, LIST_OF_MONTHS, PLOT_LABELS_LONG
from stauning_imports import import_phi, import_ab, import_coeff

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
        phi_data = import_phi(var=var, source=source)

    # For Phi_year, reshape to 12x288 for plotting
    if var == 'year':
        phi_data = phi_data.reshape(-1, 288)

    fig, ax = plt.subplots(figsize=(10,6), dpi=300)
    cb = plt.contourf(phi_data, 12, cmap='rainbow')
    _ = plt.contour(phi_data, 12, cs='black', linewidths=0.5)
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
            _ = ax.contour(data, 12, cs='black', linewidths=0.5)

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

def plot_coeff(coeff_data=None, source='npz', coeff=None):
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

    fig, axs = plt.subplots(nrows=nrows, figsize=(10,2*nrows+0.5), sharex=True, sharey=True, dpi=300)

    if nrows==1:
        axs = [axs]

    for r, ax in enumerate(axs):

        if coeff is None:
            coeff = list(coeff_data.columns)[r]
        data = coeff_data[coeff].to_numpy()

        n_days = len(data) // 1440
        data = data[:n_days * 1440]
        daily = data.reshape(n_days, 1440)  # (n_days, 1440)

        if coeff.endswith('_unc'):
            cmap = 'plasma'
            cb = ax.pcolormesh(np.arange(1440), np.arange(n_days), daily, cmap=cmap)
        else:
            cmap = 'turbo'

            ncontours = 20
            cb = ax.contourf(np.arange(1440), np.arange(n_days), daily, ncontours, cmap=cmap)
            _ = ax.contour(np.arange(1440), np.arange(n_days), daily, ncontours, colors='black', linewidths=0.5)

        ax.set_xticks(np.arange(0, 1440+1, 60))
        ax.set_xticklabels(np.arange(0, 24+1))

        ax.set_yticks(np.arange(0, n_days+1, 90))
        ax.set_yticklabels(np.arange(0, 12+1, 3))

        cbar = plt.colorbar(cb)
        if nrows>1:
            cbar.ax.yaxis.set_ticks_position('left')
            cbar.ax.set_ylabel(PLOT_LABELS_SHORT.get(coeff,''))

        ax.set_ylabel('Month')

    axs[-1].set_xlabel('UT hour')

    if nrows==1:
        ax.set_title(PLOT_LABELS_LONG.get(coeff,''))

   # plt.title(f'{source.upper()}')

    plt.tight_layout()
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
        fig, axs = plt.subplots(nrows=3, figsize=(14, 10), sharex=True, dpi=500)
        ax1, ax2, ax3 = axs
    else:
        fig, axs = plt.subplots(nrows=2, figsize=(14, 6), sharex=True, dpi=500)
        ax1, ax2 = axs
    fig.subplots_adjust(hspace=0)

    # Panel 1: PCN comparison
    ax1.plot(df_original.index, df_original[pcn_col], lw=0.8, c='tomato', label=source2, alpha=1)
    ax1.plot(df.index, df[pcn_col], lw=0.8, c='dodgerblue',  label=source, alpha=0.8)

    ax1.set_ylabel('PCN [mV/m]')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.tick_params(labelbottom=False)

    # Panel 2: Difference
    diff = df[pcn_col] - df_original[pcn_col].reindex(df.index)
    ax2.plot(df.index, diff, lw=0.8, c='w', alpha=0.8)
    ax2.axhline(0, c='tomato', lw=0.5, ls='--')
    ax2.text(0.01, 0.95, f'$\\langle \\Delta \\rangle$ = {diff.mean():.2g} mV/m',
         transform=ax2.transAxes, fontsize=9, va='top', color='white', alpha=0.9)

    ax2.set_ylabel(r'$\Delta$PCN [mV/m]')

    if unc_col in df:
        ax2.tick_params(labelbottom=False)
        ax2.fill_between(df.index, diff - df[unc_col], diff + df[unc_col], alpha=0.2, color='grey')


        # Panel 3: Uncertainty
        abs_unc = df[unc_col]
        rel_unc = np.where(np.abs(df[pcn_col]) > threshold, df[unc_col] / np.abs(df[pcn_col]) * 100, np.nan)
        rel_unc_smooth = pd.Series(rel_unc, index=df.index).rolling(window=2880, center=True, min_periods=1).mean()

        ax4 = ax3.twinx()
        ax3.plot(df.index, abs_unc, lw=0.8, c='khaki', label=r'Abs $\sigma$')
        ax4.plot(df.index, rel_unc_smooth, lw=0.8, c='hotpink', alpha=0.9, label=r'Rel $\sigma$')

        ax3.text(0.01, 0.95,
             f'med abs = {abs_unc.median():.3g} mV/m  |  '
             f'med rel = {np.nanmedian(rel_unc):.3g}%  |  '
             f'max abs = {abs_unc.max():.3g} mV/m',
             transform=ax3.transAxes, fontsize=9, va='top', color='white', alpha=0.9)

        ax3.set_ylim(0)
        ax4.set_ylim(0)

        ax3.set_ylabel(r'Abs $\sigma$ [mV/m]', c='khaki')
        ax4.set_ylabel(r'Rel $\sigma$ (%) - smoothed',   c='hotpink')

        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax4.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

    fig.suptitle(f'{list(df.index.year)[0]} ({source})', fontsize=12)
    plt.tight_layout(h_pad=0)
    plt.show()
    plt.close(fig)

    return fig, axs

# %% Prints


def print_coeffs_monthly_ut(df):

    df_2d = df.copy()

    index_names = df.index.names

    time_vals = df.index.get_level_values('time')
    hours = time_vals.map(lambda t: t.hour)

    if 'month' in index_names:
        months = df.index.get_level_values('month')

    elif 'doy' in index_names:
        doy = df.index.get_level_values('doy')
        months = doy.map(lambda d: pd.Timestamp('2000-01-01') + pd.Timedelta(days=int(d)-1)).month

    df_2d = df.iloc[:, 0].groupby([months, hours]).mean().unstack(level=0)
    df_2d.index.name   = 'hour'
    df_2d.columns.name = 'month'


    df_2d.columns = LIST_OF_MONTHS


    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_2d)

    return df_2d
