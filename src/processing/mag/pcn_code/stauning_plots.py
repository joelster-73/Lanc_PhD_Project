# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:44:09 2026

@author: richarj2
"""

import os
import numpy as np

import matplotlib.pyplot as plt

from config import DIRECTORIES, PLOT_LABELS_SHORT
from stauning_imports import import_phi, import_ab

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

    return phi_data

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