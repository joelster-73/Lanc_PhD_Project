# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 09:19:33 2026

@author: richarj2
"""

import numpy as np
import matplotlib.pyplot as plt

from .utils import def_param_names, get_variable_range, mask_df

from ....plotting.utils import save_figure
from ....plotting.formatting import create_label, add_figure_title
from ....plotting.config import black

from ....processing.reading import import_processed_data
from ....processing.omni.analysis import calc_omni_uncertainty

from scipy.stats import spearmanr

def annotate_corr(ax, x, y, mask=None):
    """
    Add Spearman rho as text at the top of the axis.
    """
    if mask is not None:
        x, y = x[mask], y[mask]
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    rho, _ = spearmanr(x, y)

    ax.text(0.5, 0.98, r'$\rho$ = ' + f'{rho:.2f}', transform=ax.transAxes, ha='center', va='top', fontsize=8)

def plot_independent_uncertainties(*ind_vars, resolution='1min', spacecraft='omni', region='sw', bounds=None, restrict=True, skip_zero=False, **kwargs):
    """
    Uncertainties on the drivers to see of a significant regression to the mean effect would be observed.
    Three panels: absolute against driver, relative against driver, histogram of relative.
    """
    if spacecraft=='omni':
        df = import_processed_data('omni', resolution=resolution)
        ncols = 4
    else:
        df = import_processed_data(region, dtype='plasma', resolution=resolution, file_name=f'{region}_times_{spacecraft}')
        ncols = 3

    nrows = len(ind_vars)

    def scatter(ax, x, y, limit=True):
        mask = np.ones_like(x, dtype=bool)
        if limit:
            d = 1
            mask = (y < np.percentile(y,100-d))
            # if np.abs(np.min(x)) > np.abs(np.max(x)): # negative parameter
            #     mask &= (x > np.percentile(x,d))
            # else:
            #     mask &= (x < np.percentile(x,100-d))

        ax.scatter(x[mask], y[mask], marker='.', s=0.4, c=black, alpha=0.5, edgecolors='none', linewidths=0)


    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4.5, nrows*3), dpi=800)

    for row, ind_var in enumerate(ind_vars):

        ax_row = axs[row]

        bin_width, limits, invert = get_variable_range(ind_var, region, restrict=restrict, bounds=bounds)
        kwargs['window_width'] = bin_width

        df_ind = mask_df(df, ind_var, limits)
        ind_vals = df_ind[ind_var].to_numpy()

        if spacecraft=='omni':
            ind_err = calc_omni_uncertainty(df_ind, ind_var)
        else:
            ind_err, ind_count = def_param_names(df, ind_var)
            ind_err = df_ind[ind_err].to_numpy()

        if np.all(ind_err==0):
            print('All errors are 0.')
            continue

        ratio = 100*np.divide(ind_err, np.abs(ind_vals), out=np.full_like(ind_err, np.nan, dtype=float), where=np.abs(ind_vals)>=0.25)

        mask = np.isfinite(ratio)

        #-----Panel 1-----#

        scatter(ax_row[0], ind_vals, ind_err)

        annotate_corr(ax_row[0], ind_vals, ind_err)
        ax_row[0].set_xlabel(create_label(ind_var,units=df_ind.attrs['units']))
        ax_row[0].set_ylabel(r'$\sigma$'+f' [{df_ind.attrs["units"][ind_var]}]')

        #-----Panel 2-----#

        scatter(ax_row[1], ind_vals[mask], ratio[mask])

        annotate_corr(ax_row[1], ind_vals, ratio, mask=mask)
        ax_row[1].set_xlabel(create_label(ind_var,units=df_ind.attrs['units']))
        ax_row[1].set_ylabel(r'$\sigma_{rel}$ [%]')

        #-----Panel 3-----#

        bin_width = 10
        bins = np.arange(np.floor(ratio[mask].min()),
                         np.ceil(ratio[mask].max()) + bin_width,
                         bin_width)

        ax_row[2].hist(ratio[mask], bins=bins, color=black)

        ax_row[2].set_xlabel(r'$\sigma_{rel}$ [%]')
        ax_row[2].set_ylabel('Count')

        ax_row[2].set_xlim(0,200)
        ax_row[2].set_yscale('log')

        good = np.sum(ratio[mask]<30)/np.sum(mask)*100

        ax_row[2].axvline(x=30, ls='--', c='r')

        ax_row[2].text(0.5, 0.98, f'{good:.3g} %', transform=ax_row[2].transAxes, ha='center', va='top', fontsize=8)


        if invert:
            ax_row[0].invert_xaxis()
            ax_row[1].invert_xaxis()

        #-----Panel 4-----#

        if spacecraft=='omni':

            scatter(ax_row[3], ind_vals, df_ind['rms_timeshift'].to_numpy())

            ax_row[3].set_xlabel(create_label(ind_var,units=df_ind.attrs['units']))
            ax_row[3].set_ylabel(r'rms(t) [s]')
            annotate_corr(ax_row[3], ind_vals, df_ind['rms_timeshift'].to_numpy())

            if invert:
                ax_row[3].invert_xaxis()

    if ncols>1:
        fig.align_ylabels(axs[:,0])

    title = 'OMNI' if spacecraft=='omni' else f'{region.upper()} spacecraft'
    add_figure_title(fig, title=title)

    file_name = 'uncs_'
    file_name += 'OMNI' if spacecraft=='omni' else f'{spacecraft}_{region.upper()}'
    file_name += f'_{resolution}'

    plt.tight_layout();
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()