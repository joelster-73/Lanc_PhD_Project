# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:36:58 2026

@author: richarj2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from .utils import def_param_names, get_variable_range, mask_df
from ..delay import merge_with_lag, get_response_delay

from ....plotting.utils import save_figure, calculate_bins
from ....plotting.formatting import create_label
from ....plotting.config import black, blue, red, green

from ....processing.reading import import_processed_data
from ....processing.mag.indices import import_processed_index
from ....processing.omni.analysis import calc_omni_uncertainty

def binned_band(ax, x, y, nbins=40, colour=black):
    """
    Median and 5th-95th percentile band of y in binned x.
    """
    edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
    idx = np.digitize(x, edges) - 1
    cen, lo, med, hi = [], [], [], []
    for i in range(nbins):
        yi = y[idx == i]
        if len(yi)>5:
            q = np.percentile(yi, [5, 50, 95])
            cen.append(0.5 * (edges[i] + edges[i + 1]))
            lo.append(q[0]); med.append(q[1]); hi.append(q[2])
    ax.fill_between(cen, lo, hi,  color=colour, alpha=0.25, lw=0)
    ax.plot(cen, med, color=colour, lw=1)

def driver_uncertainty_stats(x, s, rel_floor=0.25, rel_cut=30):
    """
    Summary of averaging uncertainty s on driver values x.
    """
    # relative uncertainty (%), only where |x| is not tiny (as in the original figure)
    big = np.abs(x) >= rel_floor
    rel = 100 * s[big] / np.abs(x[big])

    return {'sigma (p50)':        np.median(s),
            'sigma (p50) %':  np.median(rel),
            'sigma (p95) %':     np.percentile(rel, 95),
            f'Amount >{rel_cut}%': 100 * np.mean(rel > rel_cut)}

def plot_independent_uncertainties(*ind_vars, resolution='5min', spacecraft='combined', regions=('sw','msh'), bounds=None, restrict=True, **kwargs):
    """
    Absolute uncertainty on each driver against the driver (one panel per driver, one row), and a table of <sigma^2>/var(x) as a measure of the regression dilution from the measurement uncertainty.

    These results are to justify whether there is evidence of regression to the mean bias as a cause of saturation.
    """
    if spacecraft=='omni':
        regions = ('sw',)

    ncols = len(ind_vars)
    nrows = len(regions)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows), dpi=300, squeeze=False)

    table_rows = []

    for i, region in enumerate(regions):

        if spacecraft=='omni':
            df = import_processed_data('omni', resolution=resolution)
            colour = 'orange'
        else:
            df = import_processed_data(region, dtype='plasma', resolution=resolution, file_name=f'{region}_times_{spacecraft}')
            colour = {'sw': blue, 'msh': red, 'msh20': green}.get(region)

        dataset = 'OMNI' if spacecraft == 'omni' else f'{region.upper()}'

        for j, ind_var in enumerate(ind_vars):

            ax = axs[i][j]

            ###-----INDEPENDENT DATA-----###
            _, limits, invert = get_variable_range(ind_var, region, restrict=restrict, bounds=bounds)

            ind_err, ind_count = def_param_names(df, ind_var)

            if ind_var in ('E_yz','S_x_GSE'): # restrict to -ve Bz

                condition = '<0'
                cond_col  = 'B_z_GSM'

            else:
                condition, cond_col = None, None

            df_ind = mask_df(df, ind_var, limits, ind_count, resolution, condition=condition, cond_col=cond_col, invert=invert)

            if invert:
                ind_var = f'-{ind_var}'

            ind_vals = df_ind[ind_var].to_numpy()

            ###-----ERROR CALCULATIONS-----###

            if spacecraft == 'omni':
                ind_err = calc_omni_uncertainty(df_ind, ind_var)
            else:
                ind_err = df_ind[ind_err].to_numpy()

            if np.all(ind_err == 0):
                print(f'All errors are 0 for {ind_var}.')
                continue

            mask = np.isfinite(ind_vals) & np.isfinite(ind_err)

            xs, errs = ind_vals[mask], ind_err[mask]
            ratio = np.mean(errs**2) / np.var(xs)

            ###-----PLOTTING-----###

            binned_band(ax, xs, errs, colour=colour)

            if j==0:
                ax.text(0.05, 0.95, region, transform=ax.transAxes, ha='left', va='top', fontsize=8)

            # eta = $\langle\sigma^2\rangle/\mathrm{{V}}[x]$
            ax.text(0.5, 0.95, rf'$\eta$ = {ratio:.2g}' '\n' rf'$\rho$ = {spearmanr(xs, errs)[0]:.3g}', transform=ax.transAxes, ha='center', va='top', fontsize=8)

            if i==(nrows-1):
                ax.set_xlabel(create_label(ind_var, units=df_ind.attrs['units']))
            ax.set_ylabel(create_label('sigma', unit=df_ind.attrs['units'][ind_var]))

            stats = driver_uncertainty_stats(xs, errs)
            table_rows.append({'dataset': dataset, 'driver': ind_var, '<sigma2>/V[x]': ratio, **stats})

    file_name = 'uncs_'
    file_name += 'OMNI' if spacecraft == 'omni' else f'{spacecraft}'
    file_name += f'_{resolution}'

    plt.tight_layout()
    save_figure(fig, file_name=file_name, overwrite=True)
    plt.show()
    plt.close()

    table = pd.DataFrame(table_rows)
    print(table.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    return table


def table_response_uncertainties(ind_vars, dep_vars, spacecraft='combined', region='sw', resolution='5min', ref=0.01, file_name=None):
    """
    Measurement variance of each response vs the scatter of that response, in bins of each driver.

    These results are to justify an unweighted fit in the main section of the work.
    """
    df_sc = import_processed_data(region, dtype='plasma', resolution=resolution, file_name=f'{region}_times_{spacecraft}')

    rows = []
    full_data = {}

    for dep_var in dep_vars:

        ###-----DEPENDENT DATA-----###

        df_pc = import_processed_index(dep_var, resolution=resolution, return_series=False, return_full=True)
        df_pc = mask_df(df_pc, dep_var)

        lag   = get_response_delay(dep_var)

        dep_err, _ = def_param_names(df_pc, dep_var)

        for ind_var in ind_vars:

            ###-----INDEPENDENT DATA-----###

            bin_width, _, invert = get_variable_range(ind_var, region)

            ind_err, ind_count = def_param_names(df_sc, ind_var)

            if ind_var in ('E_yz','S_x_GSE'): # restrict to -ve Bz

                condition = '<0'
                cond_col  = 'B_z_GSM'

            else:
                condition, cond_col = None, None

            df_ind = mask_df(df_sc, ind_var, count_col=ind_count, resolution=resolution, condition=condition, cond_col=cond_col, invert=invert)

            if invert:
                ind_var = f'-{ind_var}'

            ###-----MERGING AND SLICING-----###

            df_ind, df_dep = merge_with_lag(df_ind, df_pc, lag, resolution)

            ind_vals = df_ind[ind_var].to_numpy()
            ind_err  = df_ind[ind_err].to_numpy()

            dep_vals = df_dep[dep_var].to_numpy()
            dep_err2 = df_dep[dep_err].to_numpy()**2

            bins = calculate_bins(ind_vals,bin_width)

            if np.all(ind_err == 0):
                print(f'All errors are 0 for {ind_var}.')
                continue

            ###-----ERROR CALCULATIONS-----###

            idx = np.clip(np.digitize(ind_vals, bins) - 1, 0, len(bins) - 2)

            full_data[(ind_var,dep_var)] = np.mean(dep_err2) / np.var(dep_vals)

            for i in range(len(bins)):
                bin_mask = idx == i
                count = bin_mask.sum()

                if count < 10: # at least 10 y values in the bin
                    continue

                y_var = np.var(dep_vals[bin_mask])
                if y_var <= 0:
                    continue
                rows.append({'driver': ind_var, 'response': dep_var,
                             'bin_centre': 0.5 * (bins[i] + bins[i + 1]),
                             'count': count,
                             'ratio': np.mean(dep_err2[bin_mask]) / y_var})

    ###-----SUMMARY-----###
    df_grid = pd.DataFrame(rows)

    def highest(g):
        pos = g['ratio'].idxmax()
        return pd.Series({'max_ratio': g.loc[pos, 'ratio'], 'driver_size': g.loc[pos, 'bin_centre'], 'count': g.loc[pos, 'count']})

    df_summary = df_grid.groupby(['response', 'driver']).apply(highest, include_groups=False).reset_index()
    df_summary['full_ratio'] = df_summary.apply(
        lambda row: full_data.get((row['driver'],row['response']), np.nan), axis=1)

    print()
    print(df_summary.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    print('\nOverall max:')
    print(df_summary.loc[df_summary['max_ratio']==df_summary['max_ratio'].max()])
    print()
    return df_grid, df_summary