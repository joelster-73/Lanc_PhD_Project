# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 15:47:24 2026

@author: richarj2
"""

import numpy as np
import pandas as pd

import itertools as it
from uncertainties import UFloat, ufloat, unumpy as unp

import matplotlib.pyplot as plt
from matplotlib.colors import is_color_like
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from collections import Counter

from ..config import black, blue, colour_dict, database_colour_dict
from ..formatting import  create_label, data_string, format_label, get_plot_label, add_legend
from ..utils import calculate_bins

from ...analysing.fitting import fit_function
from ...analysing.calculations import average_of_averages, std_of_averages, median_with_counts

def is_marker_like(marker):
    try:
        MarkerStyle(marker)
        return True
    except ValueError:
        return False

def plot_fit(xs, ys, x_range=None, **kwargs):

    fit_type    = kwargs.get('fit_type',None)
    inc_errs    = kwargs.get('inc_errs',True)
    ys_unc      = kwargs.get('ys_unc',None)

    colour      = kwargs.get('fit_colour','r')
    ls          = kwargs.get('fit_style','-')
    lw          = kwargs.get('fit_width',1.5)

    show_error  = kwargs.get('show_error',False)
    as_text     = kwargs.get('as_text',True)
    orientation = kwargs.get('orientation','vertical')

    if ys_unc is None:
        inc_errs = False

    ax = kwargs.get('ax', None)
    if ax is None:
        fig, ax = plt.subplots()

    if fit_type is None:
        #print('No fit type entered.')
        return {}

    try:
        unit  = ys.attrs.get('units',{})[ys.name]
        xunit = xs.attrs.get('units',{})[xs.name]
    except:
        unit  = kwargs.get('unit','')
        xunit = kwargs.get('unit','')

    if unit=='1':
        unit = ''

    fit_dict = fit_function(xs, ys, **kwargs)

    if len(fit_dict)==0:
        return {}

    func = fit_dict['func']
    popt = [v.n for v in fit_dict['params'].values()]
    param_dict = fit_dict['params']

    if fit_dict['R2'] < 0:
        print(f'Fitting failed, R2<0. popt: {popt}.')
        return fit_dict

    if x_range is None and func.__name__ == 'straight_line':
        if len(popt)==1:
            m = popt[0]
            c = 0
        else:
            m, c = popt

        if orientation == 'vertical':
            ax.axline((0, c), slope=m, color=colour, linestyle=ls, linewidth=lw)
        else:
            ax.axline((c, 0), slope=1/m, color=colour, linestyle=ls, linewidth=lw)
    else:

        if x_range is None:
            x_range = np.linspace(np.min(xs),np.max(xs),100)

        x_vals = x_range
        y_vals = func(x_range, *popt)
        if orientation=='horizontal':
            x_vals, y_vals = y_vals, x_vals

        ax.plot(x_vals, y_vals, c=colour, ls=ls, lw=lw)

        if show_error:
            popt_u = [v for v in fit_dict['params'].values()]  # for error region
            y_vals_u = func(x_range, *popt_u)
            y_std = unp.std_devs(y_vals_u)
            ax.fill_between(x_range, y_vals - y_std, y_vals + y_std, color=colour, alpha=0.3)


    label = get_plot_label(fit_type)

    if fit_type in ('bimodal','bimodal_offset'):
        label1, label2 = format_label(label[0], param_dict, inc_errs, unit), format_label(label[1], param_dict, inc_errs, unit)

        if as_text and orientation=='vertical':
            text = label1.replace('\n',', ') + '\n' + label2.replace('\n',', ')
            ax.text(0.5, 0.975, text, ha='center', va='top', transform=ax.transAxes, family='monospace')
        elif as_text and orientation=='horizontal':
            text = label1 + '\n' + label2
            ax.text(0.5, 0.975, text, ha='center', va='top', transform=ax.transAxes, family='monospace')
        else:
            ax.plot([],[],' ',label=label1)
            ax.plot([],[],' ',label=label2)

    else:
        statistics = {'chi2': fit_dict['chi2']}
        if fit_type=='straight':
            statistics['R2'] = fit_dict['R2']

        label = format_label(label, param_dict, inc_errs, unit, xunit, **statistics)

        if as_text:
            ax.text(0.5, 0.975, label, color=colour, ha='center', va='top', transform=ax.transAxes, family='monospace')
        else:
            ax.plot([],[],' ',label=label)

    return fit_dict

# %% styles

def plot_scatter(xs, ys, **kwargs):

    xs_unc       = kwargs.get('xs_unc',None)
    ys_unc       = kwargs.get('ys_unc',None)
    zs           = kwargs.get('zs',None)

    data3_name    = kwargs.get('data3_name',None)

    data_colour  = kwargs.get('data_colour',blue)
    error_colour = kwargs.get('error_colour','k')
    scat_size    = kwargs.get('scatter_size',0.5)
    scat_marker  = kwargs.get('scat_marker','o')
    zero_lines   = kwargs.get('zero_lines',False)

    z_min        = kwargs.get('z_min',None)
    z_max        = kwargs.get('z_max',None)

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)

    if fig is None or ax is None:
        fig, ax = plt.subplots()


    if xs_unc is not None or ys_unc is not None:
        ax.errorbar(xs, ys, xerr=xs_unc, yerr=ys_unc, fmt='.', ms=0, ecolor=error_colour, capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

    if not is_marker_like(scat_marker):
        scat_marker = 'o'

    if zs is not None:
        # Using a dictionary to map the colours
        if data_colour in ('spacecraft','detector','database'):

            if data_colour == 'database':
                sc_colours = database_colour_dict
            else:
                sc_colours = colour_dict

            spacecraft_counts = Counter(zs)
            colours = pd.Series(zs).str.upper().map(sc_colours).fillna(black).to_numpy()
            scatter = ax.scatter(xs, ys, c=colours, marker=scat_marker, s=scat_size, alpha=0.8)

            legend_elements = [Line2D([0], [0], marker='o', color=colour, label=f'{label}: {spacecraft_counts.get(label, 0)}', markersize=1, linestyle='None') for label, colour in sc_colours.items() if spacecraft_counts.get(label, 0) > 0]

            ax.legend(handles=legend_elements, fontsize=8, loc='upper left', title=f'N={sum(spacecraft_counts.values()):,}', title_fontsize=10)

        else:
            # Using a z-value to apply a continuous colour map
            cmap = plt.cm.get_cmap('cool').copy()
            cmap.set_bad(color='black')
            scatter = ax.scatter(xs, ys, c=zs, cmap=cmap, vmin=z_min, vmax=z_max, marker=scat_marker, s=scat_size)

            z_label = create_label(zs.name, data_name=data3_name, units=zs.attrs.get('units',{}))

            cbar = plt.colorbar(scatter)
            cbar.set_label(z_label)

    elif is_color_like(data_colour):
        ax.scatter(xs, ys, c=data_colour, marker=scat_marker, s=scat_size)

    else:
        ax.scatter(xs, ys, c=blue, marker=scat_marker, s=scat_size)

    if zero_lines:
        ax.axhline(0,c='grey',ls=':')
        ax.axvline(0,c='grey',ls=':')

    return fig, ax

def plot_scatter_binned(xs, ys, **kwargs):

    zs           = kwargs.get('zs',None)

    xs_unc       = kwargs.get('xs_unc',None)
    ys_unc       = kwargs.get('ys_unc',None)
    zs_unc       = kwargs.get('zs_unc',None)

    xs_counts    = kwargs.get('xs_counts',None)
    ys_counts    = kwargs.get('ys_counts',None)
    zs_counts    = kwargs.get('zs_counts',None)

    data3_name    = kwargs.get('data3_name',None)

    data_colour  = kwargs.get('data_colour',blue)
    error_colour = kwargs.get('error_colour','k')
    scat_size    = kwargs.get('scatter_size',0.5)
    scat_marker  = kwargs.get('scat_marker','s')

    bin_step     = kwargs.get('bin_step',1)
    z_min        = kwargs.get('z_min',None)
    z_max        = kwargs.get('z_max',None)

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)

    z_min = np.min(zs) if z_min is None else z_min
    z_max = np.max(zs) if z_max is None else z_max

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if not is_marker_like(scat_marker):
        scat_marker = 's'

    scatter = None

    for X_bin in calculate_bins(xs,bin_step):

        mask = (xs>=X_bin) & (xs<(X_bin+bin_step))

        if np.sum(mask)==0:
            continue

        elif np.sum(mask)==1: # No standard dev.

            x = ufloat(xs.loc[mask].iloc[0],0)
            y = ufloat(ys.loc[mask].iloc[0],0)

            scatter_size = scat_size

        else:
            x = average_of_averages(xs, xs_unc, xs_counts, mask)
            y = average_of_averages(ys, ys_unc, ys_counts, mask)

            if (isinstance(x, float) and np.isnan(x)) or (isinstance(y, float) and np.isnan(y)):
                continue

            scaling = int(np.log10(np.sum(mask)))
            scatter_size = scat_size * scaling**2

            ax.errorbar(x.n, y.n, xerr=x.s, yerr=y.s, fmt='.', ms=0, ecolor=error_colour, capsize=0.4*scaling, capthick=0.3*scaling, lw=0.3*scaling, zorder=2)

        if zs is not None:
            z = average_of_averages(zs, zs_unc, zs_counts, mask)
            if isinstance(z, UFloat):
                z = z.n

            scatter = ax.scatter(x.n, y.n, c=z, cmap='cool', marker=scat_marker, vmin=z_min, vmax=z_max, s=scatter_size)

        else:
            colour = data_colour if is_color_like(data_colour) else blue

            ax.scatter(x.n, y.n, c=colour, s=scatter_size, marker=scat_marker)

    if scatter:
        cbar = plt.colorbar(scatter)

        z_label = create_label(zs.name, data_name=data3_name, units=zs.attrs.get('units',{}))

        cbar.set_label(z_label)

        return fig, ax, cbar

    return fig, ax

def plot_rolling_multiple(xs, ys, **kwargs):

    ys_unc       = kwargs.get('ys_unc',None)
    ys_counts    = kwargs.get('ys_counts',None)

    zs           = kwargs.get('zs',None)
    zs_edges     = kwargs.get('zs_edges',None)
    data3_name   = kwargs.get('data3_name',None)

    z_min        = kwargs.get('z_min',None)
    z_max        = kwargs.get('z_max',None)

    cmap_name    = kwargs.get('cmap_name','cool')

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if zs is None or zs_edges is None:
        print('Need zs and zs edges for splitting scatter.')
        return plot_scatter_binned(xs, ys, **kwargs)

    z_bins, z_labels, z_vals = create_multiple_labels(zs, zs_edges, z_min, z_max, z_name=data3_name)

    z_min = np.min(zs) if z_min is None else z_min
    z_max = np.max(zs) if z_max is None else z_max

    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=z_min, vmax=z_max)

    for z_i, (z_bin, z_val, z_label) in  enumerate(zip(z_bins, z_vals, z_labels)):

        mask = (zs>=z_bin[0]) & (zs<z_bin[1])

        if np.sum(mask)==0:
            continue

        kwargs_bin = kwargs.copy()
        colour = cmap(norm(z_val))
        kwargs_bin.update({'data_colour': colour, 'error_colour': colour})

        if ys_unc is not None:
            kwargs_bin['ys_unc'] = ys_unc.loc[mask]
        if ys_counts is not None:
            kwargs_bin['ys_counts'] = ys_counts.loc[mask]

        _ = plot_rolling_window(xs.loc[mask], ys.loc[mask], **kwargs_bin)

        ax.plot([], [], c=colour, label=z_label)


    return fig, ax


def create_multiple_labels(zs, zs_edges, z_min, z_max, z_name=None):

    z_unit = zs.attrs.get('units',{}).get(zs.name, None)
    z_edges_label = zs_edges

    if z_unit in ('rad','deg','°'):
        z_unit = '°'
        z_edges_label = np.degrees(z_edges_label)
    elif z_unit is not None and z_unit not in ('1','NUM',''):
        z_unit = f' {z_unit}'
    else:
        z_unit = ''

    if z_name is None:
        z_name = zs.name

    if len(zs_edges)==1:
        z_min = np.min(zs) if z_min is None else z_min
        z_max = np.max(zs) if z_max is None else z_max

        z_string = f'{z_edges_label[0]:.1f}'
        if z_edges_label[0]<0.1:
            z_string = f'{z_edges_label[0]:.1g}'

        zs_bins  = [[np.min(zs),zs_edges[0]],[zs_edges[0],np.max(zs)]]
        z_labels = [f'${data_string(z_name)}$<{z_string}{z_unit}',
                    f'${data_string(z_name)}$$\\geq${z_string}{z_unit}']
        z_vals   = [z_min, z_max]

    else:
        z_min = zs_edges[0]
        z_max = zs_edges[-1]

        zs_edges = np.array(zs_edges)
        zs_bins  = []
        z_labels = []
        z_vals   = []
        for z_i, _ in enumerate(zs_edges):

            if z_i==0:
                lower = np.min(zs)
            else:
                lower = (zs_edges[z_i]+zs_edges[z_i-1])/2

            if z_i==(len(zs_edges)-1):
                upper = np.max(zs)
            else:
                upper = (zs_edges[z_i]+zs_edges[z_i+1])/2

            zs_bins.append([lower,upper])
            z_labels.append(f'${data_string(z_name)}$={z_edges_label[z_i]:.1f}{z_unit}')
            z_vals.append(zs_edges[z_i])

    return zs_bins, z_labels, z_vals

def plot_scatter_with_dict(xs, ys, **kwargs):

    xs_unc       = kwargs.get('xs_unc',None)
    ys_unc       = kwargs.get('ys_unc',None)

    error_colour = kwargs.get('error_colour',black)
    scat_size    = kwargs.get('scatter_size',40)
    sc_ups       = kwargs.get('sc_ups',None)
    sc_dws       = kwargs.get('sc_dws',None)

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)

    if sc_ups is None:
        raise Exception('Missing upstream spacecraft.')
    elif sc_dws is None:
        raise Exception('Missing downstream spacecraft.')

    if fig is None or ax is None:
        fig, ax = plt.subplots()


    if xs_unc is not None or ys_unc is not None:
        ax.errorbar(xs, ys, xerr=xs_unc, yerr=ys_unc, fmt='.', ms=0, ecolor=error_colour, capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

    monitors = ('ACE', 'WIND', 'DSC', 'Cluster', 'THEMIS', 'OMNI')
    marker_dict = {'WIND': 'x', 'ACE': '+', 'DSC': '^', 'Cluster': 'o', 'THEMIS': 's'}

    for upstream, downstream in it.permutations(monitors, 2):
        sc_mask = np.all(len(xs))
        if upstream=='Cluster':
            sc_mask &= np.isin(sc_ups, ('C1','C2','C3','C4'))
        elif upstream=='THEMIS':
            sc_mask &= np.isin(sc_ups, ('THA','THB','THC','THD','THE'))
        else:
            sc_mask &= (sc_ups==upstream)

        if downstream=='Cluster':
            sc_mask &= np.isin(sc_dws, ('C1','C2','C3','C4'))
            sc_c=colour_dict['C1']
        elif downstream=='THEMIS':
            sc_mask &= np.isin(sc_dws, ('THA','THB','THC','THD','THE'))
            sc_c=colour_dict['THA']
        else:
            sc_mask &= (sc_dws==downstream)
            sc_c=colour_dict[downstream]

        count = np.sum(sc_mask)
        if count==0:
            continue
        ax.scatter(xs[sc_mask], ys[sc_mask], c=sc_c, s=scat_size, marker=marker_dict[upstream], label=f'{upstream} | {downstream}: {count}')


    return fig, ax

def plot_rolling_window(xs, ys, window_width=5, window_step=0.5, **kwargs):

    ys_unc         = kwargs.get('ys_unc',None)
    ys_counts      = kwargs.get('ys_counts',None)
    min_count      = kwargs.get('min_count',10) # data points in bin
    max_expansions = kwargs.get('max_expansions',5)

    line_style  = kwargs.get('line_style','-')
    want_legend = kwargs.get('want_legend',False)
    show_fit    = kwargs.get('show_fit',False)
    show_count  = kwargs.get('show_count',False)

    mean_colour = kwargs.get('data_colour',blue)
    std_colour  = kwargs.get('error_colour','r')
    region      = kwargs.get('error_region','sem') # std or sem
    data_type   = kwargs.get('data_type','counts')

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',False)

    y_unit = ys.attrs.get('units',{}).get(ys.name,'')
    if y_unit in ('rad','deg'):
        y_unit = '°'

    window_step = min(window_step, window_width/5)

    if fig is None or ax is None:
        fig, ax = plt.subplots()
    if show_count:
        ax2 = ax.twinx()
        ax2.set_yscale('log')
        ax2.set_ylabel(data_type.capitalize())

    x_centres  = np.arange(np.floor(np.min(xs)/window_step)*window_step,
                           np.ceil(np.max(xs)/window_step)*window_step+window_step,
                           window_step)

    y_vals     = np.full(len(x_centres),np.nan)
    if region == 'med':
        y_errs = np.zeros((2,len(x_centres)))
    else:
        y_errs = np.zeros(len(x_centres))
    counts     = np.zeros(len(x_centres))

    w = window_width
    expansions = 0

    for i, x_c in enumerate(x_centres):

        mask = (xs >= x_c-w/2) & (xs <= x_c+w/2)

        if (np.sum(mask) < min_count) and (expansions < max_expansions):
            w += window_step
            expansions += 1
            mask = (xs >= x_c-w/2) & (xs <= x_c+w/2)

        if np.sum(mask)>=min_count:

            if region=='med':
                med, q1, q3 = median_with_counts(ys, ys_counts, mask)
                y_vals[i] = med
                y_errs[0,i], y_errs[1,i] = med - q1, q3 - med

            elif region in ('max','min'):
                if region=='max':
                    mask &= ys >= np.percentile(ys,90)
                elif region=='min':
                    mask &= ys <= np.percentile(ys,10)

                if np.sum(mask)>=min_count//10:
                    val = average_of_averages(ys, ys_unc, ys_counts, mask)
                    if not isinstance(val,float):
                        y_vals[i] = val.n
                        y_errs[i] = val.s

            else:
                val = average_of_averages(ys, ys_unc, ys_counts, mask)
                y_vals[i] = val.n
                if region=='sem':
                    y_errs[i] = val.s
                elif region=='std':
                    y_errs[i] = std_of_averages(ys, ys_unc, ys_counts, mask)
                else:
                    y_errs[i] = 0

            counts[i] = np.sum(mask)

    not_nan = ~np.isnan(y_vals)

    segments = []
    current_segment = []
    for i, y_val in enumerate(y_vals):
        if not_nan[i]:
            if region=='med':
                current_segment.append((x_centres[i], y_val, (y_errs[0,i], y_errs[1,i]), counts[i]))
            else:
                current_segment.append((x_centres[i], y_val, y_errs[i], counts[i]))
        else:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
    if current_segment:
        segments.append(current_segment)

    for segment in segments:
        x_s, y_s, err_s, c_s = zip(*segment)

        x_s   = np.array(x_s)
        y_s   = np.array(y_s)
        err_s = np.array(err_s)
        c_s   = np.array(c_s)

        if len(xs)==1:
            ax.plot(x_s, y_s, c=mean_colour, marker='.')
        else:
            ax.plot(x_s, y_s, c=mean_colour, ls=line_style)

        if region!='none':
            if region =='med':
                ax.fill_between(x_s, y_s-err_s[:,0], y_s+err_s[:,1], color=std_colour, alpha=0.2)
            else:
                ax.fill_between(x_s, y_s-err_s, y_s+err_s, color=mean_colour, alpha=0.3)

        if show_count:
            ax2.plot(x_s, c_s, c='m', ls=':')

    if show_fit:
        _ = plot_fit(x_centres, y_vals, ys_unc=(None if region=='med' else y_errs), unit=y_unit, fit_type='straight', fit_colour='red', fit_style='--', as_text=True, fig=fig, ax=ax, return_objs=True)

    if return_objs:
        return fig, ax

    ax.plot([], [], c=mean_colour, ls=line_style, label='mean')
    if region!='none':
        labels = {'med': 'iqr', 'std': r'$\pm$s', 'sem': r'$\pm s_{\bar{x}}$'}
        ax.fill_between([], [], [], color=std_colour, alpha=0.4, label=labels.get(region,r'$\pm\sigma$'))
        ax.set_ylabel(r'$\mu \pm \hat{\sigma}$'+f' [{y_unit}]')
    add_legend(fig, ax, legend_on=want_legend)

