# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:27:55 2025

@author: richarj2
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from ..config import black, white
from ..formatting import add_legend, add_figure_title, create_label, dark_mode_fig, data_string
from ..utils import save_figure, calculate_bins

from ...analysing.comparing import difference_columns
from ...analysing.fitting import straight_best_fit

def compare_columns(df, col1, col2, **kwargs):

    series1 = df.loc[:,col1]
    series2 = df.loc[:,col2]

    compare_series(series1, series2, **kwargs)

def investigate_difference(df, col1, col2, ind_col, **kwargs):

    diff_type = kwargs.get('diff_type','absolute')
    ind_name  = kwargs.get('ind_name',None)

    series1 = difference_columns(df, col1, col2, diff_type)
    series2 = df.loc[:,ind_col]

    kwargs['reference_line'] = 0
    kwargs['data2_name'] = ind_name

    compare_series(series1, series2, **kwargs)

def compare_series(series1, series2, **kwargs):

    best_fit        = kwargs.get('stats',False)
    display         = kwargs.get('display','Scatter')
    bin_width       = kwargs.get('bin_width',None)
    scat_size       = kwargs.get('scatter_size',0.4)
    want_legend     = kwargs.get('want_legend',True)
    reference_line  = kwargs.get('reference_line',None)

    data1_name    = kwargs.get('data1_name',None)
    data2_name    = kwargs.get('data2_name',None)
    brief_title   = kwargs.get('brief_title',None)

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',False)

    is_heat = False
    if display == 'Heat':
        is_heat = True

    time_mask = (np.isfinite(series1)) & (np.isfinite(series2))
    if np.sum(time_mask)==0:
        print('No valid overlap of data')
        return

    series1 = series1[time_mask]
    series2 = series2[time_mask]

    ###---------------CONSTRUCT COLUMN LABELS---------------###

    data1_str = data_string(series1.name)
    data2_str = data_string(series2.name)
    unit1 = series1.attrs['units'].get(series1.name, None)
    unit2 = series2.attrs['units'].get(series2.name, None)

    data1_label = create_label(data1_str, unit=unit1, data_name=data1_name, name_latex=True)
    data2_label = create_label(data2_str, unit=unit2, data_name=data2_name, name_latex=True)
    title_str = f'Comparing ${data1_str}$ and ${data2_str}$' if brief_title is None else brief_title


    ###---------------PLOTS MAIN SCATTER/HEAT DATA---------------###
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if display == 'Scatter':
        ax.scatter(series2, series1, c='b', s=scat_size)

    elif display == 'Heat':
        if hasattr(bin_width, '__len__') and len(bin_width) == 2:
            n_bins = (calculate_bins(series2,bin_width[0]), calculate_bins(series1,bin_width[1]))
        else:
            n_bins = (calculate_bins(series2,bin_width), calculate_bins(series1,bin_width))
        h = ax.hist2d(series2, series1, bins=n_bins, norm=mpl.colors.LogNorm(), cmap='hot')

        cbar = fig.colorbar(h[3], ax=ax)
        cbar.ax.tick_params(colors=black)
        cbar.set_label('Number of Points', color=black)
        cbar.outline.set_edgecolor(black)

        line_color = 'w'

    else:
        raise ValueError(f'"{display}" not valid display mode.')

    if reference_line is not None:
        if reference_line=='x':
            ax.axline((0, 0), slope=1, color=line_color, label='y=x', lw=2, ls=':')
        elif isinstance(reference_line, int) or isinstance(reference_line, int):
            ax.axhline(y=reference_line, color=line_color, label=f'y={reference_line}', lw=2, ls=':')

    ax.set_xlabel(data2_label, c=black)
    ax.set_ylabel(data1_label, c=black)

    if best_fit:
        m, y0, _ = straight_best_fit(series2, series1, name=title_str)
        ax.axline((0,y0), slope=m, c='magenta', label=f'Best Fit: {m:.3f}x+{y0:.3f} {unit2}', lw=2.5, ls='--')

    ###---------------LABELLING AND FINISHING TOUCHES---------------###
    add_legend(fig, ax, legend_on=want_legend, heat=is_heat)
    add_figure_title(fig, black, title_str, ax=ax)
    dark_mode_fig(fig,black,white,is_heat)
    plt.tight_layout();
    if return_objs:
        if display=='scatter':
            return fig, ax
        return fig, ax, cbar

    save_figure(fig)
    plt.show()
    plt.close()


