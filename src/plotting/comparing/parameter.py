# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:27:55 2025

@author: richarj2
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from .procedures import plot_fit, plot_scatter, plot_scatter_binned, plot_scatter_with_dict, plot_rolling_window

from ..config import black
from ..formatting import add_legend, add_figure_title, create_label, data_string, standard_angle_ticks
from ..utils import save_figure, calculate_bins, change_series_name

from ...analysing.comparing import difference_columns



def compare_columns(df, col1, col2, col3=None, col1_err=None, col2_err=None, col1_counts=None, col2_counts=None, **kwargs):
    """
    Plots two columns from one dataframe against each other.
    This ensures the data are in the correct format.
    """

    series1 = df.loc[:,col1]
    series2 = df.loc[:,col2]

    # Include col 3 error etc
    if col3 is not None:
        kwargs['zs'] = df.loc[:,col3]

    if col1_err is not None:
        kwargs['xs_unc'] = df.loc[:,col1_err]

    if col2_err is not None:
        kwargs['ys_unc'] = df.loc[:,col2_err]

    if col1_counts is not None:
        kwargs['xs_counts'] = df.loc[:,col1_counts]

    if col2_counts is not None:
        kwargs['ys_counts'] = df.loc[:,col2_counts]

    return compare_series(series1, series2, **kwargs)


def compare_dataframes(df1, df2, col1, col2, col1_err=None, col2_err=None, col1_counts=None, col2_counts=None, df3=None, col3=None, **kwargs):
    """
    Plots two columns from two different dataframes against each other.
    This ensures the data are in the correct format.
    """

    series1 = df1.loc[:,col1]
    series2 = df2.loc[:,col2]

    # Include col 3 error etc
    if col3 is not None:
        kwargs['zs'] = df3.loc[:,col3]

    if col1_err is not None:
        kwargs['xs_unc'] = df1.loc[:,col1_err]

    if col2_err is not None:
        kwargs['ys_unc'] = df2.loc[:,col2_err]

    if col1_counts is not None:
        kwargs['xs_counts'] = df1.loc[:,col1_counts]

    if col2_counts is not None:
        kwargs['ys_counts'] = df2.loc[:,col2_counts]

    return compare_series(series1, series2, **kwargs)

def investigate_difference(df, col1, col2, ind_col, **kwargs):

    diff_type = kwargs.get('diff_type','absolute')

    series1 = df.loc[:,ind_col]
    series2 = difference_columns(df, col2, col1, diff_type)

    kwargs['reference_line'] = 0
    if kwargs.get('ind_name_lat',None) is not None:
        change_series_name(series1, kwargs.get('ind_name_lat'))
    if kwargs.get('ind_name_str',None) is not None:
        kwargs['data1_name'] = kwargs.get('ind_name_str')

    return compare_series(series1, series2, **kwargs)

# %% Main function

def compare_series(series1, series2, **kwargs):

    display         = kwargs.get('display','scatter')
    bin_width       = kwargs.get('bin_width',None)
    want_legend     = kwargs.get('want_legend',True)
    reference_line  = kwargs.get('reference_line',None)
    add_count       = kwargs.get('add_count',False)
    legend_loc      = kwargs.get('legend_loc',None)
    invert          = kwargs.get('invert',False)

    data1_name    = kwargs.get('data1_name',None)
    data2_name    = kwargs.get('data2_name',None)
    name1_latex   = kwargs.get('name1_latex',False)
    name2_latex   = kwargs.get('name1_latex',False)
    brief_title   = kwargs.get('brief_title','')

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',False)
    return_fit  = kwargs.get('return_fit', False)

    ###---------------CONSTRUCT COLUMN LABELS---------------###

    unit1 = series1.attrs.get('units',{}).get(series1.name, None)
    unit2 = series2.attrs.get('units',{}).get(series2.name, None)

    brief_title = f'Comparing ${data_string(series1.name)}$ and ${data_string(series2.name)}$' if brief_title is None else brief_title


    ###---------------PLOTS MAIN SCATTER/HEAT DATA---------------###
    if fig is None or ax is None:
        fig, ax = plt.subplots()
        kwargs['fig'] = fig
        kwargs['ax'] = ax

    if display == 'scatter_dict':
        try:
            _ = plot_scatter_with_dict(series1, series2, **kwargs)
        except:
            display = 'scatter'
            print('Using normal scatter.')

    if display == 'scatter':
        _ = plot_scatter(series1, series2, **kwargs)

    elif display == 'scatter_binned':
        _ = plot_scatter_binned(series1, series2, **kwargs)

    elif display == 'rolling':
        _ = plot_rolling_window(series1, series2, **kwargs)

    elif display == 'heat':
        if hasattr(bin_width, '__len__') and len(bin_width)==2:
            n_bins = (calculate_bins(series1,bin_width[0]), calculate_bins(series2,bin_width[1]))
        else:
            n_bins = (calculate_bins(series1,bin_width), calculate_bins(series2,bin_width))

        h = ax.hist2d(series1, series2, bins=n_bins, norm=mpl.colors.LogNorm(), cmap='hot')
        ax.set_facecolor('k')

        cbar = fig.colorbar(h[3], ax=ax)
        cbar.set_label('Counts')

    elif display != 'scatter_dict':
        raise ValueError(f'"{display}" not valid display mode.')

    if reference_line is not None:
        if reference_line=='x':
            ax.axline((0, 0), slope=1, color='k', lw=2, ls='-')
            ax.axline((0, 0), slope=1, color='w', lw=2, ls=':')
        elif isinstance(reference_line, int) or isinstance(reference_line, int):
            ax.axhline(y=reference_line, color='k', label=f'y={reference_line}', lw=2, ls='-')
            ax.axhline(y=reference_line, color='w', lw=2, ls=':')

    fit_kwargs = kwargs.copy()
    fit_kwargs['fit_colour'] = kwargs.get('fit_colour',black)
    fit_kwargs['fit_style'] = kwargs.get('fit_style','--')
    fit_kwargs['fit_width'] = 1.25
    fit_kwargs['xunit'] = unit1
    fit_kwargs['yunit'] = unit2

    fit_dict = plot_fit(series1, series2, **fit_kwargs)

    if invert:
        ax.invert_xaxis()

    ###---------------CONVERT TICKS TO DEGREES---------------###
    def rad2deg(x, pos):
        return f'{np.degrees(x):.0f}'

    if unit1 == 'rad':
        standard_angle_ticks(ax, series1, 'x', np.pi/4)
        unit1 = '°'

    if unit2 == 'rad':
        standard_angle_ticks(ax, series2, 'y', np.pi/4)
        unit2 = '°'

    data1_label = create_label(series1.name, unit=unit1, data_name=data1_name, name_latex=name1_latex)
    data2_label = create_label(series2.name, unit=unit2, data_name=data2_name, name_latex=name2_latex)

    ax.set_xlabel(data1_label)
    ax.set_ylabel(data2_label)

    if brief_title=='amount':
        brief_title = f'{len(series1):,}'
    elif brief_title != '' and add_count:
        brief_title += f', N={len(series1):,}'

    ###---------------LABELLING AND FINISHING TOUCHES---------------###
    add_legend(fig, ax, legend_on=want_legend, heat=display=='heat', loc=legend_loc)
    add_figure_title(fig, brief_title, ax=ax)
    plt.tight_layout();

    if return_objs:
        try:
            return fig, ax, cbar, fit_dict if return_fit else None
        except:
            return fig, ax, fit_dict if return_fit else None

    save_figure(fig)
    plt.show()
    plt.close()

# %% Nguyen

def plot_nguyen(df, col_n, col_b, col_n_sw=None, col_b_sw=None, **kwargs):

    if col_n_sw is None:
        col_n_sw = col_n
    if col_b_sw is None:
        col_b_sw = col_b

    mask = np.ones(len(df),dtype=bool)
    for param in (f'{col_n}_msh',f'{col_n_sw}_sw',f'{col_b}_msh',f'{col_b_sw}_sw'):
        mask &= ~np.isnan(df[param])

    series_n = (df.loc[mask,f'{col_n}_msh']/df.loc[mask,f'{col_n_sw}_sw'])
    series_b = (df.loc[mask,f'{col_b}_msh']/df.loc[mask,f'{col_b_sw}_sw'])

    series_n.name = r'N_C1 / N_OMNI'
    series_b.name = r'B_C1 / B_OMNI'

    # rename

    kwargs['bin_width'] = (0.025,0.05)
    kwargs['display'] = 'heat'
    kwargs['return_objs'] = True

    fig, ax, cbar = compare_series(series_n, series_b, **kwargs)

    x1 = np.linspace(0, 6, 200)
    m1 = 10
    y_upper = m1*x1

    c2 = 2.5
    m2 = -(c2/2)

    x2 = np.linspace(c2/(m1-m2),-c2/m2, 200)
    y_lower = m2*x2 + c2

    ax.plot(x1, y_upper, 'c', lw=4) # MS/MSH boundary
    ax.plot(x2, y_lower, 'c', lw=4) # SW/MSH boundary

    ax.set_xlim(right=8)
    ax.set_ylim(top=12)

    save_figure(fig)
    plt.show()
    plt.close()