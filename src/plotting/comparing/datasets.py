# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:27:55 2025

@author: richarj2
"""

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

from ..config import black, white
from ..formatting import dark_mode_fig
from ..utils import save_figure
from ..distributions import plot_freq_hist, plot_q_q

from ...analysing.comparing import difference_series



def plot_compare_dataset_parameters(df1, df2, *columns, **kwargs):

    df1_colour     = kwargs.get('df1_colour','g')
    df2_colour     = kwargs.get('df2_colour','b')
    df1_name       = kwargs.get('df1_name','')
    df2_name       = kwargs.get('df2_name','')

    column_names   = kwargs.get('column_names',{})
    bin_widths     = kwargs.get('bin_widths',{})
    display_ranges = kwargs.get('display_ranges',{})
    fit_types      = kwargs.get('fit_types',{})

    compare_type   = kwargs.get('compare_type',None)
    compare_label  = kwargs.get('compare_label',None)

    df2_columns    = kwargs.get('df2_columns',{})
    contemp_times  = kwargs.get('contemp_times',False)

    if contemp_times:
        if len(df2)>len(df1):
            df2 = df2[df2.index.isin(df1.index)]
        else:
            df1 = df1[df1.index.isin(df2.index)]

    n_cols = 3 if (compare_type is not None) else 2
    n_rows = len(columns)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols*6, n_rows*4+0.5)
    )

    units = df1.attrs['units']

    if n_rows == 1:
        axs = (axs,)

    ###-------------------PLOT PARAMETERS-------------------###

    for ax_row, col in zip(axs, columns):
        col2 = df2_columns.get(col,col)

        series1 = df1.loc[:,col]
        series2 = df2.loc[:,col2]

        display_range = display_ranges.get(col,(0,100))

        kwargs['perc_low']  = display_range[0]
        kwargs['perc_high'] = display_range[1]
        kwargs['bin_width'] = bin_widths.get(col,0.25)
        kwargs['fit_type']  = fit_types.get(col,None)
        kwargs['data_name'] = column_names.get(col,None)

        ax1, ax2 = ax_row[0], ax_row[1]
        ##-------------------PLOT COLUMNS-------------------###
        for ax_col, series, colour, title in zip((ax1, ax2), (series1, series2), (df1_colour, df2_colour), (df1_name, df2_name)):
            _ = plot_freq_hist(series, colour=colour, brief_title=title, fig=fig, ax=ax_col, return_objs=True, **kwargs)

            ax_col.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

        ax_row[1].set_ylabel(None)

        if compare_type is not None:
            ax3 = ax_row[-1]

            if compare_type=='q_q_plot':
                _ = plot_q_q(series1, series2, series1_colour=df1_colour, series2_colour=df2_colour, series1_name=df1_name, series2_name=df2_name, fig=fig, ax=ax3, return_objs=True, **kwargs)

            elif compare_type=='hist_diff':
                kwargs3 = kwargs
                kwargs3['fit_type']  = 'mean'
                kwargs3['data_name'] = compare_label
                kwargs3['bin_width'] = 0.5*kwargs['bin_width']
                if units[col] not in ('rad','deg','°'):
                    if display_range[0]<0.5:
                        kwargs3['perc_low'] = 0.5
                    if display_range[1]>99.5:
                        kwargs3['perc_high'] = 99.5

                _ = plot_freq_hist(difference_series(series1,series2), colour='grey', fig=fig, ax=ax3, return_objs=True, **kwargs3)

            else:
                print(f'{compare_type} is not valid compare type.')

            ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

    if n_rows>1:
        for ax_row in axs[1:]:
            for ax_col in ax_row:
                ax_col.set_title(None)


    dark_mode_fig(fig,black,white)
    #add_figure_title(fig,black,'Jelínek Dataset vs GRMB Dataset')
    plt.tight_layout()

    save_figure(fig)
    plt.show()
    plt.close()