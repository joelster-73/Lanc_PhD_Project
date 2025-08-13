# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:27:55 2025

@author: richarj2
"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import Counter

from ..config import black, white, blue
from ..formatting import add_legend, add_figure_title, create_label, dark_mode_fig, data_string
from ..utils import save_figure, calculate_bins, calculate_bins_edges
from ..distributions import plot_gaussian, plot_bimodal, plot_bimodal_offset, plot_lognormal

from ...analysing.calculations import calc_mean_error
from ...analysing.comparing import difference_columns, difference_dataframes
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



def compare_dataframes_xy(df1, df2, data_x, data_y, **kwargs):
    """
    Investigates differences between two data sources for a given field, plotting the results and optionally
    displaying statistical summaries.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to be analyzed and plotted. The DataFrame must have a DateTimeIndex
        and columns for the specified x and y fields.


    Returns
    -------
    None : This procedure generates and displays a plot based on the specified parameters and data.

    """
    datax_name  = kwargs.get('datax_name',None)
    datay_name  = kwargs.get('datay_name',None)
    df1_name    = kwargs.get('df1_name','df1')
    df2_name    = kwargs.get('df2_name','df2')
    title       = kwargs.get('title',None)
    stats       = kwargs.get('stats',True)
    quality_col    = kwargs.get('quality_col',None)
    quality_nums    = kwargs.get('quality_nums',None)
    quality_labels = kwargs.get('quality_labels',None)

    ###---------------CONSTRUCT COLUMN LABELS---------------###

    # Validate column labels
    for label in (data_x, data_y):
        if label not in df1.keys():
            raise ValueError(f'Field data "{label}" not found in dataframe {df1}.')
        if label not in df2.keys():
            raise ValueError(f'Field data "{label}" not found in dataframe {df2}.')

    datax_str = data_string(data_x)
    datay_str = data_string(data_y)
    unitx = df1.attrs['units'].get(data_x, None)
    unity = df2.attrs['units'].get(data_y, None)

    datax_label = create_label(datax_str, unit=unitx, data_name=datax_name)
    datay_label = create_label(datay_str, unit=unity, data_name=datay_name)
    if title is None:
        title_str = f'${datay_str}$ vs. ${datax_str}$'
    if df1_name is not None and df2_name is not None:
        title_str += f' for {df1_name} and {df2_name}'
    else:
        title_str +=  ' for two dataframes'
    quality_str = None

    ###---------------CREATES FIGURE WITH DESIRED SUBPLOTS---------------###

    fig, ax_main = plt.subplots()

    unique_indices = df1.index.union(df2.index)
    common_indices = df1.index.intersection(df2.index)
    df_compare = df1.loc[common_indices][[data_x,data_y]]

    ax_main.scatter(df1[data_x],df1[data_y],s=0.1,label=f'{df1_name}: {len(df1):,}',c=blue)
    ax_main.scatter(df2[data_x],df2[data_y],s=0.1,label=f'{df2_name}: {len(df2):,}',c='r')
    ax_main.scatter(df_compare[data_x],df_compare[data_y],s=0.1,label=f'∩: {len(df_compare):,}',c='yellow')

    ax_main.set_xlabel(datax_label, c=black)
    ax_main.set_ylabel(datay_label, c=black)

    ###---------------PLOT COLUMN CHARTS OF MEANS AND COUNTS---------------###
    if stats:

        # (1) Adds lines to scatter plots
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        min_limit = max(x_min, y_min)
        max_limit = min(x_max, y_max)
        line = np.linspace(min_limit, max_limit, 500)
        xs = np.linspace(x_min, x_max, 500)
        ax_main.plot(line, line, c=black, label='y=x', lw=2, ls=':')

        m, y0, _ = straight_best_fit(df1[data_x], df1[data_y], name=df1_name)
        ax_main.plot(xs, m*xs+y0, c='magenta', label=f'Best Fit {df1_name}\n{m:.3f}x+{y0:.3f} [{unity}]', lw=2, ls='-.') # Best fit
        m, y0, _ = straight_best_fit(df2[data_x], df2[data_y], name=df2_name)
        ax_main.plot(xs, m*xs+y0, c='green',  label=f'Best Fit {df2_name}\n{m:.3f}x+{y0:.3f} [{unity}]', lw=2, ls='-.') # Best fit

        if quality_col is not None:
            if quality_col in df1:
                my_list = df1[quality_col]
            elif quality_col in df2:
                my_list = df2[quality_col]
            counts = dict(Counter(my_list))
            try:
                counter = 0
                for q_i in quality_nums:
                    counter += counts[q_i]
                good_perc = (counter / sum(counts.values())) * 100
            except:
                good_perc = None
            if good_perc is not None:
                if quality_labels is not None:
                    quality_str = f' {good_perc:.3g}% of data of quality {quality_labels}.'
                else:
                    quality_str = f' {good_perc:.3g}% of data of quality {quality_nums}.'


    ###---------------LABELLING AND FINISHING TOUCHES---------------###
    ax_main.legend(loc='upper left', frameon=False, labelcolor=black)
    dark_mode_fig(fig,black,white)

    start_date_str = min(df1.index[0], df2.index[0]).strftime('%Y-%m-%d')
    end_date_str = max(df1.index[-1], df2.index[-1]).strftime('%Y-%m-%d')

    fig.suptitle(f'{title_str}. Data from {start_date_str} to {end_date_str}. '
                 f'Scatter shows {len(unique_indices):,} unique data points.{quality_str}', c=black, wrap=True)
    plt.tight_layout();
    save_figure(fig)
    plt.show()
    plt.close()




def plot_compare_dataset_parameters(df1, df2, *columns, **kwargs):

    df1_colour           = kwargs.get('df1_colour','g')
    df2_colour           = kwargs.get('df2_colour','b')
    fit_colour           = kwargs.get('fit_colour','r')
    df1_name             = kwargs.get('df1_name','Jelínek')
    df2_name             = kwargs.get('df2_name','GRMB')

    column_names         = kwargs.get('column_names',{})
    column_bin_widths    = kwargs.get('column_bin_widths',{})
    column_display_range = kwargs.get('column_display_ranges',{})
    fit_types            = kwargs.get('fit_types',{})
    q_q_plot             = kwargs.get('q_q_plot',False)
    q_q_title            = kwargs.get('q_q_title',False)

    omni_columns          = kwargs.get('omni_columns',{})
    hist_diff             = kwargs.get('hist_diff',False)
    diff_type             = kwargs.get('diff_type','Absolute')
    contemp_omni          = kwargs.get('contemp_omni',False)

    if not contemp_omni:
        hist_diff = False # prevents errors

    if contemp_omni:
        if len(df2)>len(df1):
            df2 = df2[df2.index.isin(df1.index)]
        else:
            df1 = df1[df1.index.isin(df2.index)]

    n_cols = 3 if q_q_plot or hist_diff else 2
    n_rows = len(columns)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols*6, n_rows*4+0.5)
    )

    units = df1.attrs['units']

    if n_rows == 1:
        axs = (axs,)

    ###-------------------PLOT PARAMETERS-------------------###

    for i_r, (ax_row, col) in enumerate(zip(axs, columns)):


        col_name      = column_names.get(col,None)
        col_fit       = fit_types.get(col,None)
        bin_width     = column_bin_widths.get(col,0.25)
        display_range = column_display_range.get(col,None)

        x_unit = units[col]
        if x_unit == 'rad':
            x_unit = '°'
        x_axis_label = create_label(col,x_unit,col_name)

        ###-------------------PLOT COLUMNS-------------------###
        for ax_col, df, colour, name in zip(ax_row[:2], (df1, df2), (df1_colour,df2_colour), (df1_name,df2_name)):

            if name == 'OMNI':
                col = omni_columns.get(col,col)
            series = df[col].to_numpy()
            series = series[~np.isnan(series)]

            if x_unit == '°':
                series = np.degrees(series)

            bin_edges = calculate_bins_edges(series,bin_width)
            counts, bins, _ = ax_col.hist(
                series, bins=bin_edges,
                alpha=1.0, color=colour,
            )
            n_peaks = 0
            if col_fit == 'Gaussian':
                peak, height = plot_gaussian(ax_col,counts,bins,c=fit_colour,ls='-',name=col,detailed=True,return_peak=True)
                n_peaks = 1
                ax_col.text(0.025,0.075,f'$\\longleftarrow$\n{np.min(series):.1f}', transform=ax_col.transAxes)
                ax_col.text(0.875,0.075,f'$\\longrightarrow$\n{np.max(series):.1f}', transform=ax_col.transAxes)
            elif col_fit == 'Bimodal':
                peaks, heights = plot_bimodal(ax_col,counts,bins,c=fit_colour,ls='-',name=col,detailed=True,return_peak=True)
                n_peaks = 2
                ax_col.text(0.025,0.075,f'$\\longleftarrow$\n{np.min(series):.1f}', transform=ax_col.transAxes)
                ax_col.text(0.875,0.075,f'$\\longrightarrow$\n{np.max(series):.1f}', transform=ax_col.transAxes)
            elif col_fit == 'Bimodal_offset':
                peaks, heights = plot_bimodal_offset(ax_col,counts,bins,c=fit_colour,ls='-',name=col,detailed=True,return_peak=True)
                n_peaks = 2
            elif col_fit == 'Log_Normal':
                peak, height = plot_lognormal(ax_col,counts,bins,c=fit_colour,ls='-',name=col,detailed=True,return_peak=True)
                n_peaks = 1
                ax_col.text(0.875,0.075,f'$\\longrightarrow$\n{np.max(series):.1f}', transform=ax_col.transAxes)


            if n_peaks >0:
                label = None
                if n_peaks == 1:
                    try:
                        position = peak.n
                        if col_fit == 'Log_Normal':
                            label = f'${peak:.1uL}$ {x_unit}'
                    except:
                        position = peak
                        if col_fit == 'Log_Normal':
                            label = f'{peak:.3g} {x_unit}'
                    ax_col.vlines(x=position,ymin=0,ymax=height,ls='--',colors='r')
                    ax_col.text(x=position+0.75,y=0.9*ax_col.get_ylim()[1],s=label)
                    location = 'upper right'

                if n_peaks == 2:
                    try:
                        positions = [peaks[0].n,peaks[1].n]
                    except:
                        positions = peaks
                    ax_col.vlines(x=positions,ymin=0,ymax=heights,ls='--',colors='r')
                    location = 'split'

            if display_range is not None:
                ax_col.set_xlim(display_range[0], display_range[-1])
            if col_fit == 'Bimodal_offset':
                ax_col.set_ylim(0.8*np.min(counts),1.15*np.max(counts))

            ax_col.set_xlabel(x_axis_label,c=black)
            add_legend(fig,ax_col,loc=location,edge_col='w',frame_on=False)
            if i_r == 0:
                if name == 'OMNI' and contemp_omni:
                    add_figure_title(fig,'Contemporaneous OMNI',ax=ax_col)
                else:
                    add_figure_title(fig,f'{name} Dataset',ax=ax_col)


        if q_q_plot:
            ax_q = ax_row[2]

            series_1 = df1[col].dropna().to_numpy()
            series_2 = df2[col].dropna().to_numpy()
            quantiles = np.arange(1, 100, 1)  # Restrict to 1 to 99 for the main plot
            deciles = np.arange(10, 100, 10)

            quantiles_1 = np.percentile(series_1, quantiles)
            quantiles_2 = np.percentile(series_2, quantiles)

            deciles_1 = np.percentile(series_1, deciles)
            deciles_2 = np.percentile(series_2, deciles)

            zeroth_1   = np.percentile(series_1, 0)
            zeroth_2   = np.percentile(series_2, 0)
            hundreth_1 = np.percentile(series_1, 100)
            hundreth_2 = np.percentile(series_2, 100)

            if x_unit == '°':
                quantiles_1 = np.degrees(quantiles_1)
                quantiles_2 = np.degrees(quantiles_2)

                deciles_1 = np.degrees(deciles_1)
                deciles_2 = np.degrees(deciles_2)

                zeroth_1   = np.degrees(zeroth_1)
                zeroth_2   = np.degrees(zeroth_2)
                hundreth_1 = np.degrees(hundreth_1)
                hundreth_2 = np.degrees(hundreth_2)

            ax_q.axline(xy1=(0, 0), slope=1, lw=1.5, c='k', zorder=1)
            ax_q.scatter(quantiles_1, quantiles_2, c='darkgrey', edgecolors='grey', zorder=2, alpha=0.8)
            ax_q.scatter(deciles_1, deciles_2, c='r', edgecolors='k', zorder=3)

            if zeroth_1 >= quantiles_1[0]-10 and zeroth_2 >= quantiles_2[0]-10:
                ax_q.scatter(zeroth_1, zeroth_2, c='magenta', edgecolors='k', zorder=3)

            if hundreth_1 <= quantiles_1[-1]+10 and hundreth_2 <= quantiles_2[-1]+10:
                ax_q.scatter(hundreth_1, hundreth_2, c='magenta', edgecolors='k', zorder=3)


            quantiles_full = np.arange(0, 100.1, 1)  # Restrict to 1 to 99 for the main plot
            quantiles_1_full = np.percentile(series_1, quantiles_full)
            quantiles_2_full = np.percentile(series_2, quantiles_full)

            #correlation = np.corrcoef(quantiles_1_full, quantiles_2_full)[0, 1]

            add_legend(fig, ax_q, loc='upper left')
            ax_q.set_xlabel(f'{df1_name} Quantile [{x_unit}]', c=df1_colour)
            ax_q.set_ylabel(f'{df2_name} Quantile [{x_unit}]', c=df2_colour)

            if i_r == 0:
                if q_q_title:

                    m, _, _ = straight_best_fit(quantiles_1, quantiles_2, name=col, detailed=True)
                    add_figure_title(fig,f'$m = {m:.1uL}$',ax=ax_q)
                else:
                    add_figure_title(fig,f'{len(df1):,} mins | {len(df2):,} mins',ax=ax_q)

            if x_unit != '°':

                # Inset plot (zoomed out, 0 to 100 percentiles)
                inset_ax = inset_axes(ax_q, width='30%', height='30%', loc='upper left')
                deciles_full = np.arange(0, 100.1, 10)  # Full range for the inset plot
                deciles_1_full = np.percentile(series_1, deciles_full)
                deciles_2_full = np.percentile(series_2, deciles_full)

                inset_ax.axline(xy1=(0, 0), slope=1, lw=1, c='k', zorder=1)
                inset_ax.scatter(quantiles_1_full, quantiles_2_full, c='darkgrey', s=20, edgecolors='grey', zorder=2, alpha=0.8)
                inset_ax.scatter(deciles_1_full, deciles_2_full, c='r', s=20, edgecolors='k', zorder=3, alpha=0.9)
                inset_ax.scatter(deciles_1_full[0], deciles_2_full[0], c='magenta', s=20, edgecolors='k', zorder=3)
                inset_ax.scatter(deciles_1_full[-1], deciles_2_full[-1], c='magenta', s=20, edgecolors='k', zorder=3)

                inset_ax.set_xticks([np.floor(deciles_1_full[0]), np.ceil(deciles_1_full[-1])])
                inset_ax.set_yticks([np.floor(deciles_2_full[0]), np.ceil(deciles_2_full[-1])])
                inset_ax.yaxis.tick_right()
                inset_ax.tick_params(axis='x', which='both', labelsize=7)
                inset_ax.tick_params(axis='y', which='both', labelsize=7)

                #inset_ax.set_xticks([])  # Remove ticks for clarity
                #inset_ax.set_yticks([])

        if hist_diff:
            ax_h = ax_row[2]
            col_omni = omni_columns.get(col,col)

            if diff_type == 'Absolute':
                differences = difference_dataframes(df1, df2, col, col_omni)

                x_axis_label = create_label(None,unit=x_unit,data_name='C1 - OMNI')

            elif diff_type == 'Relative':
                differences = difference_dataframes(df1, df2, col, col_omni) / df2[col_omni]
                differences.replace([np.inf, -np.inf], np.nan, inplace=True)
                differences.dropna(inplace=True)

                x_axis_label = create_label(column=None,data_name='(C1 - OMNI) / OMNI',unit=None)

            if x_unit == '°':
                differences = np.degrees(differences)

            n_bins = calculate_bins(differences,bin_width/10)
            counts, bins, _ = ax_h.hist(
                differences, bins=n_bins, alpha=1.0,
                color='grey',
            )

            the_mean = calc_mean_error(differences)
            ax_h.axvline(x=the_mean.n,ls='--',c='r',lw=1.5)
            ax_h.plot([],[],' ',label=f'Mean:\n${the_mean:.1uL}$ {x_unit}')

            if x_unit != '°':
                perc=0.5
                perc_range = (np.percentile(differences, perc), np.percentile(differences, 100-perc))
                ax_h.set_xlim(perc_range)

                ax_h.text(0.025,0.075,f'$\\longleftarrow$\n{np.min(differences):.1f}', transform=ax_h.transAxes)
                ax_h.text(0.9,0.075,f'$\\longrightarrow$\n{np.max(differences):.1f}', transform=ax_h.transAxes)

            ax_h.set_xlabel(x_axis_label)
            if i_r == 0:
                #add_figure_title(fig,f'${the_mean:.1uL}$ {x_unit}',ax=ax_h)
                add_figure_title(fig,f'{len(df1):,} minutes',ax=ax_h)
            add_legend(fig, ax_h, loc='upper right', edge_col='w', frame_on=False)
            ax_h.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))


    for ax_row in axs:
        for ax_col in ax_row[:2]:
            ax_col.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))
        ax_row[0].set_ylabel('Count',c=black)

    dark_mode_fig(fig,black,white)
    #add_figure_title(fig, 'Jelínek Dataset vs GRMB Dataset\n')
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()