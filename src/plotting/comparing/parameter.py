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

from ..config import black, white, blue
from ..formatting import add_legend, add_figure_title, create_label, dark_mode_fig, data_string
from ..utils import save_figure, calculate_bins
from ..distributions import plot_gaussian

from ...analysing.comparing import difference_columns
from ...analysing.kobel import load_compression_ratios, are_points_above_line
from ...analysing.fitting import straight_best_fit
from ...processing.filtering import filter_by_spacecraft, filter_sign


def compare_columns(df, data1_col, data2_col, **kwargs):
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
    df = df.copy()
    df = df[np.isfinite(df[data2_col]) & np.isfinite(df[data1_col])]

    best_fit        = kwargs.get('stats',False)
    display         = kwargs.get('display','Scatter')
    bin_width       = kwargs.get('bin_width',None)
    hist_bin_width  = kwargs.get('hist_bin_width',1)
    scat_size       = kwargs.get('scatter_size',0.4)
    want_legend     = kwargs.get('want_legend',True)
    hist_diff       = kwargs.get('hist_diff',False)
    fit_gaus        = kwargs.get('fit_gaus',False)
    perc_filter     = kwargs.get('perc_filter',None)
    compressions    = kwargs.get('compressions',None)
    contam_info     = kwargs.get('contam_info',False)

    data1_name    = kwargs.get('data1_name',None)
    data2_name    = kwargs.get('data2_name',None)
    brief_title   = kwargs.get('brief_title',None)

    quality_col    = kwargs.get('quality_col',None)
    quality_labels = kwargs.get('quality_labels',None)

    complexity_col    = kwargs.get('complexity_col',None)
    complexity_labels = kwargs.get('complexity_labels',None)

    sc_id   = kwargs.get('sc_id',None)
    sc_col  = kwargs.get('sc_col',None)
    sc_keys = kwargs.get('sc_keys',None)

    is_heat = False
    if display == 'Heat':
        is_heat = True

    ###---------------CONSTRUCT COLUMN LABELS---------------###

    # Validate column labels
    for label in (data1_col, data2_col):
        if label not in df.keys():
            raise ValueError(f'Field data "{label}" not found in data.')

    data1_str = data_string(data1_col)
    data2_str = data_string(data2_col)
    unit1 = df.attrs['units'].get(data1_col, None)
    unit2 = df.attrs['units'].get(data2_col, None)

    data1_label = create_label(data1_str, unit=unit1, data_name=data1_name)
    data2_label = create_label(data2_str, unit=unit2, data_name=data2_name)
    title_str = f'Comparing ${data1_str}$ and ${data2_str}$' if brief_title is None else brief_title

    if compressions is not None:
        B_imf, B_msh, _ = load_compression_ratios(compressions)


    ###---------------FILTERS DATA BY SPACECRAFT---------------###
    if sc_id and sc_col:
        filter_by_spacecraft(df, sc_col, sc_id)
    elif sc_id:
        raise ValueError('To filter by spacecraft, argument needed for "sc_col" and "sc_id".')


    ###---------------CREATES FIGURE WITH DESIRED SUBPLOTS---------------###
    if hist_diff:
        fig, (ax_main, ax_hist) = plt.subplots(
            nrows=1, ncols=2, figsize=(22, 8)
        )
    else:
        fig, ax_main = plt.subplots()

    line_color=black

    ###---------------PLOTS MAIN SCATTER/HEAT DATA---------------###
    if display == 'Scatter':
        ax_main.scatter(df[data2_col], df[data1_col], c='b', s=scat_size)

    # elif display == 'Scatter_grad_time':
    #     t = datetime_to_cdf_epoch(df.index)
    #     norm = plt.Normalize(t.min(), t.max())
    #     cmap = plt.get_cmap('plasma')
    #     ax_main.scatter(df[data2_col], df[data1_col], c=t, cmap=cmap, norm=norm, s=scat_size, label='Blue to yellow in time')

    elif display == 'Scatter_grad_sc':
        for id_value, group in df.groupby(sc_col):
            id_label = f'{sc_keys[id_value]} ({int(id_value)})' if sc_keys else f'ID {int(id_value)}'
            ax_main.scatter(group[data2_col], group[data1_col], label=f'{id_label}', s=scat_size)

    elif display == 'Scatter_grad_quality':
        if quality_col is not None and quality_col in df:
            for id_value, group in df.groupby(quality_col):
                id_label = f'{quality_labels[id_value]} quality' if quality_labels else f'{id_value}/4 quality'
                id_label += f': {len(group)/len(df)*100:.1f}%'
                ax_main.scatter(group[data2_col], group[data1_col], label=f'{id_label}', s=scat_size)
        else:
            raise ValueError(f'Column "{quality_col}" is not valid.')

    elif display == 'Scatter_grad_complexity':
        if complexity_col is not None and complexity_col in df:
            colors = [blue,'r']
            for i, (id_value, group) in enumerate(df.groupby(complexity_col)):
                id_label = f'{complexity_labels[id_value]}' if complexity_labels else f'{id_value}/1 complexity'
                id_label += f': {len(group)/len(df)*100:.1f}%'
                ax_main.scatter(group[data2_col], group[data1_col], label=f'{id_label}', s=scat_size, c=colors[i])
        else:
            raise ValueError(f'Column "{complexity_col}" is not valid.')

    elif display == 'Heat':
        if hasattr(bin_width, '__len__') and len(bin_width) == 2:
            n_bins = (calculate_bins(df[data2_col],bin_width[0]), calculate_bins(df[data1_col],bin_width[1]))
        else:
            n_bins = (calculate_bins(df[data2_col],bin_width), calculate_bins(df[data1_col],bin_width))
        h = ax_main.hist2d(df[data2_col], df[data1_col], bins=n_bins, norm=mpl.colors.LogNorm(), cmap='hot')

        cbar = fig.colorbar(h[3], ax=ax_main)
        cbar.ax.tick_params(colors=black)
        cbar.set_label('Number of Points', color=black)
        cbar.outline.set_edgecolor(black)

        line_color = 'w'

    else:
        raise ValueError(f'"{display}" not valid display mode.')

    ax_main.axline((0, 0), slope=1, color=line_color, label='y=x', lw=2, ls=':')
    if compressions is not None:

        ax_main.plot(B_imf, B_msh, color='cyan', label='Kobel Threshold', lw=2)

        limit = B_msh[-1]/B_imf[-1]
        limit_text = f'y≃{limit:.2f}x'

        extreme_times = are_points_above_line(B_imf, B_msh, df[data2_col], df[data1_col])
        total_text   = f'{len(df):,}'
        num_text     = f'{np.sum(extreme_times):,}'
        perc_text    = f'{np.sum(extreme_times)/len(df)*100:.2f}%'

        if contam_info:
            ax_main.plot([], [], ' ', label=f'\n{total_text}\n{num_text}\n{perc_text}')

        df_extreme = df[extreme_times].copy()
        extreme_days = np.unique(df_extreme.index.date)

        print(f'Number of unique days:   {len(extreme_days):,}')
        print(f'Length of df:            {total_text}')
        print(f'Limit:                   {limit_text}')
        print(f'Number above threshold:  {num_text}')
        print(f'Percent above threshold: {perc_text}')

    ax_main.set_xlabel(data2_label, c=black)
    ax_main.set_ylabel(data1_label, c=black)

    if best_fit:
        m, y0, _ = straight_best_fit(df[data2_col], df[data1_col], name=title_str)
        ax_main.axline((0,y0), slope=m, c='magenta', label=f'Best Fit: {m:.3f}x+{y0:.3f} {unit2}', lw=2.5, ls='--') # Best fit
        #ax_main.axline((0, 0), slope=-1, color='#F28500', label='y=-x', lw=2.5)

    if hist_diff:

        df['diff'] = difference_columns(df, data1_col, data2_col)
        x_axis_label = f'${data1_str}$ - ${data2_str}$ [{unit1}]'

        n_bins = calculate_bins(df['diff'],hist_bin_width)
        counts, bins, _ = ax_hist.hist(
            df['diff'], bins=n_bins,
            alpha=1.0, color='b', edgecolor='grey'
        )
        if fit_gaus:
            plot_gaussian(ax_hist,counts,bins,c='w',ls='-',name='Frequency Histogram')

        ax_hist.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

        if perc_filter is not None:
            perc_range=(np.percentile(df['diff'], perc_filter[0]), np.percentile(df['diff'], perc_filter[1]))
            ax_hist.set_xlim(perc_range)

        add_legend(fig, ax_hist, legend_on=want_legend, heat=is_heat)
        ax_hist.set_xlabel(x_axis_label, c=black)
        ax_hist.set_ylabel('Frequency', c=black)
        add_figure_title(fig,'Frequency Histogram of differences',ax=ax_hist)

    ###---------------LABELLING AND FINISHING TOUCHES---------------###
    add_legend(fig, ax_main, legend_on=want_legend, heat=is_heat)
    add_figure_title(fig, black, title_str, ax=ax_main)
    dark_mode_fig(fig,black,white,is_heat)
    plt.tight_layout();

    save_figure(fig)
    plt.show()
    plt.close()

def compare_columns_and_datasets(df1, df2, data1_col, data2_col, **kwargs):
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
    df1 = df1.copy()
    df1 = df1[np.isfinite(df1[data2_col]) & np.isfinite(df1[data1_col])]
    df2 = df2.copy()
    df2 = df2[np.isfinite(df2[data2_col]) & np.isfinite(df2[data1_col])]

    best_fit        = kwargs.get('stats',False)
    display         = kwargs.get('display','Scatter')
    bin_width       = kwargs.get('bin_width',None)
    scat_size       = kwargs.get('scatter_size',0.4)
    want_legend     = kwargs.get('want_legend',True)
    compressions    = kwargs.get('compressions',None)
    contam_info     = kwargs.get('contam_info',False)

    data1_name    = kwargs.get('data1_name',None)
    data2_name    = kwargs.get('data2_name',None)
    title1        = kwargs.get('title1',None)
    title2        = kwargs.get('title2',None)

    quality_col    = kwargs.get('quality_col',None)
    quality_labels = kwargs.get('quality_labels',None)

    complexity_col    = kwargs.get('complexity_col',None)
    complexity_labels = kwargs.get('complexity_labels',None)

    sc_id   = kwargs.get('sc_id',None)
    sc_col  = kwargs.get('sc_col',None)
    sc_keys = kwargs.get('sc_keys',None)

    is_heat = False
    if display == 'Heat':
        is_heat = True

    ###---------------CONSTRUCT COLUMN LABELS---------------###

    # Validate column labels
    for label in (data1_col, data2_col):
        if label not in df1.keys():
            raise ValueError(f'Field data "{label}" not found in data.')

    data1_str = data_string(data1_col)
    data2_str = data_string(data2_col)
    unit1 = df1.attrs['units'].get(data1_col, None)
    unit2 = df1.attrs['units'].get(data2_col, None)

    data1_label = create_label(data1_str, unit=unit1, data_name=data1_name)
    data2_label = create_label(data2_str, unit=unit2, data_name=data2_name)

    if compressions is not None:
        B_imf, B_msh, _ = load_compression_ratios(compressions)


    ###---------------FILTERS DATA BY SPACECRAFT---------------###
    if sc_id and sc_col:
        filter_by_spacecraft(df1, sc_col, sc_id)
        filter_by_spacecraft(df2, sc_col, sc_id)
    elif sc_id:
        raise ValueError('To filter by spacecraft, argument needed for "sc_col" and "sc_id".')


    ###---------------CREATES FIGURE WITH DESIRED SUBPLOTS---------------###
    fig, axs = plt.subplots(
        nrows=1, ncols=2, figsize=(20, 6)
    )

    line_color=black

    ###---------------PLOTS MAIN SCATTER/HEAT DATA---------------###
    for df, ax, brief_title in zip((df1,df2), axs, (title1,title2)):
        if display == 'Scatter':
            ax.scatter(df[data2_col], df[data1_col], c='b', s=scat_size)

        elif display == 'Scatter_grad_time':
            t = datetime_to_cdf_epoch(df.index)
            norm = plt.Normalize(t.min(), t.max())
            cmap = plt.get_cmap('plasma')
            ax.scatter(df[data2_col], df[data1_col], c=t, cmap=cmap, norm=norm, s=scat_size, label='Blue to yellow in time')

        elif display == 'Scatter_grad_sc':
            for id_value, group in df.groupby(sc_col):
                id_label = f'{sc_keys[id_value]} ({int(id_value)})' if sc_keys else f'ID {int(id_value)}'
                ax.scatter(group[data2_col], group[data1_col], label=f'{id_label}', s=scat_size)

        elif display == 'Scatter_grad_quality':
            if quality_col is not None and quality_col in df:
                for id_value, group in df.groupby(quality_col):
                    id_label = f'{quality_labels[id_value]} quality' if quality_labels else f'{id_value}/4 quality'
                    id_label += f': {len(group)/len(df)*100:.1f}%'
                    ax.scatter(group[data2_col], group[data1_col], label=f'{id_label}', s=scat_size)
            else:
                raise ValueError(f'Column "{quality_col}" is not valid.')

        elif display == 'Scatter_grad_complexity':
            if complexity_col is not None and complexity_col in df:
                colors = [blue,'r']
                for i, (id_value, group) in enumerate(df.groupby(complexity_col)):
                    id_label = f'{complexity_labels[id_value]}' if complexity_labels else f'{id_value}/1 complexity'
                    id_label += f': {len(group)/len(df)*100:.1f}%'
                    ax.scatter(group[data2_col], group[data1_col], label=f'{id_label}', s=scat_size, c=colors[i])
            else:
                raise ValueError(f'Column "{complexity_col}" is not valid.')

        elif display == 'Heat':
            if hasattr(bin_width, '__len__') and len(bin_width) == 2:
                n_bins = (calculate_bins(df[data2_col],bin_width[0]), calculate_bins(df[data1_col],bin_width[1]))
            else:
                n_bins = (calculate_bins(df[data2_col],bin_width), calculate_bins(df[data1_col],bin_width))
            h = ax.hist2d(df[data2_col], df[data1_col], bins=n_bins, norm=mpl.colors.LogNorm(), cmap='hot')

            cbar = fig.colorbar(h[3], ax=ax)
            cbar.ax.tick_params(colors=black)
            cbar.outline.set_edgecolor(black)

            line_color = 'w'

        else:
            raise ValueError(f'"{display}" not valid display mode.')

        ax.axline((0, 0), slope=1, color=line_color, label='y=x', lw=2, ls=':')
        if compressions is not None:

            ax.plot(B_imf, B_msh, color='cyan', label='Kobel Threshold', lw=2, ls=':')

            limit = B_msh[-1]/B_imf[-1]
            limit_text = f'y≃{limit:.2f}x'

            extreme_times = are_points_above_line(B_imf, B_msh, df[data2_col], df[data1_col])
            total_text   = f'{len(df):,}'
            num_text     = f'{np.sum(extreme_times):,}'
            perc_text    = f'{np.sum(extreme_times)/len(df)*100:.2f}%'

            if contam_info:
                ax.plot([], [], ' ', label=f'\n{total_text}\n{num_text}\n{perc_text}')

            df_extreme = df[extreme_times].copy()
            extreme_days = np.unique(df_extreme.index.date)

            print(f'Number of unique days:   {len(extreme_days):,}')
            print(f'Length of df:            {total_text}')
            print(f'Limit:                   {limit_text}')
            print(f'Number above threshold:  {num_text}')
            print(f'Percent above threshold: {perc_text}')

        ax.set_xlabel(data2_label, c=black)

        if best_fit:
            m, y0, _ = straight_best_fit(df[data2_col], df[data1_col], name=brief_title)
            ax.axline((0,y0), slope=m, c='magenta', label=f'Best Fit: {m:.3f}x+{y0:.3f} {unit2}', lw=2.5, ls='--') # Best fit


        ###---------------LABELLING AND FINISHING TOUCHES---------------###
        add_legend(fig, ax, legend_on=want_legend, heat=is_heat)
        add_figure_title(fig, brief_title, ax=ax)


    axs[0].set_ylabel(data1_label, c=black)
    cbar.set_label('Number of Points', color=black)

    dark_mode_fig(fig,black,white,is_heat)
    plt.tight_layout();
    save_figure(fig)
    plt.show()
    plt.close()


def investigate_difference(df, data1_col, data2_col, ind_col, **kwargs):
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
    df = df.copy()
    df = df[np.isfinite(df[data2_col]) & np.isfinite(df[data1_col])]

    display     = kwargs.get('display','Scatter')
    diff_type   = kwargs.get('diff_type','Absolute')
    bin_width   = kwargs.get('bin_width',None)
    scat_size   = kwargs.get('scatter_size',0.4)
    want_legend = kwargs.get('want_legend',True)

    x_data_name = kwargs.get('x_data_name',None)
    brief_title = kwargs.get('brief_title',None)

    polarity    = kwargs.get('polarity','Both')
    abs_x       = kwargs.get('abs_x',False)

    sc_id   = kwargs.get('sc_id',None)
    sc_col  = kwargs.get('sc_col',None)

    is_heat = False
    if display == 'Heat':
        is_heat = True

    ###---------------CONSTRUCT COLUMN LABELS---------------###
    x_label = ind_col
    y_label='diff'
    y1_label = data1_col
    y2_label = data2_col

    # Validate column labels
    for label in (x_label, y1_label, y2_label):
        if label not in df.keys():
            raise ValueError(f'Field data "{label}" not found in data.')

    x_data_str = data_string(x_label)
    x_unit = df.attrs['units'].get(x_label, None)
    x_axis_label = create_label(x_label,x_unit,x_data_name,True)
    x_title_label = create_label(x_label,None,x_data_name,True)

    y1_data_str = data_string(y1_label)
    y2_data_str = data_string(y2_label)
    y_unit = df.attrs['units'].get(y1_label, None)

    ###---------------FILTERS DATA BY SPACECRAFT---------------###
    if sc_id and sc_col:
        filter_by_spacecraft(df, sc_col, sc_id)
    elif sc_id:
        raise ValueError('To filter by spacecraft, argument needed for "sc_col" and "sc_id".')

    if abs_x:
        df[x_label] = np.abs(df[x_label])
        x_data_str = f'|{x_data_str}|'

    ###---------------CALCULATES DATA FOR DIFFERENCE TYPE---------------###
    if 'Absolute' in diff_type:
        df[y_label] = difference_columns(df, y1_label, y2_label)

        y_axis_label = f'${y1_data_str}$ - ${y2_data_str}$ [{y_unit}]'
        y_title_label = f'${y1_data_str}$ - ${y2_data_str}$'

    elif 'Relative' in diff_type:
        df[y_label] = difference_columns(df, y1_label, y2_label) / df[y2_label]
        df[y_label].replace([np.inf, -np.inf], np.nan, inplace=True)

        y_axis_label = f'(${y1_data_str}$ - ${y2_data_str}$) / ${y2_data_str}$'
        y_title_label = y_axis_label

    else:
        raise ValueError(f'"{diff_type}" not valid difference mode.')

    if 'Modulus' in diff_type:
        df[y_label] = np.abs(df[y_label])
    if polarity != 'Both':
        df = filter_sign(df, y_label, polarity)

    ###---------------CREATES FIGURE WITH DESIRED SUBPLOTS---------------###
    fig, ax_main = plt.subplots()
    line_color = black

    ###---------------PLOTS MAIN SCATTER/HEAT DATA---------------###
    if display == 'Scatter':
        ax_main.scatter(df[x_label], df[y_label], c=blue, s=scat_size)

    elif display == 'Heat':
        if hasattr(bin_width, '__len__') and len(bin_width) == 2:
            n_bins = (calculate_bins(df[x_label],bin_width[0]), calculate_bins(df[y_label],bin_width[1]))
        else:
            n_bins = (calculate_bins(df[x_label],bin_width), calculate_bins(df[y_label],bin_width))
        h = ax_main.hist2d(df[x_label], df[y_label], bins=n_bins, norm=mpl.colors.LogNorm(), cmap='hot')

        cbar = fig.colorbar(h[3], ax=ax_main)
        cbar.ax.tick_params(colors=black)
        cbar.set_label('Number of Points', color=black)
        cbar.outline.set_edgecolor(black)

        line_color = 'w'

    else:
        raise ValueError(f'"{display}" not valid display mode.')

    ax_main.axhline(y=0, color=line_color, label='y=0', lw=2, ls=':')

    ###---------------LABELLING AND FINISHING TOUCHES---------------###
    ax_main.set_xlabel(x_axis_label, c=black)
    ax_main.set_ylabel(y_axis_label, c=black)

    if 'r_x' in x_label or 'r_y' in x_label:
        plt.gca().invert_xaxis()
    if diff_type == 'Relative':
        ax_main.set_yscale('symlog', linthresh=1)  # linthresh defines the linear threshold around 0
    elif diff_type == 'Modulus Relative':
        ax_main.set_yscale('log')

    add_legend(fig, ax_main, legend_on=want_legend, heat=is_heat)
    dark_mode_fig(fig, black, white, is_heat)
    add_figure_title(fig, black, brief_title, x_title_label, y_title_label, ax=ax_main)
    plt.tight_layout();
    save_figure(fig)
    plt.show()
    plt.close()


def compare_dataframes(df1, df2, data_x, data_y, **kwargs):
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
                differences = difference_columns_dfs(df1, df2, col, col_omni)

                x_axis_label = create_label(None,unit=x_unit,data_name='C1 - OMNI')

            elif diff_type == 'Relative':
                differences = difference_columns_dfs(df1, df2, col, col_omni) / df2[col_omni]
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