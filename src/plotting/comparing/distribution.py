# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:53:22 2025

@author: richarj2
"""

import numpy as np
import pandas as pd

from datetime import datetime

from scipy.signal import find_peaks

import matplotlib as mpl
import matplotlib.pyplot as plt


from magnetosheath import bs_boundaries


def plot_compare_datasets_space(df1, df2, rx_col, ryz_col, df_omni, **kwargs):

    bin_width    = kwargs.get('bin_width',0.1)
    rx_name      = kwargs.get('rx_name',None)
    ryz_name     = kwargs.get('ryz_name',None)
    df1_name     = kwargs.get('df1_name','Jelínek Bow shock')
    df2_name     = kwargs.get('df2_name','GRMB Database')

    n_cols = 2
    n_rows = 1
    fig, axs = plt.subplots(
        n_rows, n_cols, sharex='row', figsize=(n_cols*6, n_rows*5)
    )

    units = df1.attrs['units']

    ###-------------------ORBITS-------------------###
    first_row = axs

    x_axis_label = create_label(rx_col,None,rx_name,True,units)
    y_axis_label = create_label(ryz_col,None,ryz_name,True,units)

    for ax, df in zip(first_row, (df1, df2)):

        # Data outside buffered bowshock
        n_bins = (calculate_bins(df[rx_col],bin_width),calculate_bins(df[ryz_col],bin_width))
        h = ax.hist2d(df[rx_col], df[ryz_col], bins=n_bins, norm=mpl.colors.LogNorm(), cmap='hot')

        # Median p and v Bow shock
        pressures = df_omni[df_omni.index.isin(df.index)]['p_flow']
        pressures = pressures[~np.isnan(pressures)]
        velocities = df_omni[df_omni.index.isin(df.index)]['v_x_GSE']
        velocities = velocities[~np.isnan(velocities)]

        bs_jel = bs_boundaries('jelinek', Pd=np.median(pressures), vsw=np.median(velocities))
        #bs_jel = bs_boundaries('jelinek', Pd=np.median(pressures))

        bs_x_coords = bs_jel.get('x')
        bs_y_coords = bs_jel.get('y')
        y_neg = bs_jel.get('y')<0
        # Stand-off is in -ve quadrant

        bd_R0 = bs_jel.get('R0')
        alpha = bs_jel.get('alpha')

        # In the x-y plane
        ax.plot(bs_x_coords[y_neg], -bs_y_coords[y_neg], linestyle='-', color='lime', lw=3)

        stand_off_x = bd_R0*np.cos(alpha)
        stand_off_y = bd_R0*np.sin(alpha)
        ax.scatter(stand_off_x, -stand_off_y, c='lime')
        ax.text(stand_off_x - 0.5, -stand_off_y + 0.5, f'$R_0$ = {bd_R0:.1f} $R_E$, {np.degrees(alpha):.1f}$^\\circ$', fontsize=10, color='lime')
        ax.plot([0,stand_off_x],[0,-stand_off_y],ls=':',c='w')
        ax.set_xlabel(x_axis_label,c=black)

        cbar = fig.colorbar(h[3], ax=ax)
        cbar.ax.tick_params(colors=black)
        cbar.outline.set_edgecolor(black)
        ax.set_facecolor('k')

        #ax.scatter(0, 0, color='b', marker='o', s=1600)  # Earth

        create_half_circle_marker(ax, center=(0, 0), radius=1, full=False)

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        add_legend(fig, ax, heat=True)

    first_row[0].invert_xaxis()
    first_row[0].set_ylabel(y_axis_label,c=black)

    add_figure_title(fig, df1_name, ax=first_row[0])
    add_figure_title(fig, df2_name, ax=first_row[1])


    #dark_mode_fig(fig,black,white,heat=True)
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

def plot_compare_datasets_with_activity(*dfs, **kwargs):


    df_colours   = kwargs.get('df_colours',('g','b'))
    df_names     = kwargs.get('df_names',('Jelínek','GRMB'))
    df_markers   = kwargs.get('df_markers',('o','s'))
    plot_when    = kwargs.get('plot_when','middle')

    # Load the sunspot data from SILSO (adjust the file path or URL as needed)
    url = 'https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.txt'
    columns = ['year', 'month', 'decimal_year', 'sunspot_count', 'standard_deviation', 'number_of_observations', 'provisional']
    data = pd.read_csv(url, delim_whitespace=True, names=columns, comment='*')

    data = data[(data['year'] >= 2000) & (data['year'] <= 2025)]
    data['date'] = pd.to_datetime(data[['year', 'month']].assign(day=1))
    data['smoothed_sunspot_count'] = data['sunspot_count'].rolling(window=13, center=True).mean()
    data = data.dropna(subset=['smoothed_sunspot_count'])

    # Identify maxima and minima using find_peaks
    maxima_indices, _ = find_peaks(data['smoothed_sunspot_count'], distance=100)
    minima_indices, _ = find_peaks(-data['smoothed_sunspot_count'], distance=100)

    # To extend the peak-finding
    if True:
        maxima_indices[-1] = np.where(~np.isnan(data['smoothed_sunspot_count']))[0][-1]

    maxima = data.iloc[maxima_indices]
    minima = data.iloc[minima_indices]
    combined = pd.concat([maxima[['date', 'smoothed_sunspot_count']], minima[['date', 'smoothed_sunspot_count']]])
    extrema = combined.sort_values(by='date').reset_index(drop=True)


    # Pair maxima with the nearest following minima
    midpoints = []
    for i in range(len(extrema)-1):
        current_ext = extrema.iloc[i]
        next_ext = extrema.iloc[i+1]
        halfway_count = (current_ext['smoothed_sunspot_count'] + next_ext['smoothed_sunspot_count']) / 2
        subset = data[(data['date'] >= current_ext['date']) & (data['date'] <= next_ext['date'])].copy()
        subset.loc[:, 'abs_diff'] = np.abs(subset['smoothed_sunspot_count'] - halfway_count)
        midpoint_row = subset.loc[subset['abs_diff'].idxmin()]
        midpoints.append(midpoint_row['date'])

    high_activity = ((datetime(2000,1,1),midpoints[0]),
                     (midpoints[1], midpoints[2]),
                     (midpoints[3], datetime(2025,3,1)))


    # Plot the data
    fig, ax = plt.subplots()

    for minimum in minima['date']:
        ax.axvline(x=minimum, c='k', ls=':')
    for maximum in maxima['date']:
        ax.axvline(x=maximum, c='k', ls='--')

    for active in high_activity:
        ax.axvspan(active[0], active[1], color='deeppink', alpha=0.16, zorder=0)

    for df, colour, label, marker in zip(dfs,df_colours,df_names,df_markers):
        series = df.index.year.to_numpy()
        series = series[~np.isnan(series)]

        bin_edges = range(np.min(series), np.max(series) + 2, 1)
        counts, _ = np.histogram(series, bins=bin_edges)
        counts = counts.astype(np.float64)
        counts = (counts/np.sum(counts)) * 100

        if plot_when == 'median':

            bin_midpoints = [
                np.median(df[df.index.year == year].index.astype('int64') / 10**9)
                for year in bin_edges[:-1]
            ]
            bin_midpoints = [pd.to_datetime(median, unit='s') for median in bin_midpoints]

        elif plot_when == 'middle':

            bin_midpoints = [datetime(year,7,2) for year in bin_edges[:-1]]

        elif plot_when == 'start':

            bin_midpoints = [datetime(year,1,1) for year in bin_edges[:-1]]

        mask = pd.Series(False, index=df.index)
        for start, end in high_activity:
            mask |= df.index.to_series().between(pd.Timestamp(start), pd.Timestamp(end))
        total = np.sum(mask) / len(mask) * 100

        first_time = df.index[0]
        last_time = df.index[-1]

        date_range = pd.date_range(start=first_time, end=last_time, freq='T')
        df_blank = pd.DataFrame(index=date_range)
        mask_total = pd.Series(False, index=df_blank.index)
        for start, end in high_activity:
            mask_total |= df_blank.index.to_series().between(pd.Timestamp(start), pd.Timestamp(end))
        high_activity_percentage = np.sum(mask_total) / len(mask_total) * 100

        if len(dfs)>1:
            plot_label = f'{label}: {total:.1f}%'
        else:
            plot_label = f'High Activity: {total:.1f}%\nExpected:      {high_activity_percentage:.1f}%'

        ax.plot(bin_midpoints, counts, color=colour, lw=2,
                marker=marker, markersize=7, markerfacecolor='w', markeredgecolor=colour, label=plot_label)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.1f}%'))

        date_range = pd.date_range(start=first_time, end=last_time, freq='T')
        df_blank = pd.DataFrame(index=date_range)
        mask_total = pd.Series(False, index=df_blank.index)
        for start, end in high_activity:
            mask_total |= df_blank.index.to_series().between(pd.Timestamp(start), pd.Timestamp(end))
        high_activity_percentage = np.sum(mask_total) / len(mask_total) * 100

        print(f'{first_time} - {last_time} - {high_activity_percentage:.2f}% - ({label})')

    ax.set_xlabel('Year')
    ax.set_ylabel('Dataset Yearly Proportion')
    ax.legend(loc='upper left', facecolor='w', frameon=True)
    ax.set_xlim(high_activity[0][0],high_activity[-1][-1])


    ax2 = ax.twinx()

    ax2.yaxis.set_ticks_position('right')
    ax2.spines['right'].set_visible(True)

    ax2.scatter(data['date'], data['sunspot_count'], alpha=0.6, color='darkgray',s=14, edgecolor='gray', label='Monthly Sunspot Count')
    ax2.plot(data['date'], data['smoothed_sunspot_count'], color='r', lw=2, label='13-Month Smoothed')
    #ax2.tick_params(axis='y', labelcolor='#F28500')
    ax2.set_ylabel('SIDC Sunspot Count')
    ax2.legend(facecolor='w')

    add_figure_title(fig, 'Cluster Solar Wind Yearly Data Counts')
    #plt.grid()
    dark_mode_fig(fig,black,white)
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()



def plot_compare_datasets_years(df1, df2, **kwargs):

    show_apogee  = kwargs.get('show_apogee',False)
    full_cluster = kwargs.get('full_cluster',None)
    show_months  = kwargs.get('show_months',False)
    df1_colour   = kwargs.get('df1_colour','g')
    df2_colour   = kwargs.get('df2_colour','b')
    df1_name     = kwargs.get('df1_name','Jelínek Bow shock')
    df2_name     = kwargs.get('df2_name','GRMB Database')
    print_data   = kwargs.get('print_data',False)

    n_cols = 2
    n_rows = 1 + int(show_months)
    fig, axs = plt.subplots(
        n_rows, n_cols, sharex='row', figsize=(n_cols*6, n_rows*4)
    )


    ###-------------------PLOT YEARS-------------------###
    first_row = axs[0] if n_rows > 1 else axs


    add_figure_title(fig, df1_name, ax=first_row[0])
    add_figure_title(fig, df2_name, ax=first_row[1])

    if show_apogee and full_cluster is not None:

        r_mag = np.sqrt(full_cluster['r_x_GSE']**2+full_cluster['r_y_GSE']**2+full_cluster['r_z_GSE']**2)

        window_size = 7 * 24 * 60
        apogees = r_mag.rolling(window=window_size, center=True).max()
        apogee_decimal_year = datetime_to_decimal_year_vectorised(apogees.index)

    for ax1, df, colour, name in zip(first_row, (df1, df2), (df1_colour,df2_colour), (df1_name, df2_name)):

        series = df.index.year.to_numpy()
        series = series[~np.isnan(series)]

        bin_edges = range(np.min(series),np.max(series)+2,1)

        counts, bins, _ = ax1.hist(
            series, bins=bin_edges,
            alpha=1.0, color=colour, edgecolor='grey'
        )

        ax1.set_xlabel('Year',c=black)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

        if show_apogee and full_cluster is not None:
            ax_twin = ax1.twinx()
            ax_twin.plot(apogee_decimal_year,apogees,color='r', lw=2.25)
        else:
            for year in (2003,2013,2023):
                ax1.axvline(x=year,ls='--',lw=2.5,c='k')

        ax1.plot([],[],' ',label=f'{len(df):,} mins')

        add_figure_title(fig, name, ax=ax1)
        add_legend(fig, ax1, loc='upper right', edge_col='w')

        if print_data:
            year_counts = df.index.year.value_counts().sort_index()
            total_rows = year_counts.sum()

            summary_df = pd.DataFrame({
                'year': year_counts.index.astype(str),
                'row_count': year_counts.values,
                'proportion': year_counts.values / total_rows * 100
            })

            total_row = pd.DataFrame({
                'year': ["Total"],
                'row_count': [total_rows],
                'proportion': [100]
            }, index=[100])
            summary_df = pd.concat([summary_df, total_row])

            print(summary_df)

    first_row[0].set_ylabel('Count',c=black)
    ax_twin.set_ylabel(r'$C_1$ Apogee [$R_E$]',c='r')

    ###-------------------PLOT MONTHS-------------------###
    if show_months:
        second_row = axs[1]

        for ax2, df, colour in zip(second_row, (df1, df2), ('g','b')):

            series = df.index.month.to_numpy()
            series = series[~np.isnan(series)]

            bin_edges = range(np.min(series),np.max(series)+2,1)
            counts, bins, _ = ax2.hist(
                series, bins=bin_edges,
                alpha=1.0, color=colour, edgecolor='grey'
            )

            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            ax2.set_xticks(ticks=np.arange(1, 13), labels=month_names)
            ax2.tick_params(axis='x', labelrotation=90)  # Rotate x-tick labels 90° counterclockwise
            ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

        second_row[0].set_ylabel('Count',c=black)

    dark_mode_fig(fig,black,white)
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

def plot_dataset_years_months(df, **kwargs):

    df_colour    = kwargs.get('df_colour','g')
    show_apogee  = kwargs.get('show_apogee',False)
    full_cluster = kwargs.get('full_cluster',None)

    n_cols = 2
    n_rows = 1
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols*6, n_rows*4)
    )

    ###-------------------PLOT HISTOGRAMS-------------------###
    first_row = axs[0] if n_rows > 1 else axs

    years = df.index.year.to_numpy()
    years = years[~np.isnan(years)]

    months = df.index.month.to_numpy()
    months = months[~np.isnan(months)]

    for ax1, data, name in zip(first_row, (years,months), ('Years', 'Months')):

        bin_edges = range(np.min(data),np.max(data)+2,1)

        counts, bins, _ = ax1.hist(
            data, bins=bin_edges,
            alpha=1.0, color=df_colour, edgecolor='grey'
        )

        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))
        add_figure_title(fig, name, ax=ax1)

    ###--------------YEARS--------------###
    ax_year = first_row[0]
    ax_year.set_ylabel('Count',c=black)


    if show_apogee and full_cluster is not None:

        r_mag = np.sqrt(full_cluster['r_x_GSE']**2+full_cluster['r_y_GSE']**2+full_cluster['r_z_GSE']**2)

        window_size = 7 * 24 * 60
        apogees = r_mag.rolling(window=window_size, center=True).max()
        apogee_decimal_year = datetime_to_decimal_year_vectorised(apogees.index)

        ax_twin = ax_year.twinx()
        ax_twin.plot(apogee_decimal_year,apogees,color='r', lw=2.25, label=r'Apogee [$R_E$]')

        # Max and min apogees
        apo_max = np.max(apogees)
        apo_min = np.min(apogees)

        apo_max_time = datetime_to_decimal_year_vectorised(apogees.idxmax())
        apo_min_time = datetime_to_decimal_year_vectorised(apogees.idxmin())

        ax_twin.scatter(apo_max_time, apo_max, c='r', s=60, zorder=5)
        ax_twin.scatter(apo_min_time, apo_min, c='r', s=60, zorder=5)

        ax_twin.text(apo_max_time+1, apo_max-0.2, f'{apo_max:.2f}', color='r', ha='left', va='center')
        ax_twin.text(apo_min_time, apo_min+0.5, f'{apo_min:.2f}', color='r', ha='center', va='bottom')

        add_legend(fig,ax_twin,loc='upper right',edge_col='w')

    ###--------------MONTHS--------------###
    ax_month = first_row[1]

    percentages = (counts / counts.sum()) * 100
    for percentage, count, edge_left, edge_right in zip(percentages, counts, bin_edges[:-1], bin_edges[1:]):
        if percentage > 0:
            bin_centre = (edge_left + edge_right) / 2
            if percentage < 0.05:
                ax_month.text(bin_centre-0.1, count, f'{percentage:.2f}%', ha='center', va='bottom', fontsize=10)
            elif percentage < 10:
                ax_month.text(bin_centre, count, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)
            else:
                ax_month.text(bin_centre, count, f'{percentage:.2g}%', ha='center', va='bottom', fontsize=10)

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax_month.set_xticks(ticks=np.arange(1, 13))
    label_positions = np.arange(1, 13) + 0.5
    for pos, label in zip(label_positions, month_names):
        ax_month.text(pos, -0.05, label, ha='center', va='top', transform=ax_month.get_xaxis_transform())
    ax_month.set_xticklabels([])

    ax_month.plot([],[],' ',label=f'{len(df):,} mins')
    add_legend(fig, ax_month, loc='upper right', edge_col='w')

    ###--------------TIDYING--------------###
    dark_mode_fig(fig,black,white)
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()


def omni_distribution(df_omni):

    n_cols = 2
    n_rows = 1
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols*6, n_rows*4)
    )

    pressures = df_omni[df_omni['p_flow'].notna()]
    time_stamps = pressures.index

    ###-------------------PLOT YEARS-------------------###
    axs[0].set_title(f'{len(df_omni):,} mins in OMNI')

    series = time_stamps.year.to_numpy()

    bin_edges = range(np.min(series),np.max(series)+2,1)
    counts, bins, _ = axs[0].hist(
        series, bins=bin_edges,
        alpha=1.0, color='r', edgecolor='grey'
    )

    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('Count')
    axs[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

    ###-------------------PLOT MONTHS-------------------###
    axs[1].set_title(f'{len(pressures):,} mins with pressure')

    series = time_stamps.month.to_numpy()

    bin_edges = range(np.min(series),np.max(series)+2,1)
    counts, bins, _ = axs[1].hist(
        series, bins=bin_edges,
        alpha=1.0, color='r', edgecolor='grey'
    )

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    axs[1].set_xticks(ticks=np.arange(1, 13), labels=month_names)
    axs[1].tick_params(axis='x', labelrotation=90)  # Rotate x-tick labels 90° counterclockwise

    axs[1].set_xlabel('Month')
    axs[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))


    plt.tight_layout()
    save_figure(fig)
    plt.show()

def omni_distribution_years(df_omni, years=range(2014,2024)):

    n_cols = 2
    n_rows = len(years)//2
    fig, axs = plt.subplots(
        n_rows, n_cols, sharey=True, figsize=(n_cols*6, n_rows*4)
    )

    pressures = df_omni[df_omni['p_flow'].notna()]
    time_stamps = pressures.index

    for i, year in enumerate(years):

    ###-------------------PLOT MONTHS-------------------###
        ax = axs[i//2][i%2]
        ax.set_title(f'Counts for {year}')

        times = time_stamps[time_stamps.year==year]
        series = times.month.to_numpy()

        bin_edges = range(np.min(series),np.max(series)+2,1)
        counts, bins, _ = ax.hist(
            series, bins=bin_edges,
            alpha=1.0, color='r', edgecolor='grey'
        )

        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        ax.set_xticks(ticks=np.arange(1, 13), labels=month_names)
        ax.tick_params(axis='x', labelrotation=90)  # Rotate x-tick labels 90° counterclockwise

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))

    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

def plot_omni_parameters(df_omni, *columns, **kwargs):

    column_names          = kwargs.get('column_names',{})
    column_bin_widths     = kwargs.get('column_bin_widths',{})

    column_display_ranges     = kwargs.get('column_display_ranges',{})

    n_cols = 1
    n_rows = len(columns)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols*6, n_rows*4)
    )

    units = df_omni.attrs['units']

    ###-------------------PLOT PARAMETERS-------------------###

    for ax_row, column in zip(axs, columns):

        col_name      = column_names.get(column,None)
        bin_width     = column_bin_widths.get(column,0.25)

        x_unit = units[column]
        if x_unit == 'rad':
            x_unit = '°'
        x_axis_label = create_label(column,x_unit,col_name)
        display_range = column_display_ranges.get(column,None)


        ###-------------------PLOT COLUMNS-------------------###

        series = df_omni[column].to_numpy()
        series = series[~np.isnan(series)]

        if x_unit == '°':
            series = np.degrees(series)

        bin_edges = calculate_bins_edges(series,bin_width)
        counts, bins, _ = ax_row.hist(
            series, bins=bin_edges,
            alpha=1.0, color='r', edgecolor='grey'
        )

        ax_row.set_xlabel(x_axis_label)
        ax_row.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))
        add_legend(fig,ax_row,loc='upper right')

        ax_row.set_ylabel('Count')

        if display_range is not None:
            ax_row.set_xlim(display_range[0], display_range[-1])

    dark_mode_fig(fig,black,white)
    add_figure_title(fig, f'OMNI parameters ({len(df_omni):,} minutes)\n')
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()