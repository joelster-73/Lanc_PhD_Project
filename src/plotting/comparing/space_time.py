# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:53:22 2025

@author: richarj2
"""

import numpy as np
import pandas as pd
from datetime import datetime

from scipy.signal import find_peaks

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

from ..config import black, white
from ..formatting import add_figure_title, dark_mode_fig
from ..utils import save_figure
from ..space_time import plot_orbit, plot_time_distribution

def plot_compare_datasets_space(df1, df2, plane='x-rho', coords='GSE', **kwargs):

    df1_name = kwargs.get('df1_name',None)
    df2_name = kwargs.get('df2_name',None)

    kwargs['equal_axes'] = False
    kwargs['want_legend'] = False
    kwargs['nose_text'] = 'BS'
    kwargs['models'] = 'Median BS'

    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    for i, (df, name, ax) in enumerate(zip((df1, df2), (df1_name, df2_name), axes)):

        _, axis, cbar = plot_orbit(df, plane=plane, coords=coords, fig=fig, ax=ax, return_objs=True, brief_title=name, **kwargs)
        if i==0:
            cbar.set_label(None)
        elif i==1:
            axis.set_ylabel(None)

    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

def plot_compare_datasets_time(df1, df2, **kwargs):

    df1_name = kwargs.get('df1_name',None)
    df2_name = kwargs.get('df2_name',None)

    df1_colour = kwargs.get('df1_colour','k')
    df2_colour = kwargs.get('df2_colour','k')

    fig, axes = plt.subplots(1, 2, figsize=(14,4))

    for i, (df, name, colour, ax) in enumerate(zip((df1, df2), (df1_name, df2_name), (df1_colour, df2_colour), axes)):

        _, axis = plot_time_distribution(df, fig=fig, ax=ax, return_objs=True, colour=colour, brief_title=name, **kwargs)
        if i==1:
            axis.set_ylabel(None)

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
    data = pd.read_csv(url, sep='\s+', names=columns, comment='*')

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

        date_range = pd.date_range(start=first_time, end=last_time, freq='min')
        df_blank = pd.DataFrame(index=date_range)
        mask_total = pd.Series(False, index=df_blank.index)
        for start, end in high_activity:
            mask_total |= df_blank.index.to_series().between(pd.Timestamp(start), pd.Timestamp(end))
        high_activity_percentage = np.sum(mask_total) / len(mask_total) * 100

        if len(dfs)>1:
            plot_label = f'{label}: {total:.1f}%'
        else:
            plot_label = f'High Activity: {total:.1f}%\nExpected:      {high_activity_percentage:.1f}%'

        ax.plot(bin_midpoints, counts, color=colour, lw=2, marker=marker, markersize=7, markerfacecolor='w', markeredgecolor=colour, label=plot_label)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.1f}%'))

        date_range = pd.date_range(start=first_time, end=last_time, freq='min')
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

    add_figure_title(fig, black, 'Cluster Solar Wind Yearly Data Counts')
    #plt.grid()
    dark_mode_fig(fig,black,white)
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()

