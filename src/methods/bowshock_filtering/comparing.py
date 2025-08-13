# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 14:25:55 2025

@author: richarj2
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from ...analysing.calculations import calc_mean_error, calc_circular_mean_error, circular_standard_deviation, percentile_func
from ...analysing.kobel import load_compression_ratios, are_points_above_line

from ...plotting.comparing.parameter import compare_series
from ...plotting.formatting import add_legend
from ...plotting.utils import save_figure

from ...processing.dataframes import next_index


def compare_columns_with_compression(df, col1, col2, **kwargs):

    series1 = df.loc[:,col1]
    series2 = df.loc[:,col2]

    compressions    = kwargs.get('compressions',None)
    contam_info     = kwargs.get('contam_info',False)

    fig, ax, _ = compare_series(series1, series2, return_objs=True, **kwargs)

    if compressions is not None:
        B_imf, B_msh, _ = load_compression_ratios(compressions)

        ax.plot(B_imf, B_msh, c='cyan', lw=2, label='Kobel Threshold')

        limit = B_msh[-1]/B_imf[-1]
        limit_text = f'y≃{limit:.2f}x'

        extreme_times = are_points_above_line(B_imf, B_msh, series2, series1)
        total_text   = f'{len(df):,}'
        num_text     = f'{np.sum(extreme_times):,}'
        perc_text    = f'{np.sum(extreme_times)/len(df)*100:.2f}%'

        if contam_info:
            ax.plot([], [], ' ', label=f'\n{total_text}\n{num_text}\n{perc_text}')

        display         = kwargs.get('display','Scatter')
        want_legend     = kwargs.get('want_legend',True)
        is_heat = False
        if display == 'Heat':
            is_heat = True

        add_legend(fig, ax, legend_on=want_legend, heat=is_heat)

        print(f'Length of df:            {total_text}')
        print(f'Limit:                   {limit_text}')
        print(f'Number above threshold:  {num_text}')
        print(f'Percent above threshold: {perc_text}')


    save_figure(fig)
    plt.show()
    plt.close()

def extreme_diffs(df1, df2, df_merged, data_name, source1, source2, df_regions, **kwargs):

    filtering    = kwargs.get('filtering','Kobel')
    on_day       = kwargs.get('on_day',None)
    all_cluster  = kwargs.get('all_cluster',None)

    df_merged = df_merged.copy()

    col1 = f'{data_name}_{source1}'
    col2 = f'{data_name}_{source2}'

    if filtering == 'Abs':
        threshold  = kwargs.get('threshold',10)
        polarity   = kwargs.get('polarity','Positive')
        if polarity == 'Positive':
            extreme_times = df_merged[col1] - df_merged[col2] > threshold
        elif polarity == 'Both':
            extreme_times = np.abs(df_merged[col1] - df_merged[col2]) > threshold
        elif polarity == 'Negative':
            extreme_times = df_merged[col1] - df_merged[col2] < threshold
        else:
            raise ValueError(f'Invalid polarity choice: {polarity}')

    elif filtering == 'Rel':
        extreme_times = df_merged[col1] > threshold * df_merged[col2]

    elif filtering == 'Kobel':
        compressions = kwargs.get('compressions',None)
        if compressions is None:
            raise ValueError ('Need compressions directory to filter with Kobel.')
        B_imf, B_msh, _ = load_compression_ratios(compressions)
        extreme_times = are_points_above_line(B_imf, B_msh, df_merged[col2], df_merged[col1])

    df_extreme = df_merged[extreme_times].copy()
    del df_merged

    # Start of region 12s
    band_starts = df_regions.index[df_regions['loc_num']==12]
    if on_day is not None:
        target_date = pd.Timestamp(on_day).date()
        band_starts = band_starts[band_starts.to_series().dt.date == target_date]
    band_ends   = [next_index(df_regions, band_start) for band_start in band_starts]


    for band_start, band_end in zip(band_starts, band_ends):

        band_start = pd.Timestamp(band_start)
        band_end = pd.Timestamp(band_end)

        print(f'Start: {band_start}, End: {band_end}')

        data1_mean = calc_mean_error(df1[data_name], band_start, band_end)
        data2_mean = calc_mean_error(df2[data_name], band_start, band_end)

        print(f'{source1} mean: ${data1_mean:L}$, {source2} mean: ${data2_mean:L}$')

        in_interval = (df_extreme.index >= band_start) & (df_extreme.index <= band_end)
        number_in_interval = int(np.sum(in_interval))
        print(f'Mins in interval: {number_in_interval}')

        extreme_minutes = df_extreme[in_interval]
        if number_in_interval>0:
            print('Extreme Minutes C1 |B|:\n',extreme_minutes[col1])
            if all_cluster is not None:
                pd.set_option('display.max_rows', None)
                if all_cluster == 'all':
                    data1_hours = df1[(df1.index >= band_start) & (df1.index <= band_end)]
                    print('All Cluster |B|:\n',data1_hours[data_name])
                else:
                    time_start, time_end = pd.Timestamp(all_cluster[0]), pd.Timestamp(all_cluster[1])
                    data1_hours = df1[(df1.index >= time_start) & (df1.index <= time_end)]
                    print('All Cluster |B|:\n',data1_hours[data_name])
        print()



def compare_datasets(df1, df2, df_omni, *columns, **kwargs):

    column_names = kwargs.get('column_names',{})
    units = df1.attrs['units']

    df_comparison_years = pd.DataFrame(columns=['Years','Counts_1','Percent_1','Counts_2','Percent_2'])
    df_comparison_years['Years'] = list(range(2001,2024)) + ['2001 to 2023','2003 to 2012','2013 to 2022']


    for i, df in enumerate((df1, df2)):

        series = df.index.year.to_numpy()
        series = series[~np.isnan(series)]

        bin_edges = range(np.min(series),np.max(series)+2,1)
        counts, bins = np.histogram(series, bins=bin_edges)

        year_counts = {f'{int(bins[k])}': int(counts[k]) for k in range(len(counts))}
        total_1 = np.sum([int(year_counts.get(str(y),0)) for y in range(2003,2013)])
        total_2 = np.sum([int(year_counts.get(str(y),0)) for y in range(2013,2023)])

        df_comparison_years[f'Counts_{i+1}'] = [year_counts.get(str(year),0) for year in range(2001, 2024)] + [np.sum(counts), total_1, total_2]

    df_comparison_years['Percent_1'] = (df_comparison_years['Counts_1'] / len(df1)) * 100
    df_comparison_years['Percent_2'] = (df_comparison_years['Counts_2'] / len(df2)) * 100

    df_comparison_years['Counts_1'] = df_comparison_years['Counts_1'].apply(lambda x: f'{x:,}')
    df_comparison_years['Counts_2'] = df_comparison_years['Counts_2'].apply(lambda x: f'{x:,}')
    df_comparison_years['Percent_1'] = df_comparison_years['Percent_1'].apply(lambda x: f'{x:,.2f}%')
    df_comparison_years['Percent_2'] = df_comparison_years['Percent_2'].apply(lambda x: f'{x:,.2f}%')

    print(df_comparison_years)

    df_comparison = pd.DataFrame(columns=['Data_Name','Mean_1','STD_1','Median_1','Mean_2','STD_2','Median_2'])
    ###-------------------PLOT PARAMETERS-------------------###

    for col in columns:

        x_unit = units[col]
        if x_unit == 'rad':
            x_unit = 'deg'

        col_name = column_names.get(col,col)
        new_row = [f'{col_name} [{x_unit}]']

        ###-------------------PLOT COLUMNS-------------------###

        for df in (df1, df2):

            series = df[col].to_numpy()
            series = series[~np.isnan(series)]

            if x_unit == 'deg':
                mean = calc_circular_mean_error(series,unit=x_unit)
                std = np.degrees(circular_standard_deviation(series))
                series = np.degrees(series)
            else:
                mean = calc_mean_error(series)
                std = np.std(series,ddof=1)

            new_row.extend([mean,std,percentile_func(series)])

        df_comparison.loc[len(df_comparison)] = new_row

    df_comparison['Difference'] = df_comparison['Mean_1'] - df_comparison['Mean_2']
    df_comparison['No_STDs'] = df_comparison['Difference'].apply(lambda x: x.n / x.s)


    print(df_comparison)