import numpy as np
import pandas as pd

from handling_files import write_to_cdf
from data_analysis_kobel import load_compression_ratios, are_points_above_line
from data_processing import merge_dataframes, next_index


def analyse_merged(df, func, output_file=None, **kwargs):
    """
    Applies a specified function to a DataFrame and optionally writes the result to a CDF file.

    This function takes a DataFrame `df` and applies the provided function `func` to it. Additional
    keyword arguments can be passed to `func`. If an `output_file` is provided, the resulting DataFrame
    will be saved to that file in CDF format, overwriting any existing file.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to be processed by the function `func`.

    func : function
        A function to be applied to the DataFrame `df`. It must accept `df` as its first argument
        and can take additional keyword arguments as specified by `kwargs`.

    output_file : str, optional
        The path to the CDF file where the resulting DataFrame will be saved. If not provided, the
        processed DataFrame will be returned instead of being saved.

    **kwargs : additional arguments, optional
        Additional keyword arguments passed to `func`.

    Returns
    -------
    pandas.DataFrame or None
        If `output_file` is not provided, the processed DataFrame is returned. Otherwise, the result
        is written to the specified CDF file, and the function returns `None`.
    """
    new_df = func(df, **kwargs)
    if not output_file:
        return new_df
    write_to_cdf(new_df, output_file, overwrite=True)


def difference_columns(df, c1, c2):
    """
    Computes the difference between two columns in a DataFrame, optionally
    adjusting for angular differences (radians).

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the columns `c1` and `c2` to be subtracted.

    c1 : str
        The name of the first column in the DataFrame.

    c2 : str
        The name of the second column in the DataFrame.

    Returns
    -------
    data : pandas.Series
        A Series containing the difference between the values of `c1` and `c2`.
        If the unit of `c1` is 'rad' of 'deg', the result will be adjusted to handle
        angular differences within the range [-π, π) or [-180, 180).
    """
    data = df[c1] - df[c2]
    if df.attrs['units'][c1] == 'rad':
        data = (data + np.pi) % (2 * np.pi) - np.pi
    elif df.attrs['units'][c1] in ('deg','°'):
        data = (data + 180) % 360 - 180
    return data

def difference_series(data1, data2, data_col):
    """
    Computes the difference between two columns in a DataFrame, optionally
    adjusting for angular differences (radians).

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the columns `c1` and `c2` to be subtracted.

    c1 : str
        The name of the first column in the DataFrame.

    c2 : str
        The name of the second column in the DataFrame.

    Returns
    -------
    data : pandas.Series
        A Series containing the difference between the values of `c1` and `c2`.
        If the unit of `c1` is 'rad' of 'deg', the result will be adjusted to handle
        angular differences within the range [-π, π) or [-180, 180).
    """
    differences = merge_dataframes(data1, data2, '1', '2')
    differences['diff'] = difference_columns(differences, data_col+'_1', data_col+'_2')
    return differences

def difference_columns_dfs(df1, df2, data_col1, data_col2=None):
    """
    Computes the difference between two columns in a DataFrame, optionally
    adjusting for angular differences (radians).

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the columns `c1` and `c2` to be subtracted.

    c1 : str
        The name of the first column in the DataFrame.

    c2 : str
        The name of the second column in the DataFrame.

    Returns
    -------
    data : pandas.Series
        A Series containing the difference between the values of `c1` and `c2`.
        If the unit of `c1` is 'rad' of 'deg', the result will be adjusted to handle
        angular differences within the range [-π, π) or [-180, 180).
    """
    data_col2 = data_col1 if data_col2 is None else data_col2
    differences = merge_dataframes(df1, df2, '1', '2')
    return difference_columns(differences, data_col2+'_1', data_col2+'_2')


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

