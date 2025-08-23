# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:13:15 2025

@author: richarj2
"""
import numpy as np
import pandas as pd

from data_processing import filter_percentile

def compute_bins(data, min_bin_count, initial_bins, min_bins, decrement=5):
    num_bins = initial_bins
    bin_counts, bin_edges = np.histogram(data, bins=num_bins)

    # Reduce the number of bins until the minimum count condition is satisfied
    while min(bin_counts) < min_bin_count:
        num_bins -= decrement
        if num_bins <= min_bins:
            break
        bin_counts, bin_edges = np.histogram(data, bins=num_bins)

    # Calculate bin centers and bin width
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]  # Uniform bin width

    return bin_edges, bin_centers, bin_width


def df_bin_stats(df, x_col, y_col, percentile=50, **kwargs):
    """
    Compute the mean, standard deviation, and counts for `y_col` within bins of `x_col`, ensuring that
    each bin contains at least `min_bin_count` data points. If the initial number of bins results in
    insufficient data per bin, the number of bins is reduced (down to `min_bins`).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be binned.

    x_col : str
        The name of the column to be used for binning the data.

    y_col : str
        The name of the column for which statistics (mean, std, count) are computed within each bin.

    percentile : int
        The desired percentile of the bin data. The default is 50 (median).

    min_bin_count : int, optional (default=500)
        The minimum number of data points required in each bin for the bin to be valid.

    initial_bins : int, optional (default=30)
        The initial number of bins to use for binning `x_col`.

    min_bins : int, optional (default=10)
        The minimum number of bins to allow if the number of bins is reduced due to insufficient data.

    Returns
    -------
    pd.DataFrame :
        - 'bin': The range of values for each bin.
        - 'center': The center of each bin.
        - 'count': The number of data points within each bin.
        - 'bin_mean': The mean of `y_col` within each bin.
        - 'bin_std': The standard deviation of `y_col` within each bin.
    """
    bin_edges = kwargs.get('bin_edges',None)
    bin_width = kwargs.get('bin_width',None)
    bin_num   = kwargs.get('bin_num',None)

    if bin_edges is None:
        bin_edges = calculate_bins(df[x_col], bin_width=bin_width, n_bins=bin_num)

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = np.diff(bin_edges)

    # Assign each data point in df to a bin
    df['bin'] = pd.cut(df[x_col], bins=bin_edges, include_lowest=True)

    # Compute mean, std, median, and count for `y_col` within each bin
    stats = df.groupby('bin', observed=True).agg(
        bin_mean = (y_col, 'mean'),
        bin_std  = (y_col, lambda x: np.std(x, ddof=1)),  # Sample standard deviation
        bin_sem  = (y_col, lambda x: sem(x)),
        bin_min  = (y_col, 'min'),
        bin_low  = (y_col, lambda x: percentile_func(x, 25)),
        bin_med  = (y_col, 'median'),
        bin_high = (y_col, lambda x: percentile_func(x, 75)),
        bin_max  = (y_col, 'max'),
        bin_iqr  = (y_col, lambda x: iqr(x)),
        bin_perc = (y_col, lambda x: percentile_func(x, percentile)),
        bin_mad  = (y_col, lambda x: mad(x)),
        bin_kps  = (y_col, lambda x: kps(x)),
        count    = (y_col, 'size')
    ).reset_index()

    # Add bin centers to the result
    stats['center'] = bin_centers

    # Add attributes to the result DataFrame
    stats.attrs['bin_width']      = bin_width
    stats.attrs['total_points']   = len(df)
    stats.attrs['num_bins']       = len(bin_centers)
    stats.attrs['min_counts']     = min(stats['count'])
    stats.attrs['bin_percentile'] = percentile
    stats.attrs['bin_edges']      = ", ".join(f"({bin_edges[i]}, {bin_edges[i+1]})" for i in range(len(bin_edges) - 1))

    return stats



def df_bin_extreme_stats(df, x_col, y_col, percentile=90, filt_range='Above', **kwargs):
    """
    Compute the mean, standard deviation, and counts for `y_col` within bins of `x_col`, ensuring that
    each bin contains at least `min_bin_count` data points. If the initial number of bins results in
    insufficient data per bin, the number of bins is reduced (down to `min_bins`).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be binned.

    x_col : str
        The name of the column to be used for binning the data.

    y_col : str
        The name of the column for which statistics (mean, std, count) are computed within each bin.

    percentile : int
        The desired percentile of the bin data. The default is 50 (median).

    min_bin_count : int, optional (default=500)
        The minimum number of data points required in each bin for the bin to be valid.

    initial_bins : int, optional (default=30)
        The initial number of bins to use for binning `x_col`.

    min_bins : int, optional (default=10)
        The minimum number of bins to allow if the number of bins is reduced due to insufficient data.

    Returns
    -------
    pd.DataFrame :
        - 'bin': The range of values for each bin.
        - 'center': The center of each bin.
        - 'count': The number of data points within each bin.
        - 'bin_mean': The mean of `y_col` within each bin.
        - 'bin_std': The standard deviation of `y_col` within each bin.
    """
    min_bin_count = kwargs.get('min_bin_count',600)
    initial_bins = kwargs.get('initial_bins',30)
    min_bins = kwargs.get('min_bins',10)

    bin_edges, bin_centers, bin_width = compute_bins(df[x_col], min_bin_count, initial_bins, min_bins)

    # Assign each data point in df to a bin
    df['bin'] = pd.cut(df[x_col], bins=bin_edges, include_lowest=True)

    # Apply filtering
    filt_df = filter_percentile(df, y_col, df['bin'], percentile, filt_range)

    # Compute statistics on filtered data
    stats = filt_df.groupby('bin', observed=True).agg(
        bin_mean = (y_col, 'mean'),
        bin_std  = (y_col, lambda x: np.std(x, ddof=1)),  # Sample standard deviation
        bin_sem  = (y_col, lambda x: sem(x)),
        bin_min  = (y_col, 'min'),
        bin_max  = (y_col, 'max'),
        bin_iqr  = (y_col, lambda x: iqr(x)),
        count    = (y_col, 'size')
    ).reset_index()

    # Add bin centers to the result
    stats['center'] = bin_centers

    # Add bin_percentile to stats
    stats['cutoff'] = stats['bin'].map(filt_df.attrs['bin_limits'])


    # Add attributes to the result DataFrame
    stats.attrs['bin_width']    = bin_width
    stats.attrs['total_points'] = len(filt_df)
    stats.attrs['num_bins']     = len(bin_centers)
    stats.attrs['min_counts']   = min(stats['count'])

    return stats, filt_df



def df_rolling_stats(df, x_col, y_col, percentile=50, window_size=1000):
    """
    A

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be analysed.

    Returns
    -------
    pd.DataFrame :
        - 'A': A.
    """
    # Sort by x_col to ensure proper rolling
    sorted_df = df.sort_values(by=x_col).reset_index(drop=True)

    # Define rolling object
    rolling = sorted_df[y_col].rolling(window=window_size, center=True)

    # Calculate rolling statistics
    stats = pd.DataFrame({
        'mean':   rolling.mean(),
        'std':    rolling.std(),
        'count':  rolling.count(),
        'min':    rolling.min(),
        'q1':     rolling.apply(lambda x: np.percentile(x.dropna(), 25), raw=False),
        'median': rolling.median(),
        'q3':     rolling.apply(lambda x: np.percentile(x.dropna(), 75), raw=False),
        'max':    rolling.max(),
        'iqr':    rolling.apply(lambda x: iqr(x.dropna()), raw=False),
        'perc':   rolling.apply(lambda x: np.percentile(x.dropna(), percentile), raw=False),
    })

    # Add x_col for reference
    stats[x_col] = sorted_df[x_col]

    # Add attributes to the result DataFrame
    stats.attrs['window_width'] = window_size
    stats.attrs['total_points'] = len(df)
    stats.attrs['window_percentile'] = percentile

    return stats

def df_bin_percentiles(df, x_col, y_col, percentiles=(25,50,60,70,80,90), min_bin_count=500, initial_bins=30, min_bins=10):
    """
    Compute the mean, standard deviation, and counts for `y_col` within bins of `x_col`, ensuring that
    each bin contains at least `min_bin_count` data points. If the initial number of bins results in
    insufficient data per bin, the number of bins is reduced (down to `min_bins`).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to be binned.

    x_col : str
        The name of the column to be used for binning the data.

    y_col : str
        The name of the column for which statistics (mean, std, count) are computed within each bin.

    percentile : int
        The desired percentile of the bin data. The default is 50 (median).

    min_bin_count : int, optional (default=500)
        The minimum number of data points required in each bin for the bin to be valid.

    initial_bins : int, optional (default=30)
        The initial number of bins to use for binning `x_col`.

    min_bins : int, optional (default=10)
        The minimum number of bins to allow if the number of bins is reduced due to insufficient data.

    Returns
    -------
    pd.DataFrame :
        - 'bin': The range of values for each bin.
        - 'percentile': The value of each percentile in percentiles for the bin.
    """
    num_bins = initial_bins
    bin_counts, bin_edges = np.histogram(df[x_col], bins=num_bins)

    # Reduce number of bins if the minimum count condition is not met
    while min(bin_counts) < min_bin_count:
        num_bins -= 5
        if num_bins <= min_bins:
            break
        bin_counts, bin_edges = np.histogram(df[x_col], bins=num_bins)

    # Calculate bin centers
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]  # Calculate uniform bin width

    # Assign each data point in df to a bin
    df['bin'] = pd.cut(df[x_col], bins=bin_edges, include_lowest=True)

    # Define the percentile function
    def percentile_func(series, p):
        return np.percentile(series, p)

    # Compute requested percentiles
    stats = df.groupby('bin', observed=True).agg(count=(y_col, 'size')).reset_index()

    for perc in percentiles:
        stats[f'bin_perc_{perc}'] = df.groupby('bin', observed=True)[y_col].apply(lambda x: np.percentile(x, perc)).values

    # Add bin centers to the result
    stats['center'] = bin_centers[:len(stats)]  # Slice to match reduced bins

    # Add attributes to the result DataFrame
    stats.attrs['bin_width'] = bin_width
    stats.attrs['total_points'] = len(df)
    stats.attrs['num_bins'] = num_bins
    stats.attrs['min_counts'] = min(bin_counts)
    stats.attrs['percentiles'] = percentiles

    return stats



def calc_med_hist(centres, counts):

    cumulative_counts = np.cumsum(counts)
    cumulative_distribution = cumulative_counts / cumulative_counts[-1] # normalise
    med_index = np.searchsorted(cumulative_distribution, 0.5)

    # Step 4: Interpolate to get the exact median
    if med_index == 0:
        # Median falls in the first bin
        return centres[0]
    else:
        # Linear interpolation
        x1, x2 = centres[med_index - 1], centres[med_index]
        y1, y2 = cumulative_distribution[med_index - 1], cumulative_distribution[med_index]
        median = x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1)
        return median