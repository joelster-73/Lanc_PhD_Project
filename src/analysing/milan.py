
import numpy as np
import pandas as pd

from IPython.display import display as display_df
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from data_processing import filter_by_spacecraft, filter_sign
from data_analysis import difference_columns, straight_best_fit, percentile_func
from data_plotting import common_prefix, create_label, data_string, dark_mode_fig, save_figure

dark_mode = True

white = 'w'
black = 'k'
blue = 'b'
if dark_mode:
    white = 'k'
    black = 'w'
    blue = 'c'

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.size'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.title_fontsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 14

def investigate_difference_milan(df, data1_col, data2_col, dist_col, ind_col=None, **kwargs):
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

    print_stats = kwargs.get('print_stats',False)
    display     = kwargs.get('display','Scatter')
    diff_type   = kwargs.get('diff_type','Absolute')

    x_data_name = kwargs.get('x_data_name',None)
    y_data_name = kwargs.get('y_data_name',common_prefix(data1_col,data2_col)[:-1])
    title       = kwargs.get('title',None)
    title_info  = kwargs.get('title_info','')

    polarity    = kwargs.get('polarity','Both')
    abs_x       = kwargs.get('abs_x',False)

    sc_id   = kwargs.get('sc_id',None)
    sc_col  = kwargs.get('sc_col',None)
    sc_keys = kwargs.get('sc_keys',None)
    sc_name = None

    ###---------------CONSTRUCT COLUMN LABELS---------------###

    # Validate column labels
    for label in (data1_col, data2_col, dist_col, ind_col):
        if label is not None and label not in df.keys() and ind_col != 'year_fraction':
            raise ValueError(f'Field data "{label}" not found in data.')
    if ind_col is not None:
        x_label = ind_col
        if ind_col == 'year_fraction':
            x_data_name = 'year fraction'
    else:
        x_label = dist_col
    x_data_str = data_string(x_label)
    x_unit = df.attrs['units'].get(x_label, None)
    x_axis_label = create_label(x_data_str, unit=x_unit, data_name=x_data_name)

    y_data_str  = data_string(y_data_name)
    y1_data_str = data_string(data1_col)
    y2_data_str = data_string(data2_col)
    y_unit = df.attrs['units'].get(data1_col, None)

    filtered_data_str = ''
    sc_str = ''

    ###---------------FILTERS DATA BY SPACECRAFT---------------###
    if sc_id and sc_col:
        filter_by_spacecraft(df, sc_col, sc_id)
        sc_name = sc_keys[sc_id]
        sc_str = f' for the spacecraft {sc_name}'
    elif sc_id:
        raise ValueError('To filter by spacecraft, argument needed for "sc_col" and "sc_id".')

    if abs_x:
        df[x_label] = np.abs(df[x_label])
        x_data_str = f'|{x_data_str}|'

    ###---------------CALCULATES DATA FOR DIFFERENCE TYPE---------------###
    y_label='diff'
    if diff_type == 'Absolute':
        df[y_label] = difference_columns(df, data1_col, data2_col)
        if polarity != 'Both':
            df = filter_sign(df, y_label, polarity)
        operation = '-'
        y_axis_label = f'$\\Delta$${y_data_str}$ [{y_unit}]'

    elif diff_type == 'Modulus Absolute':
        df[y_label] = np.abs(difference_columns(df, data1_col, data2_col))
        operation = '-'
        y_axis_label = f'|$\\Delta$${y_data_str}$| [{y_unit}]'

    elif diff_type == 'Relative':
        df[y_label] = difference_columns(df, data1_col, data2_col) / df[data2_col]
        df[y_label].replace([np.inf, -np.inf], np.nan, inplace=True)
        if polarity != 'Both':
            df = filter_sign(df, y_label, polarity)
        operation = '/'
        y_axis_label = f'(${y1_data_str}$ - ${y2_data_str}$) / ${y2_data_str}$'
        y_unit = ''

    elif diff_type == 'Modulus Relative': # Ensures similar values around 0, and all >0
        df[y_label] = np.abs(difference_columns(df, data1_col, data2_col)) / np.abs(df[data2_col])
        df[y_label].replace([np.inf, -np.inf], np.nan, inplace=True)
        operation = '/'
        y_axis_label = f'|${y1_data_str}$ - ${y2_data_str}$| / |${y2_data_str}$|'
        y_unit = ''

    else:
        raise ValueError(f'"{diff_type}" not valid difference mode.')

    ###---------------PLOTS DIFFERENCE AGAINST DISTANCE/TIME/PARAM---------------###
    colours = ['Red', blue, 'Green']

    fig, (ax_main, ax_hist) = plt.subplots( # Figure with two subplots
        2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [5, 1]}
    )
    # If plotting against distance or another parameter
    if ind_col is None:
        df_stats = df_bin_milan_distance(df, x_label, y_label)
        # (1) Plotting against distance
        for i in range(len(df_stats)):
            ax_main.errorbar(
                df_stats['bin_q2'].iloc[i], df_stats['data_q2'].iloc[i],
                xerr=[[df_stats['bin_q2'].iloc[i] - df_stats['bin_q1'].iloc[i]],
                      [df_stats['bin_q3'].iloc[i] - df_stats['bin_q2'].iloc[i]]],
                yerr=[[df_stats['data_q2'].iloc[i] - df_stats['data_q1'].iloc[i]],
                      [df_stats['data_q3'].iloc[i] - df_stats['data_q2'].iloc[i]]],
                fmt='o', markersize=10, capsize=10, color=colours[i]
            )

        m, y0, R2 = straight_best_fit(df_stats['bin_q2'], df_stats['data_q2'], name='Milan Bins (q2)')
        ax_main.plot(
            df_stats['bin_q2'], m*df_stats['bin_q2']+y0,
            c=black, ls='--', label=f'Median best fit ($R^2$={R2:.3f}):\n{m:.3f}x+{y0:.3f} {y_unit}'
        )
        m, y0, R2 = straight_best_fit(df_stats['bin_q2'], df_stats['data_q3'], name='Milan Bins (q3)')
        ax_main.plot(
            df_stats['bin_q2'], m*df_stats['bin_q2']+y0,
            c=black, ls=':', label=f'Upper Quartile best fit ($R^2$={R2:.3f}):\n{m:.3f}x+{y0:.3f} {y_unit}'
        )
        # (2) Plotting counts
        ax_hist.bar(
            df_stats['center'], df_stats['count'],
            width=df_stats['width'], color=colours, edgecolor='black', align='center'
        )
        leg_title=None

    else:
        df_stats = df_bin_milan_param(df, ind_col, y_label, dist_col)
        # (1) Plotting against time/parameter
        for i, dist_bin in enumerate(np.unique(df_stats['distance_bin'])):
            data = df_stats[df_stats['distance_bin']==dist_bin]
            ax_main.plot(
                data['x_q2'],data['y_q2'],
                color=colours[i],label=f'{dist_bin} {x_unit}'
            )
            yerr = np.array([
                data['y_q2'].to_numpy() - data['y_q1'].to_numpy(),
                data['y_q3'].to_numpy() - data['y_q2'].to_numpy()
            ])
            ax_main.errorbar(
                data['x_q2'], data['y_q2'],
                yerr=yerr,
                fmt='o', markersize=10, capsize=10, color=colours[i]
            )
            # (2) Plotting counts
            ax_hist.bar(
                data['center'], data['count'],
                width=data['width'], facecolor='none', edgecolor=colours[i], linewidth=2.5, align='center'
            )
            leg_title='Distance Bins'

    ax_main.set_xlabel(x_axis_label, c=black)
    ax_main.set_ylabel(y_axis_label, c=black)

    ax_hist.set_xlim(ax_main.get_xlim())
    ax_hist.set_title('Counts in bins.', c=black)
    ax_hist.set_ylabel('Counts', c=black)
    ax_hist.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:,.0f}"))

    legend = ax_main.legend(loc='best', title=leg_title, labelcolor=black)
    legend.get_title().set_color(black)

    if 'r_x' in x_label or 'r_y' in x_label:
        ax_main.invert_xaxis()
        ax_hist.invert_xaxis()
    if diff_type == 'Relative':
        ax_main.set_yscale('symlog', linthresh=1)  # linthresh defines the linear threshold around 0
        ax_main.axhline(y=0, color=white, linestyle='--')
    elif diff_type == 'Modulus Relative':
        ax_main.set_yscale('log')

    dark_mode_fig(fig,black,white)
    if display == 'Heat': # Ensures heat map always has black background
        ax_main.set_facecolor('k')

    start_date_str = df.index[0].strftime('%Y-%m-%d')
    end_date_str   = df.index[-1].strftime('%Y-%m-%d')

    title_str =  f'${y1_data_str}$ {operation} ${y2_data_str}$' if title is None else title
    fig.suptitle(f'{title_str} against {x_axis_label}{sc_str} '
                 f'from {start_date_str} to {end_date_str}. {len(df):,} minutes of data are in the figure.'
                 f'{filtered_data_str} {title_info}', c=black, wrap=True)
    plt.tight_layout();
    save_figure(fig)
    plt.show()

    if print_stats:
        print('Stats for difference against distance:')
        display_df(df_stats)



def df_bin_milan_distance(df, x_col, y_col, bin_edges=None):
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

    Returns
    -------
    pd.DataFrame :
        - 'bin_q1':
    """
    if bin_edges is None:
        x_col_min = int(np.floor(np.min(df[x_col])))
        x_col_max = int(np.ceil(np.max(df[x_col])))
        bin_edges=np.array([x_col_min,45,70,x_col_max])

    # Assign each data point in df to a bin
    df['bin'] = pd.cut(df[x_col], bins=bin_edges, include_lowest=True)

    # Compute statistics on filtered data
    stats = df.groupby('bin', observed=True).agg(
        bin_q1  = (x_col, lambda x: percentile_func(x, 25)),
        bin_q2  = (x_col, lambda x: percentile_func(x, 50)),
        bin_q3  = (x_col, lambda x: percentile_func(x, 75)),
        data_q1 = (y_col, lambda x: percentile_func(x, 25)),
        data_q2 = (y_col, lambda x: percentile_func(x, 50)),
        data_q3 = (y_col, lambda x: percentile_func(x, 75)),
        count   = (y_col, 'size')
    ).reset_index()
    stats['width']  = np.diff(bin_edges)
    stats['center'] = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Add attributes to the result DataFrame
    stats.attrs['total_points'] = len(df)
    stats.attrs['num_bins']     = len(bin_edges) - 1
    stats.attrs['bin_edges']    = ", ".join(f"({bin_edges[i]}, {bin_edges[i+1]})" for i in range(len(bin_edges) - 1))

    return stats

def calc_year_frac(df, time_col):
    # year fraction since summer solstic, taken to be 0.5 through year
    if time_col == 'index':
        time_data = df.index
    else:
        time_data = df[time_col].dt
    fraction = time_data.dayofyear / (365 + time_data.is_leap_year.astype(int))

    fraction = (fraction + 0.5) % 1.0
    return fraction

def df_bin_milan_param(df, x_col, y_col, d_col, x_bin_num=None, d_bin_edges=None):
    """

    """

    if x_col == 'year_fraction':
        df[x_col] = calc_year_frac(df, 'index') # Compute year-fraction from timestamps

        if x_bin_num is None:
            x_bin_num = 10 # Default to 10 bins from 0 to 1
        x_bin_edges = np.linspace(0, 1, x_bin_num+1)
        x_bin_width = 0.1
    else:
        x_col_min = int(np.floor(np.min(df[x_col])))
        x_col_max = int(np.ceil(np.max(df[x_col])))
        if x_bin_num is None:
            x_bin_num = x_col_max - x_col_min
        x_bin_edges = np.linspace(x_col_min, x_col_max, x_bin_num+1)
        x_bin_width = 1

    # Define default bin edges if not provided
    if d_bin_edges is None:
        d_col_min = int(np.floor(np.min(df[d_col])))
        d_col_max = int(np.ceil(np.max(df[d_col])))
        d_bin_edges = np.array([d_col_min, 45, 70, d_col_max])

    # Bin data by distance and year-fraction
    df['distance_bin'] = pd.cut(df[d_col], bins=d_bin_edges)
    df[f'{x_col}_bin'] = pd.cut(df[x_col], bins=x_bin_edges)

    # Group by both bins and compute statistics
    stats = df.groupby([f'{x_col}_bin', 'distance_bin'], observed=False).agg(
        x_q2  = (x_col, lambda x: np.percentile(x, 50)),
        y_q1  = (y_col, lambda y: np.percentile(y, 25)),
        y_q2  = (y_col, lambda y: np.percentile(y, 50)),
        y_q3  = (y_col, lambda y: np.percentile(y, 75)),
        count = (y_col, 'size'),
    ).reset_index()
    stats['width']  = np.full(len(stats),x_bin_width)
    stats['center'] = stats[f'{x_col}_bin'].apply(lambda x: (x.left + x.right) / 2)

    # Add attributes to the result DataFrame
    stats.attrs['total_points'] = len(df)
    stats.attrs['num_distance_bins'] = len(d_bin_edges) - 1
    stats.attrs[f'num_{x_col}_bins'] = len(x_bin_edges) - 1
    stats.attrs['distance_bin_edges'] = ", ".join(map(str, np.unique(stats['distance_bin'])))
    stats.attrs[f'{x_col}_bin_edges'] = ", ".join(map(str, np.unique(stats[f'{x_col}_bin'])))

    return stats