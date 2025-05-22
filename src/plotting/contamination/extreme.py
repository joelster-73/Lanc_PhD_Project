# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:44:22 2025

@author: richarj2
"""

def plot_extreme_diffs_days(df1, df2, df_merged, data_name, source1, source2, df_regions, **kwargs):

    filtering  = kwargs.get('filtering','Abs')
    min_count  = kwargs.get('min_count',60)


    line_fmt   = kwargs.get('line_fmt','-')
    #y_label    = kwargs.get('y_label',data_name)

    df_merged = df_merged.copy()

    col1 = f'{data_name}_{source1}'
    col2 = f'{data_name}_{source2}'

    units = df1.attrs['units']
    data_unit = units.get(data_name,None)
    data_label = create_label(data_string(data_name), unit=data_unit)

    posx  = 'r_x_GSE'
    posy  = 'r_y_GSE'
    posz  = 'r_z_GSE'
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

    extreme_days = np.unique(df_extreme.index.date)
    extreme_windows = []

    for day in extreme_days:
        # Data on the "extreme day"
        extreme_range = df_extreme[df_extreme.index.date == day]
        # More than e.g. 60 extreme minutes on the day
        if len(extreme_range) < min_count:
            continue

        try:
            start = df_regions.index[df_regions.index.date==day][0]
            end   = df_regions.index[df_regions.index.date==day][-1]
        except:
            print(f'Error with {day}.')
            continue

        day_start = pd.Timestamp(day)
        day_end = pd.Timestamp(day+Timedelta(days=1))

        # Full data on the day
        df1_day = df1[df1.index.date==day][[data_name,posx,posy,posz]]
        df2_day = df2[df2.index.date==day][[data_name]]

        extreme_windows.append((start.to_pydatetime(),end.to_pydatetime(),len(extreme_range)))

        first_region = previous_index(df_regions,start)
        last_region  = next_index(df_regions,end)
        regions      = df_regions[(df_regions.index==first_region) | (df_regions.index.date==day) | (df_regions.index==last_region)]

        band_starts = regions.index[regions['loc_num']==12]
        band_ends   = [next_index(df_regions, band_start) for band_start in band_starts]

        ###-------------------PLOT FULL DAY-------------------###
        fig, ax = plt.subplots()

        plot_segments(ax, df2_day, data_name, '#F28500', source2, line_fmt)
        plot_segments(ax, df1_day, data_name, 'b', source1, line_fmt)

        extreme_minutes = df_extreme[(df_extreme.index >= day_start) & (df_extreme.index <= day_end)]
        ax.scatter(extreme_minutes.index, extreme_minutes[col1], c='r', s=3, label=f'Extreme: {len(extreme_minutes):,}')

        ax.axvspan(day_start, day_end, alpha=0.35, color='k', label='Not SW')
        for band_start, band_end in zip(band_starts, band_ends):

            band_start = pd.Timestamp(band_start)
            band_end = pd.Timestamp(band_end)

            if band_start < day_start:
                band_start = day_start
            if band_end > day_end:
                band_end = day_end
            ax.axvspan(band_start, band_end, alpha=1, color='w')
            ax.axvline(x=band_start, linewidth=1.75, color='k', linestyle='--')
            ax.axvline(x=band_end, linewidth=1.75, color='k', linestyle='--')

        date_format = mdates.DateFormatter('%H:%M:%S')
        ax.xaxis.set_major_formatter(date_format)

        plt.xlabel(day,c=black)
        plt.ylabel(data_label,c=black)

        add_legend(fig, ax)
        fig.suptitle(f'Comparing {source1} and {source2} on {day}',c=black,wrap=True)
        dark_mode_fig(fig,black,white)
        plt.tight_layout()
        save_figure(fig, sub_directory='Windows')
        plt.show()
        plt.close()


def plot_extreme_diffs_windows(df1, df2, df_merged, data_name, source1, source2, df_regions, on_day=None, **kwargs):

    filtering    = kwargs.get('filtering','Kobel')
    min_count    = kwargs.get('min_count',1)
    line_fmt     = kwargs.get('line_fmt','-')
    print_counts = kwargs.get('print_counts',True)
    print_times  = kwargs.get('print_times',False)

    df_merged = df_merged.copy()

    col1 = f'{data_name}_{source1}'
    col2 = f'{data_name}_{source2}'

    units = df1.attrs['units']
    data_unit = units.get(data_name,None)
    data_label = create_label(data_string(data_name), unit=data_unit)

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
        print_times = True
        print_counts = False
        target_date = pd.Timestamp(on_day).date()
        band_starts = band_starts[band_starts.to_series().dt.date == target_date]

    band_ends   = [next_index(df_regions, band_start) for band_start in band_starts]
    counts = []


    for band_start, band_end in zip(band_starts, band_ends):

        band_start = pd.Timestamp(band_start)
        band_end = pd.Timestamp(band_end)

        in_interval = (df_extreme.index >= band_start) & (df_extreme.index <= band_end)
        number_in_interval = int(np.sum(in_interval))

        if number_in_interval < min_count:
            # Doesn't plot intervals where no minutes are extreme or if min_count should be higher
            continue

        counts.append(number_in_interval)

        extreme_minutes = df_extreme[in_interval]

        data1_mean = calc_mean_error(df1[data_name], band_start, band_end)
        data2_mean = calc_mean_error(df2[data_name], band_start, band_end)

        # Full data within the range
        start_time = band_start.replace(minute=0, second=0, microsecond=0)
        end_time = band_end.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        start_day = start_time.date()
        end_day   = end_time.date()
        if start_day == end_day:
            day_title = f'on {start_day:%Y-%m-%d}'
        else:
            start_time -=  timedelta(hours=1)
            end_time   +=  timedelta(hours=1)
            day_title = f'from {start_day:%Y-%m-%d} to {end_day:%Y-%m-%d}'

        data1_hours = df1[(df1.index >= start_time) & (df1.index <= end_time)]
        data2_hours = df2[(df2.index >= start_time) & (df2.index <= end_time)]

        ###-------------------PLOT SPACECRAFT DATA-------------------###
        fig, ax = plt.subplots()

        plot_segments(ax, data2_hours, data_name, '#F28500', None, line_fmt)
        plot_segments(ax, data1_hours, data_name, 'b', f'{source1}: ${data1_mean:.1uL}$ {data_unit}', line_fmt)
        ax.plot([], [], ls=line_fmt, c='#F28500',label=f'{source2}: ${data2_mean:.1uL}$ {data_unit}')

        ax.axvspan(start_time, end_time, alpha=0.2, color='k')
        ax.axvspan(band_start, band_end, alpha=1, color='w')
        ax.axvline(x=band_start, linewidth=1.75, color='k', linestyle='--', label=f'{band_start:%H:%M:%S} | {band_end:%H:%M:%S}')
        ax.axvline(x=band_end, linewidth=1.75, color='k', linestyle='--')

        mins = 'min' if np.sum(in_interval) == 1 else 'mins'
        ax.scatter(extreme_minutes.index, extreme_minutes[col1], c='r', s=3, label=f'C1 > Kobel: {len(extreme_minutes):,} {mins}')

        #date_format = mdates.DateFormatter('%H:%M:%S')
        #ax.xaxis.set_major_formatter(date_format)

        formatter = FuncFormatter(custom_date_formatter)
        ax.xaxis.set_major_formatter(formatter)

        ax.set_xlabel(' ',c=black)
        ax.set_ylabel(data_label,c=black)

        ax.set_xlim(start_time,end_time)

        add_legend(fig, ax, loc='upper center', anchor=(0.5,-0.1),cols=4)
        add_figure_title(fig, f'Comparing {source1} and {source2} {day_title}')
        dark_mode_fig(fig,black,white)
        plt.tight_layout()
        save_figure(fig, sub_directory='Windows', file_name=f'{start_day:%Y-%m-%d}')
        plt.show()
        plt.close()

        if print_times:
            print(extreme_minutes[col1])

    if print_counts:

        counts_df = pd.DataFrame.from_dict(Counter(counts), orient='index', columns=['Count']).reset_index()
        counts_df.rename(columns={'index': 'Value'}, inplace=True)
        counts_df.sort_values(by='Value', inplace=True)
        counts_df.set_index('Value', inplace=True)
        pd.set_option('display.max_rows', None)
        total_count = counts_df['Count'].sum()
        print(counts_df)
        print("Sum of counts:", total_count)







def plot_on_day(df1, df2, df_regions, data_col, day, **kwargs):
    '''

    '''
    source1    = kwargs.get('source1','Cluster')
    source2    = kwargs.get('source2','OMNI')
    line_fmt   = kwargs.get('line_fmt','-')

    units = df1.attrs['units']
    data_unit = units.get(data_col,None)

    data_label = create_label(data_string(data_col), unit=data_unit)

    # Full data on the day
    day = day.date()
    df1_day    = df1[df1.index.date==day]
    df2_day    = df2[df2.index.date==day]

    first_region = previous_index(df_regions,df1_day.index[0])
    regions    = df_regions[(df_regions.index==first_region) | (df_regions.index.date==day)]

    band_starts = regions.index[regions['loc_num']==12]
    band_ends  = [next_index(df_regions, band_start) for band_start in band_starts]

    data_start = pd.Timestamp(df1_day.index[0])
    data_end = pd.Timestamp(df1_day.index[-1])

    ###-------------------PLOT SPACECRAFT DATA-------------------###
    fig, ax = plt.subplots()

    plot_segments(ax, df1_day, data_col, blue, source1, line_fmt)
    plot_segments(ax, df2_day, data_col, 'r', source2, line_fmt)

    ax.axvspan(data_start, data_end, alpha=0.4, color='k')
    for band_start, band_end in zip(band_starts, band_ends):

        band_start = pd.Timestamp(band_start)
        band_end = pd.Timestamp(band_end)

        if band_start < data_start:
            band_start = data_start
        if band_end > data_end:
            band_end = data_end
        ax.axvspan(band_start, band_end, alpha=1, color='w')

    date_format = mdates.DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(date_format)

    plt.xlabel(day,c=black)
    plt.ylabel(data_label,c=black)
    add_legend(fig, ax)
    add_figure_title(fig, f'Comparing {data_col} for {source1} and {source2} on {day}.')


    dark_mode_fig(fig,black,white)
    #current_x_ax = ax.get_xlim()
    save_figure(fig)
    plt.show()
    plt.close()
