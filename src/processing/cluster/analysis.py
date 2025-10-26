import pandas as pd
from ..reading import import_processed_data


def cluster_region_intervals(grison_dir, region='sw'):

    if region=='sw':
        label = 12
    elif region=='msh':
        label = 10

    crossings = import_processed_data(grison_dir)

    mask = (crossings['loc_num']==label) & (crossings['quality_num']>=2)

    reg_times = crossings.loc[mask].copy()
    reg_times.loc[:,'end_time'] = reg_times.index + pd.to_timedelta(reg_times.loc[:,'region_duration'], unit='s')
    reg_times.drop(columns=['loc_num','quality_num','complexity_num'], inplace=True)

    reg_times['new_group'] = (reg_times.index != reg_times['end_time'].shift()).cumsum()

    # Combine consecutive groups
    result = reg_times.groupby('new_group').agg(
        start_time=('end_time', lambda x: reg_times.loc[x.index[0]].name),
        region_duration=('region_duration', 'sum'),
        end_time=('end_time', 'last')
    ).set_index('start_time')

    reg_times = result.loc[result['region_duration']>60]

    c1_intervals = [(pd.to_datetime(str(start)), pd.to_datetime(str(end))) for start, end in zip(
            reg_times.index,
            reg_times.index + pd.to_timedelta(reg_times['region_duration'], unit='s'))]

    return c1_intervals