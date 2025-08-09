# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:10:40 2025

@author: richarj2
"""
import pandas as pd

def assign_region(df_positions, df_regions, columns={}, bad_days=None, good_days=None, print_changes=True):
    df_regions = df_regions.copy()
    df_regions['boundary'] = df_regions.index
    df_columns = ['boundary','loc_num']
    for reg_col, pos_col in columns.items():
        if reg_col in df_regions.columns() and reg_col not in df_columns:
            df_columns.append(reg_col)
        else:
            print(f'Column {reg_col} not in df_regions.')
    # Use merge_asof to assign region labels
    region_labels = pd.merge_asof(
        df_positions,
        df_regions[df_columns],
        left_index=True,  # Use the index of df_positions as the left key
        right_index=True,  # Use the index of df_regions as the right key
        direction='backward'  # Look for the closest previous time in df_regions
    )

    # Assign the 'loc_num' as 'region' in df_positions
    df_positions['time_spent'] = (df_positions.index - region_labels['boundary']).dt.total_seconds()
    df_positions.attrs['units']['time_spent'] = 's'
    df_positions['GRMB_region'] = region_labels['loc_num']

    # Sets last entries to new label as GRMB has ended labelling
    df_positions.loc[df_positions.index > df_regions.index[-1], 'GRMB_region'] = 16

    for reg_col, pos_col in columns.items():
        if reg_col in region_labels:
            df_positions[pos_col] = region_labels[reg_col]

    time_dict = df_positions.groupby('GRMB_region').size().to_dict()
    total_length = sum(time_dict.values())
    time_dict['100'] = total_length
    time_df = pd.DataFrame.from_dict(time_dict, orient='index', columns=['GRMB_minutes'])
    time_df['percentage'] = 100*time_df['GRMB_minutes']/total_length

    if bad_days is not None:
        for start, end, region_num in bad_days:
            mask = (df_positions.index >= start) & (df_positions.index <= end)
            df_positions.loc[mask, 'GRMB_region'] = region_num

        change_dict = df_positions.groupby('GRMB_region').size().to_dict()
        total_length = sum(change_dict.values())
        change_dict['100'] = total_length
        change_df = pd.DataFrame.from_dict(change_dict, orient='index', columns=['after_bad'])
        time_df['R12_analysis'] = change_df['after_bad'] - time_df['GRMB_minutes']
        time_df['Region_12_to_drop'] = change_df['after_bad']

    if good_days is not None:
        mask = df_positions.index.isin(good_days)
        df_positions.loc[mask, 'GRMB_region'] = 12

        change_dict = df_positions.groupby('GRMB_region').size().to_dict()
        total_length = sum(change_dict.values())
        change_dict['100'] = total_length
        change_df = pd.DataFrame.from_dict(change_dict, orient='index', columns=['after_good'])

        if bad_days is not None:

            time_df['R11_13_analysis'] = change_df['after_good'] - time_df['Region_12_to_drop']
            time_df.drop(columns=['Region_12_to_drop'],inplace=True)

        else:
            time_df['R11_13_analysis'] = change_df['after_good'] - time_df['GRMB_minutes']

    elif bad_days is not None:
        time_df.drop(columns=['Region_12_to_drop'],inplace=True)

    if good_days is not None or bad_days is not None:
        change_dict = df_positions.groupby('GRMB_region').size().to_dict()
        total_length = sum(change_dict.values())
        change_dict['100'] = total_length
        change_df = pd.DataFrame.from_dict(change_dict, orient='index', columns=['Final_minutes'])

        time_df['Final_minutes'] = change_df['Final_minutes']

    if print_changes:
        pd.options.display.max_columns=8
        pd.options.display.max_rows=20
        print(time_df)