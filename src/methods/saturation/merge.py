# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:21:25 2025

@author: richarj2
"""
import pandas as pd

from ...config import get_proc_directory

from ...processing.reading import import_processed_data
from ...processing.writing import write_to_cdf

# MP & BS
all_spacecraft = ('c1','mms1','tha','thb','thc','thd','the')

spacecraft     = {'sw':  {'field': all_spacecraft, 'plasma': ('c1','mms1','thb','thc')},
                  'msh': {'field': all_spacecraft, 'plasma': ('c1','mms1','the')}}


# %%% Merge_sc

def merge_sc_in_region(region, data_pop='plasma', resolution='5min', sc_keys=None, nose=False):

    """
    data_pop = 'field' means field only
    data_pop = 'plasma' means including plasma (so also field)
    """

    ### think about how to merge, whether to merge based on unique measurement time or unique lagged time
    ### then update spacecraft dictionaries above

    dfs_combined = []
    if sc_keys is None:
        sc_keys = spacecraft[region][data_pop]

    for sc in sc_keys:

        print(f'Importing {sc.upper()}.')

        df_sc = import_processed_data(region, dtype=data_pop, resolution=resolution, file_name=f'{region}_times_{sc}')

        # Need suffix when combining but removing for individual file
        df_sc.rename(columns={col: f'{col}_{sc}' for col in df_sc.columns}, inplace=True)

        dfs_combined.append(df_sc)

    ###----------COMBINING----------###
    print('Combining spacecraft')
    df_wide = pd.concat(dfs_combined, axis=1)

    mask = pd.DataFrame({sc: df_wide[f'B_avg_{sc}'].notna() for sc in sc_keys})
    first_valid = mask.idxmax(axis=1)

    result = []
    for sc in sc_keys:
        suffix = f'_{sc}'
        sc_cols = [col for col in df_wide.columns if col.endswith(suffix)]
        renamed = {col: col[:-len(suffix)] for col in sc_cols}
        subset = (df_wide.loc[first_valid == sc, sc_cols].rename(columns=renamed))
        subset[f'sc_{region}'] = sc
        result.append(subset)

    df_combined = pd.concat(result).sort_index()
    df_combined.index.name = 'epoch'

    ### UNCOMMENT ----->

    #update_attributes(df_combined, df_merged_attrs, region, '')

    print('Merging')
    df_combined.attrs['units'][f'sc_{region}'] = 'STRING'

    df_combined.drop(columns=['no_np_ratio', 'no_np_ratio_unc', 'no_np_ratio_count', 'nhe_np_ratio', 'nhe_np_ratio_unc', 'nhe_np_ratio_count'], inplace=True) # columns don't care about when combined

    # Write
    print('Writing combined to file...')
    DIR = get_proc_directory(region, dtype=data_pop, resolution=resolution)
    write_to_cdf(df_combined, directory=DIR, file_name=f'{region}_times_combined', reset_index=True)



def update_attributes(df, attrs_dict, region='sw', suffix=''):

    df.attrs = attrs_dict
    for temp in (f'T_tot{suffix}',f'T_tot_unc{suffix}'):
        df.attrs['units'][temp] = 'MK'

    df.attrs['units']['prop_time_s'] = 's'
    if region=='msh':
        df.attrs['units']['Delta B_theta'] = 'rad'
        df.attrs['units']['Delta B_z'] = '1'
