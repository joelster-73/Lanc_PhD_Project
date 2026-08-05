# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:21:25 2025

@author: richarj2
"""
import pandas as pd

from .filter import filter_spacecraft

from ...config import get_proc_directory

from ...processing.reading import import_processed_data
from ...processing.writing import write_to_cdf


# %%% Merge_sc

def merge_sc_in_region(region, data_pop='plasma', resolution='5min', sc_keys=None, nose=False):

    """
    Merge spacecrafr pre-filtered to be in the solar wind or magnetosheath
    The spacecraft are merged on their measurement not (not lagged time)
    When there are multiple spacecraft with a measurement at the same time, the one closest to the Earth-Sun line is chosen
    """

    dfs_combined = []
    if sc_keys is None:
        sc_keys = filter_spacecraft.get(region)

    attrs = {}
    units = {}

    imported_sc = []

    for sc in sc_keys:

        try:
            df_sc = import_processed_data(region, dtype=data_pop, resolution=resolution, file_name=f'{region}_times_{sc}')
        except:
            continue # there was no data in the filtered region using the filter() script

        print(f'Importing {sc.upper()}.')
        imported_sc.append(sc)

        for key, val in df_sc.attrs.items():
            if key=='units':
                units.update(val)
            else:
                attrs.update({key: val})

        # Combining spacecraft with same column names so using '_sc' suffix to identify.
        df_sc.rename(columns={col: f'{col}_{sc}' for col in df_sc.columns}, inplace=True)
        dfs_combined.append(df_sc)

    attrs['units'] = units

    ###----------COMBINING----------###
    print('Combining spacecraft')
    df_wide = pd.concat(dfs_combined, axis=1)

    # Compute distance from Earth-Sun line
    rho = pd.DataFrame({sc: (df_wide[f'r_y_GSE_{sc}']**2 + df_wide[f'r_z_GSE_{sc}']**2) for sc in imported_sc})

    # Spacecraft with smallest distance given preference
    closest_sc = rho.idxmin(axis=1)

    result = []
    for sc in imported_sc:
        suffix = f'_{sc}'
        sc_cols = [col for col in df_wide.columns if col.endswith(suffix)]
        renamed = {col: col[:-len(suffix)] for col in sc_cols}
        subset = (df_wide.loc[closest_sc == sc, sc_cols].rename(columns=renamed))
        subset[f'sc_{region}'] = sc
        result.append(subset)

    df_combined = pd.concat(result).sort_index()
    df_combined.index.name = 'epoch'

    update_attributes(df_combined, attrs, region, '')

    print('Merging')
    df_combined.attrs['units'][f'sc_{region}'] = 'STRING'

    df_combined.drop(columns=['no_np_ratio', 'no_np_ratio_unc', 'no_np_ratio_count', 'nhe_np_ratio', 'nhe_np_ratio_unc', 'nhe_np_ratio_count'], inplace=True) # columns don't care about when combined

    # Write
    print('Writing combined to file...')
    out_dir = get_proc_directory(region, dtype=data_pop, resolution=resolution)
    write_to_cdf(df_combined, directory=out_dir, file_name=f'{region}_times_combined', reset_index=True)



def update_attributes(df, attrs_dict, region='sw', suffix=''):

    df.attrs = attrs_dict
    for temp in (f'T_tot{suffix}',f'T_tot_unc{suffix}'):
        df.attrs['units'][temp] = 'MK'

    df.attrs['units']['prop_time_s'] = 's'
    if region=='msh':
        df.attrs['units']['Delta B_theta'] = 'rad'
        df.attrs['units']['Delta B_z'] = '1'
