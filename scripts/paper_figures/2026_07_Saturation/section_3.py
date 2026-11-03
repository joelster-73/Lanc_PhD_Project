# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2

Spacecraft data.
"""

# %% uncertainties

from src.methods.saturation.plotting.uncertainties import plot_independent_uncertainties

_ = plot_independent_uncertainties('B_z_GSM', 'E_R', 'E_yz', 'S_x_GSE', regions=('sw','msh'))


# %% summary

# Information at end of Section
# For each region, say spacecraft used

from src.processing.reading import import_processed_data

region = 'sw'
resolution = '5min'
df_sw = import_processed_data(region, 'plasma', resolution, f'{region}_times_combined')

region = 'msh'
df_msh = import_processed_data(region, 'plasma', resolution, f'{region}_times_combined')

# %%
import numpy as np
import pandas as pd

def upper_limit_by_days(x, days, n_days=30):
    """Largest driver value that is reached on at least n_days distinct days."""
    daily_max = pd.Series(np.asarray(x)).groupby(np.asarray(days)).max()
    return daily_max.sort_values(ascending=False).iloc[n_days - 1]


from scipy.stats import percentileofscore


from collections import Counter

from src.methods.saturation.plotting.utils import def_param_names, mask_df, get_variable_range

from src.processing.reading import import_processed_data
def info(df, region='sw'):

    print(region)

    print(f'\nRows: {len(df):,}')
    print(f'Days: {len(df.index.normalize().unique()):,}\n')

    print('Spacecraft:')
    for sc, num in Counter(df[f'sc_{region}']).items():
        print(f'{sc}: \t{num:,}\t| {num/len(df)*100:.3g} %')
        times = df.loc[df[f'sc_{region}']==sc].index
        start = times.min().year
        end   = times.max().year
        print(f'      \t{start} - {end}')

    for param in ('B_z_GSM','E_R','E_yz','S_x_GSE'):
        unit = df.attrs.get('units').get(param)

        #condition = '<0'
        condition = None
        cond_col = 'B_z_GSM'

        _, param_count = def_param_names(df, param)

        if param in ('E_R', ):
            #print(f'Condition doesn\'t apply to {param}.')
            condition = None

        length_before = np.sum(~df[param].isna())

        limits, invert = get_variable_range(param, region, restrict=True)
        series = mask_df(df, param, limits=None, count_col=param_count, resolution=resolution, condition=condition, cond_col=cond_col, invert=invert)
        if invert:
            param = f'-{param}'
        series = series[param]

        length_after = np.sum(~series.isna())

        print(f'\n{param}\n----------')
        print(f'Low counts removed:\n\t{length_before-length_after:,} | {(1-length_after/length_before)*100:.3g} %')
        print(f'Percentiles [{unit}]:')
        for label, q in (('50th',50),('95.0',95),('99.0',99),('99.5',99.5),('max',100)):
            print(f'\t{label}: \t{series.quantile(q/100):.4g}')

        if invert:
            limits = [limits[-1], -limits[0]]
        print('Limits:', limits)



        print(f'         {percentileofscore(series,limits[1]):.4g} %')


info(df_sw, 'sw')
info(df_msh, 'msh')