# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 16:16:34 2025

@author: richarj2
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 12:21:25 2025

@author: richarj2
"""
import os
import numpy as np
import pandas as pd

from src.config import PROC_CLUS_DIR_MSH, PROC_CLUS_DIR_5VPS, OMNI_DIR, PROC_THEMIS_DIR, PROC_MMS_DIR
from src.processing.reading import import_processed_data
from src.processing.dataframes import merge_dataframes
from src.coordinates.boundaries import calc_msh_r_diff

column_names = {
    'r_x_name'  : 'r_x_GSE',
    'r_y_name'  : 'r_y_GSE',
    'r_z_name'  : 'r_z_GSE',
    'r_name'    : 'r_mag',
    'r_ax_name' : 'r_x_aGSE',
    'r_ay_name' : 'r_y_aGSE',
    'r_az_name' : 'r_z_aGSE',
    'v_x_name'  : 'V_x_GSE',
    'v_y_name'  : 'V_y_GSE',
    'v_z_name'  : 'V_z_GSE',
    'p_name'    : 'P_flow',
    'bz_name'   : 'B_z_GSM'
}




# %% Combined

# With Plasma
data_pop = 'with_plasma'
cluster_directory = PROC_CLUS_DIR_MSH

# Field only
data_pop = 'field_only'
cluster_directory = PROC_CLUS_DIR_5VPS
themis_directory  = PROC_THEMIS_DIR
mms_directory     = PROC_MMS_DIR


sample_intervals = ('1min','5min')
msh_keys = ('c1','m1','tha','thb','thc','thd','the')
pos_cols = ['r_MP','r_BS','r_phi','r_F']

param_map_pc = {k: k.replace('_sw', '_pc') for k in ['AE_sw','AL_sw','AU_sw', 'AE_17m_sw','SYM_D_sw','SYM_H_sw','ASY_D_sw','ASY_H_sw','PSI_P_10_sw','PSI_P_30_sw','PSI_P_60_sw']}



sample_interval = '1min'

# Solar wind data
omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_sw    = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

dfs_combined = []

sc = 'm1'

field_col = 'B_avg'
if sc in ('c1',):
    sc_dir = cluster_directory
elif sc in ('tha','thb','thc','thd','the'):
    sc_dir = os.path.join(themis_directory, sc)
elif sc in ('m1',):
    sc_dir = mms_directory
sc_dir = os.path.join(sc_dir, sample_interval)

# Magnetosheath data
df_msh = import_processed_data(sc_dir)

location_mask = (df_msh['r_x_GSE']>0)
#location_mask &= (np.abs(df_msh['r_z_GSE'])<5)
location_mask &= ~np.isnan(df_msh[field_col])

df_msh = df_msh.loc[location_mask]

if 'r_mag' not in df_msh:
    cols     = [f'r_{comp}_GSE' for comp in ('x','y','z')]

    r = np.linalg.norm(df_msh[cols].values, axis=1)
    df_msh.insert(0, 'r_mag', r)

    unc_cols = [f'r_{comp}_GSE_unc' for comp in ('x','y','z')]
    try:
        sigma_r = np.sqrt(
            ((df_msh[cols].values / r[:, None])**2 * df_msh[unc_cols].values**2).sum(axis=1)
        )
    except:
        sigma_r = np.nan
    df_msh.insert(1, 'r_mag_unc', sigma_r)

if 'r_mag_count_msh' in df_msh:
    df_msh.drop(columns=['r_mag_count_msh'],inplace=True)

# Combine
df_merged = merge_dataframes(df_sw, df_msh, suffix_1='sw', suffix_2=sc)
df_merged.columns = [param_map_pc.get(col,col) for col in df_merged.columns] # changes sw to pc for some omni
df_merged.attrs['units'] = {param_map_pc.get(col,col): df_merged.attrs['units'].get(col,col) for col in df_merged.attrs['units']}
df_merged_attrs = df_merged.attrs

# Filter for MSH
df_positions = calc_msh_r_diff(df_merged, 'BOTH', position_key=sc, data_key='sw', column_names=column_names)
df_merged = pd.concat([df_merged,df_positions[pos_cols]],axis=1)

print(np.sum((df_merged['r_F']>0)&(df_merged['r_F']<1)))

# %%

group_ranges_dict = {}
thresholds = [0, 1]
time_diff = '30min'

for target in thresholds:
    # Detect sign change around the threshold
    sign_change = (df_merged['r_F'].shift(1) - target) * (df_merged['r_F'] - target)
    cross_idx = sign_change[sign_change < 0].index

    cross_times = []

    # Interpolate crossing times
    for i in cross_idx:
        i_prev = df_merged.index[df_merged.index.get_loc(i) - 1]
        y0, y1 = df_merged.loc[i_prev, 'r_F'], df_merged.loc[i, 'r_F']
        t0, t1 = i_prev, i

        frac = (target - y0) / (y1 - y0)
        t_cross = t0 + (t1 - t0) * frac
        cross_times.append(t_cross)

    # Group close crossings
    groups = []
    if cross_times:
        start = cross_times[0]
        end = cross_times[0]

        for t in cross_times[1:]:
            if t - end <= pd.Timedelta(time_diff):
                end = t
            else:
                groups.append((start, end))
                start = end = t

        groups.append((start, end))

    group_ranges_dict[target] = groups

# Access group ranges
group_ranges_0 = group_ranges_dict[0]
group_ranges_1 = group_ranges_dict[1]


df_merged.rename(columns={col: f'{col}_{sc}' for col in pos_cols}, inplace=True) # adds _sc suffix
df_merged.dropna(how='all',inplace=True)

# %%
import pandas as pd
import matplotlib.pyplot as plt

from src.plotting.comparing.space_time import plot_series_against_time
from src.plotting.utils import save_figure



plotting_dict = {'df1': {'B_avg': {'colour': 'orange', 'error': 'B_avg_unc', 'marker': 'x', 'label': 'B_avg_omni'}},
                 'df2': {'B_avg': {'colour': 'blue', 'error': 'B_avg_unc', 'label': 'B_avg_mms1'}}
                 }

window = '60min'


# %% BS

for (earliest, latest) in group_ranges_1:

    fig, (ax,axr) = plt.subplots(2, 1, height_ratios=[4,1], sharex=True, dpi=400)

    mask_sw  = (df_sw.index>earliest-pd.Timedelta(window))  & (df_sw.index<latest+pd.Timedelta(window))
    mask_msh = (df_msh.index>earliest-pd.Timedelta(window)) & (df_msh.index<latest+pd.Timedelta(window))


    _ = plot_series_against_time(df_sw.loc[mask_sw], df_msh.loc[mask_msh], df_col_dict=plotting_dict, delta=sample_interval, fig=fig, ax=ax, return_objs=True)

    mask_merged = (df_merged.index>earliest-pd.Timedelta(window)) & (df_merged.index<latest+pd.Timedelta(window))

    x = df_merged.loc[mask_merged].index
    y = df_merged.loc[mask_merged, f'r_F_{sc}']

    # Shade regions where r_F > 1
    ax2 = ax.twinx()
    ax2.fill_between(x, 1, y, where=(y > 1), color='grey', alpha=0.3, label=r'$r>r_{BS}$')
    ax2.legend(loc='upper right')

    axr.plot(df_msh.loc[mask_msh,'r_mag'], c='b', label=r'$r_{sc}$')
    axr.plot(df_merged.loc[mask_merged,f'r_BS_{sc}'], c='r', label=r'$r_{BS}$')
    axr.set_ylabel(r'r [$\mathrm{R_E}$]')
    axr.legend()

    date = pd.to_datetime(earliest).floor('us').to_pydatetime().strftime('%Y-%m-%d')
    ax.set_title(date)

    plt.tight_layout()
    save_figure(fig, sub_directory='boundaries_bs', file_name=f'{date}')
    plt.show()
    plt.close()

# %% MP

for (earliest, latest) in group_ranges_0:

    fig, (ax,axr) = plt.subplots(2, 1, height_ratios=[4,1], sharex=True, dpi=400)

    mask_sw  = (df_sw.index>earliest-pd.Timedelta(window))  & (df_sw.index<latest+pd.Timedelta(window))
    mask_msh = (df_msh.index>earliest-pd.Timedelta(window)) & (df_msh.index<latest+pd.Timedelta(window))


    _ = plot_series_against_time(df_sw.loc[mask_sw], df_msh.loc[mask_msh], df_col_dict=plotting_dict, delta=sample_interval, fig=fig, ax=ax, return_objs=True)

    mask_merged = (df_merged.index>earliest-pd.Timedelta(window)) & (df_merged.index<latest+pd.Timedelta(window))

    x = df_merged.loc[mask_merged].index
    y = df_merged.loc[mask_merged, f'r_F_{sc}']

    # Shade regions where r_F < 0
    ax2 = ax.twinx()
    ax2.fill_between(x, y, 0, where=(y < 0), color='grey', alpha=0.3, label=r'$r<r_{MP}$')
    ax2.legend(loc='upper right')

    axr.plot(df_msh.loc[mask_msh,'r_mag'], c='b', label=r'$r_{sc}$')
    axr.plot(df_merged.loc[mask_merged,f'r_MP_{sc}'], c='lime', label=r'$r_{MP}$')
    axr.set_ylabel(r'r [$\mathrm{R_E}$]')
    axr.legend()

    date = pd.to_datetime(earliest).floor('us').to_pydatetime().strftime('%Y-%m-%d')
    ax.set_title(date)

    plt.tight_layout()
    save_figure(fig, sub_directory='boundaries_mp', file_name=f'{date}')
    plt.show()
    plt.close()
