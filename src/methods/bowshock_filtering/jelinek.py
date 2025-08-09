# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:59:45 2025

@author: richarj2
"""
import os
import numpy as np

from ...config import R_E

from ...processing.dataframes import merge_dataframes, add_df_units
from ...processing.writing import write_to_cdf

from ...coordinates.spatial import calc_bs_pos
from ...analysing.kobel import load_compression_ratios, are_points_above_line

def insert_bs_diff(df, df_bs, sc_key, rel=False):
    diffs = df_bs[f'r_{sc_key}'] - df_bs['r_BS']
    df['r_bs_diff'] = diffs.reindex(df.index)
    if rel:
        diffs_rel = diffs/df_bs['r_BS']
        df['r_bs_diff_rel'] = diffs_rel.reindex(df.index)
    df.attrs['units']['r_bs_diff'] = df_bs.attrs['units'].get('r_BS', None)



def find_best_buffer(df, r_diff_col, y1_col, y2_col, **kwargs):

    buff_min     = kwargs.get('buff_min',0)
    buff_max     = kwargs.get('buff_max',6)
    compression  = kwargs.get('compression',2) # Shield 1969 - wrong
    compressions = kwargs.get('compressions',None)

    buffers = np.linspace(buff_min,buff_max,501)
    num_total = np.empty(len(buffers))
    num_bad = np.empty(len(buffers))
    perc_bad = np.empty(len(buffers))

    df_sw = df[df[r_diff_col]>=0]

    if compressions is not None:
        B_imf, B_msh, _ = load_compression_ratios(compressions)

    for i, buffer in enumerate(buffers):
        df_out = df_sw[df_sw[r_diff_col]>buffer]

        num_total[i] = len(df_out)
        if compressions is None:
            num_bad[i] = np.sum(df_out[y1_col]/df_out[y2_col]>compression)
        else:
            num_bad[i] = np.sum(are_points_above_line(B_imf, B_msh, df_out[y2_col], df_out[y1_col]))

        if num_total[i] > 0:
            perc_bad[i] = num_bad[i]/num_total[i]*100
        else:
            perc_bad[i] = np.nan

    best_perc = np.nanmin(perc_bad)
    where_result = np.where(perc_bad==best_perc)[0]
    ind = int(where_result[0]) if where_result.size > 0 else where_result
    best_buff = buffers[ind]
    best_length = num_total[ind]
    print(f'Buffer: {best_buff:.2f}, {best_perc:.2g}%, {best_length:,}')

def generate_bs_df(df_sc, df_omni, out_dir, sc_key,
                   sample_interval='1min', time_col='epoch', overwrite=True):

    """
    df_sc needs to be 1-minute data to match OMNI
    """

    # Merges data with OMNI
    sc_omni = merge_dataframes(df_sc, df_omni, sc_key, 'OMNI', clean=False)
    df_bs = calc_bs_pos(sc_omni, sc_key=sc_key, time_col='index')
    add_df_units(df_bs)

    # Write to file
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_file = os.path.join(out_dir, f'{sc_key}_BS_positions.cdf')
    attributes = {'sample_interval': sample_interval, 'time_col': time_col, 'R_E': R_E}
    write_to_cdf(df_bs, output_file, attributes, overwrite=True, reset_index=True)
