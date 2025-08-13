# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:59:45 2025

@author: richarj2
"""
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ...config import R_E

from ...processing.dataframes import merge_dataframes, add_df_units
from ...processing.writing import write_to_cdf

from ...coordinates.boundaries import calc_msh_r_diff
from ...analysing.kobel import load_compression_ratios, are_points_above_line

from ...plotting.config import black, white
from ...plotting.formatting import add_legend, add_figure_title, dark_mode_fig
from ...plotting.utils import save_figure

def insert_bs_diff(df, df_bs, sc_key, rel=False):
    diffs = df_bs[f'r_{sc_key}'] - df_bs['r_BS']
    df['r_bs_diff'] = diffs.reindex(df.index)
    if rel:
        diffs_rel = diffs/df_bs['r_BS']
        df['r_bs_diff_rel'] = diffs_rel.reindex(df.index)
    df.attrs['units']['r_bs_diff'] = df_bs.attrs['units'].get('r_BS', None)


def generate_bs_df(df_sc, df_omni, out_dir, sc_key,
                   sample_interval='1min', time_col='epoch', overwrite=True):

    """
    df_sc needs to be 1-minute data to match OMNI
    """

    # Merges data with OMNI
    sc_omni = merge_dataframes(df_sc, df_omni, sc_key, 'OMNI', clean=False)
    df_bs = calc_msh_r_diff(sc_omni, 'BS', position_key=sc_key, data_key='OMNI')
    add_df_units(df_bs)

    # Write to file
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_file = os.path.join(out_dir, f'{sc_key}_BS_positions.cdf')
    attributes = {'sample_interval': sample_interval, 'time_col': time_col, 'R_E': R_E}
    write_to_cdf(df_bs, output_file, attributes, overwrite=True, reset_index=True)


def find_best_buffer(df, r_diff_col, y1_col, y2_col, plot=True, **kwargs):

    buff_min     = kwargs.get('buff_min',0)
    buff_max     = kwargs.get('buff_max',6)
    log_scale    = kwargs.get('log_scale',False)
    compressions = kwargs.get('compressions',None)
    reference    = kwargs.get('reference',-1)

    density = 100
    nsteps = density*(buff_max - buff_min) + 1
    buffers = np.linspace(buff_min,buff_max,nsteps)
    num_total = np.empty(len(buffers))
    num_bad = np.empty(len(buffers))
    perc_bad = np.empty(len(buffers))

    df_sw = df[df[r_diff_col]>=0]

    if compressions is not None:
        B_imf, B_msh, _ = load_compression_ratios(compressions)

    for i, buffer in enumerate(buffers):
        df_out = df_sw[df_sw[r_diff_col]>buffer]

        num_total[i] = len(df_out)
        num_bad[i] = np.sum(are_points_above_line(B_imf, B_msh, df_out[y2_col], df_out[y1_col]))

        if num_total[i] > 0:
            perc_bad[i] = num_bad[i]/num_total[i]*100
        else:
            perc_bad[i] = np.nan

    best_perc = np.nanmin(perc_bad)

    where_result = np.where(perc_bad==best_perc)[0]
    best_ind     = int(where_result[0]) if where_result.size > 0 else where_result
    best_buff    = buffers[best_ind]
    best_length  = num_total[best_ind]
    best_perc    = perc_bad[best_ind]

    print(f'Buffer: {best_buff:.2f}, {best_perc:.2g}%, {int(best_length):,}\n')

    if not plot:
        return


    fig, ax = plt.subplots()

    ###---------- PLOT MINUTES DATA ----------###

    colour_mins=black
    colour_perc='r'
    ax.plot(buffers, num_total, c=colour_mins, label='Minutes')
    ax.plot([], [], ls='--', c=colour_perc, label='Percentage') # for legend

    # Add vertical lines

    if reference != -1:
        ref_ind     = np.where(buffers==reference)[0][0]
        ref_length  = num_total[ref_ind]
        ref_perc    = perc_bad[ref_ind]
        ax.axvline(reference, ls=':',  c='g', lw=2, label=f'{reference:.2f} $\\mathrm{{R_E}}$: {ref_perc:.2f}% $\\cdot$ {int(ref_length):,}')

    ax.axvline(best_buff, ls=':', c='b', lw=2, label=f'Min: {best_perc:.2f}%, {best_buff:.2f} $\\mathrm{{R_E}}$')

    # Set axis labels and formatting
    ax.set_xlabel(r'Buffer [$\mathrm{R_E}$]', c=black)
    ax.set_ylabel('Minutes in Dataset', c=black)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))
    if log_scale:
        ax.set_yscale('log')

    add_legend(fig, ax,loc='upper left', anchor=(0.175,1.0))
    ax.grid(ls=':')

    ###---------- PLOT PERCENTAGE DATA ----------###

    ax2 = ax.twinx()
    ax2.plot(buffers, perc_bad, ls='--', c=colour_perc, label='Percentage')
    ax2.set_ylabel('Percentage Above Threshold', c=black)

    if log_scale:
        ax2.set_yscale('log')

    def percentage_formatter(value, _):
        return f'{value:.1f}%'
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(percentage_formatter))


    ###---------- PLOT ZOOMED BIT ----------###

    if not log_scale:

        # Inset plot (zoomed out, 0 to 100 percentiles)
        inset_ax = inset_axes(ax2, width='30%', height='30%', loc='upper right')

        first_ind = np.where(buffers==(best_buff-1))[0][0]
        try:
            last_ind = np.where(buffers==(best_buff+1))[0][0]
        except:
            last_ind = -1
            first_ind = np.where(buffers==(buffers[-1]-2))[0][0]


        last_buffers = buffers[first_ind:last_ind+1]   if last_ind != -1 else buffers[first_ind:]
        last_minutes = num_total[first_ind:last_ind+1] if last_ind != -1 else num_total[first_ind:]
        last_percent = perc_bad[first_ind:last_ind+1]  if last_ind != -1 else perc_bad[first_ind:]

        inset_ax.axvline(best_buff, ls=':', c='b', lw=2)
        inset_ax.plot(last_buffers, last_minutes, c=colour_mins)

        inset_ax2 = inset_ax.twinx()
        inset_ax2.plot(last_buffers, last_percent, c=colour_perc)

        inset_ax.set_xticks([last_buffers[0], best_buff, last_buffers[-1]])
        inset_ax.set_yticks([])
        inset_ax2.set_yticks([np.min(last_percent), np.max(last_percent)])
        inset_ax2.yaxis.tick_left()
        inset_ax.tick_params(axis='x', which='both', labelsize=9)
        inset_ax2.tick_params(axis='y', which='both', labelsize=9)

        def percentage_formatter(value, _):
            return f'{value:.2f}%'
        inset_ax2.yaxis.set_major_formatter(mticker.FuncFormatter(percentage_formatter))


    # Add title and final touches
    add_figure_title(fig, black, 'Dataset "Contamination" against buffer size')
    dark_mode_fig(fig, black, white)
    plt.tight_layout()
    save_figure(fig)

    plt.show()
    plt.close()