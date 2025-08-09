# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:41:20 2025

@author: richarj2
"""
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from ...plotting.config import black, white
from ...plotting.formatting import add_legend, add_figure_title, dark_mode_fig
from ...plotting.utils import save_figure

from ...analysing.kobel import load_compression_ratios, are_points_above_line



def plot_best_buffer(df, r_diff_col, y1_col, y2_col, **kwargs):

    buff_min     = kwargs.get('buff_min',0)
    buff_max     = kwargs.get('buff_max',6)
    log_scale    = kwargs.get('log_scale',False)
    compression  = kwargs.get('compression',2) # Shield 1969 - wrong
    compressions = kwargs.get('compressions',None)
    data_name    = kwargs.get('data_name','"SW"')
    reference    = kwargs.get('reference',4)

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
    best_ind     = int(where_result[0]) if where_result.size > 0 else where_result
    best_buff    = buffers[best_ind]
    best_length  = num_total[best_ind]
    best_perc    = perc_bad[best_ind]

    ref_ind     = np.where(buffers==reference)[0][0]
    ref_length  = num_total[ref_ind]
    ref_perc    = perc_bad[ref_ind]

    fig, ax = plt.subplots()

    ###---------- PLOT MINUTES DATA ----------###

    # Add vertical lines
    ax.plot([], [], c=black, marker='o', markersize=6, markerfacecolor='w', markevery=density, label=f'{data_name} Data')
    ax.plot([], [], c='r', marker='^', markersize=6, markerfacecolor='w', markevery=density, label='"Contamination"')

    ax.axvline(reference, ls='-',  c='g', lw=2, label=f'{reference:.2f} $R_E$: {ref_perc:.2f}% $\\cdot$ {int(ref_length):,}')
    ax.axvline(best_buff, ls='--', c='b',       label=f'{best_buff:.2f} $R_E$: {best_perc:.2f}% $\\cdot$ {int(best_length):,}')

    ax.plot(buffers, num_total, c=black, marker='o', markersize=6, markerfacecolor='w', markevery=density)

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
    ax2.plot(buffers, perc_bad, c='r', marker='^', markersize=6, markerfacecolor='w', markevery=density)
    ax2.set_ylabel('Percentage Above Threshold', c=black)

    if log_scale:
        ax2.set_yscale('log')

    def percentage_formatter(value, _):
        return f'{value:.1f}%'
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(percentage_formatter))


    ###---------- PLOT ZOOMED BIT ----------###

    if not log_scale:

        # Inset plot (zoomed out, 0 to 100 percentiles)
        inset_ax = inset_axes(ax2, width='28%', height='28%', loc='upper right')

        first_ind = np.where(buffers==(best_buff-1))[0][0]
        try:
            last_ind = np.where(buffers==(best_buff+1))[0][0]
        except:
            last_ind = -1
            first_ind = np.where(buffers==(buffers[-1]-2))[0][0]


        last_buffers = buffers[first_ind:last_ind+1]   if last_ind != -1 else buffers[first_ind:]
        last_minutes = num_total[first_ind:last_ind+1] if last_ind != -1 else num_total[first_ind:]
        last_percent = perc_bad[first_ind:last_ind+1]  if last_ind != -1 else perc_bad[first_ind:]

        inset_ax.axvline(best_buff, ls='--', c='b')
        inset_ax.axvline(reference, ls='-', c='g')
        inset_ax.plot(last_buffers, last_minutes, c='k')

        inset_ax2 = inset_ax.twinx()
        inset_ax2.plot(last_buffers, last_percent, c='r')

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



