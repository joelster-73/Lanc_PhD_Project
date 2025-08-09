# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:41:20 2025

@author: richarj2
"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt


from ...plotting.config import black, white
from ...plotting.additions import create_half_circle_marker
from ...plotting.formatting import add_legend, add_figure_title, create_label, dark_mode_fig
from ...plotting.utils import save_figure, calculate_bins, save_frame, save_gif

from ...analysing.comparing import difference_columns
from ...analysing.kobel import load_compression_ratios, are_points_above_line

from ...coordinates.boundaries import msh_boundaries



def plot_grid_bowshock_buffer(df, r_diff_col, y1_col, y2_col, rx_col, ryz_col, **kwargs):

    buff_min     = kwargs.get('buff_min',0)
    buff_max     = kwargs.get('buff_max',6)
    buff_stp     = kwargs.get('buff_stp',1)
    buffers      = range(buff_min, buff_max, buff_stp)

    compression  = kwargs.get('compression',2) # Shield 1969 - wrong
    compressions = kwargs.get('compressions',None)
    bin_width    = kwargs.get('bin_width',0.1)


    r_diff_name  = kwargs.get('r_diff_name',None)
    y_diff_name  = kwargs.get('y_diff_name',None)
    y1_name      = kwargs.get('y1_name',None)
    y2_name      = kwargs.get('y2_name',None)
    rx_name      = kwargs.get('rx_name',None)
    ryz_name     = kwargs.get('ryz_name',None)

    n_cols = 3
    n_rows = len(buffers)
    fig, axs = plt.subplots(
        n_rows, n_cols, sharex='col', sharey='col', figsize=(n_cols*6, n_rows*4)
    )
    #x_col = r_bs_diff_C1 here
    y_col='diff'
    df[y_col] = np.abs(difference_columns(df, y1_col, y2_col))
    df_sw = df[df[r_diff_col]>=0]

    units = df.attrs['units']
    #param_unit = units.get(y1_col,'')

    if compressions is not None:
        B_imf, B_msh, _ = load_compression_ratios(compressions)

    ###-------------------PLOT ROWS-------------------###
    for ax_row, buffer in zip(axs, buffers):

        df_out = df_sw[df_sw[r_diff_col]>buffer]

        # Loops:
        # (1) Plot against radial distance difference
        # (2) Compare B parameter
        # (3) Spatial Distribution

        ###-------------------PLOT COLUMNS-------------------###

        for i, (ax, x_label, y_label, x_name, y_name, bin_ratio) in enumerate(zip(ax_row, (r_diff_col, y2_col, rx_col), (y_col, y1_col, ryz_col),
                                                                       (r_diff_name, y2_name, rx_name), (y_diff_name, y1_name, ryz_name), (4,2,1))):

            x_axis_label = create_label(x_label,None,x_name,True,units)
            y_axis_label = create_label(y_label,None,y_name,True,units)

            if i==0:
                n_bins = (calculate_bins(df_sw[x_label],bin_width),calculate_bins(df_sw[y_label],bin_width*bin_ratio))
                h = ax.hist2d(df_sw[x_label], df_sw[y_label], bins=n_bins, norm=mpl.colors.LogNorm(), cmap='hot')
                ax.axvspan(0, buffer, alpha=0.4, color='k')
            else:
                n_bins = (calculate_bins(df_out[x_label],bin_width),calculate_bins(df_out[y_label],bin_width*bin_ratio))
                h = ax.hist2d(df_out[x_label], df_out[y_label], bins=n_bins, norm=mpl.colors.LogNorm(), cmap='hot')

            ax.set_xlabel(x_axis_label,c=black)
            ax.set_ylabel(y_axis_label,c=black)

            cbar = fig.colorbar(h[3], ax=ax)
            cbar.ax.tick_params(colors=black)
            cbar.outline.set_edgecolor(black)

        # (1) First column
        ax_row[0].set_xlim(left=0,right=10)
        ax_row[0].set_ylim(top=40)
        ax_row[0].axvline(x=buffer,c='w',ls='--',label=f'{buffer} $R_E$')

        add_figure_title(fig,f'{buffer} $R_E$ Buffer',ax=ax_row[0])

        # (2) Second column

        if compressions is None:
            num_ext = np.sum(df_out[y1_col]/df_out[y2_col]>compression)
        else:
            num_ext = np.sum(are_points_above_line(B_imf, B_msh, df_out[y2_col], df_out[y1_col]))

        perc_ext = num_ext/len(df_out)*100

        ax_row[1].axline((0, 0), slope=1, color='w', label='y=x', ls=':') # y=x
        if compressions is not None:
            ax_row[1].plot(B_imf, B_msh, color='cyan')
        ax_row[1].set_xlim(left=0,right=25)
        ax_row[1].set_ylim(bottom=0,top=50)
        add_figure_title(fig,f'{perc_ext:.2g}%, {num_ext:,} mins',ax=ax_row[1])


        # (3) Third column
        pressures = df_out[df_out.index.isin(df.index)]['p_flow_OMNI']
        pressures = pressures[~np.isnan(pressures)]
        velocities = df_out[df_out.index.isin(df.index)]['v_x_GSE_OMNI']
        velocities = velocities[~np.isnan(velocities)]

        bs_jel = msh_boundaries('jelinek', 'bs', Pd=np.median(pressures), vsw=np.median(velocities))

        bs_x_coords = bs_jel.get('x')
        bs_y_coords = bs_jel.get('rho')
        y_pos = bs_jel.get('y')>0

        ax_row[2].plot(bs_x_coords[y_pos], bs_y_coords[y_pos], linestyle='-', color='lime')

        create_half_circle_marker(ax, center=(0, 0), radius=1, full=False)
        ax_row[2].set_xlim(left=0)
        ax_row[2].set_ylim(bottom=0)
        add_figure_title(fig,f'{len(df_out):,} mins',ax=ax_row[2])
        plt.gca().invert_xaxis()

    dark_mode_fig(fig,black,white,heat=True)
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()


def plot_buffer_gif(df, r_diff_col, y1_col, y2_col, bin_width=0.1, **kwargs):

    buff_min     = kwargs.get('buff_min',0)
    buff_max     = kwargs.get('buff_max',5.5)
    buff_stp     = kwargs.get('buff_stp',0.5)
    buffers      = np.arange(buff_min, buff_max, buff_stp)

    frame_time   = kwargs.get('frame_time',1)

    compressions = kwargs.get('compressions',None)

    r_diff_name  = kwargs.get('r_diff_name',None)
    y_diff_name  = kwargs.get('y_diff_name',None)
    y1_name      = kwargs.get('y1_name',None)
    y2_name      = kwargs.get('y2_name',None)

    rx_col       = kwargs.get('rx_col',None)
    ryz_col      = kwargs.get('ryz_col',None)
    rx_name      = kwargs.get('rx_name',None)
    ryz_name     = kwargs.get('ryz_name',None)

    plot_type    = kwargs.get('plot_type','Difference') # Compare, Difference, Orbit
    units = df.attrs['units']

    if plot_type == 'Difference':
        x_label = r_diff_col
        y_label = 'diff'
        x_name  = r_diff_name
        y_name  = y_diff_name
        units['diff'] = units[y1_col]

        df[y_label] = np.abs(difference_columns(df, y1_col, y2_col))

    elif plot_type == 'Orbit':
        x_label = rx_col
        y_label = ryz_col
        x_name  = rx_name
        y_name  = ryz_name

    else:
        x_label = y2_col
        y_label = y1_col
        x_name  = y2_name
        y_name  = y1_name

    df_sw = df[df[r_diff_col]>=0]


    x_axis_label = create_label(x_label,None,x_name,True,units)
    y_axis_label = create_label(y_label,None,y_name,True,units)

    if compressions is not None:
        B_imf, B_msh, _ = load_compression_ratios(compressions)

    frame_files = []

    ###-------------------PLOT ROWS-------------------###
    for i, buffer in enumerate(buffers):

        fig, ax = plt.subplots()

        # Data outside buffered bowshock
        df_out = df_sw[df_sw[r_diff_col]>buffer]

        if plot_type == 'Difference':
            n_bins = (calculate_bins(df_sw[x_label],bin_width),calculate_bins(df_sw[y_label],bin_width))
            h = ax.hist2d(df_sw[x_label], df_sw[y_label], bins=n_bins, norm=mpl.colors.LogNorm(), cmap='hot')
            ax.axvspan(0, buffer, alpha=0.4, color='k')
            ax.set_xlim(left=0)
            ax.set_ylim(top=40)
            ax.axvline(x=buffer,c='w',ls='--')
            add_figure_title(fig,f'{buffer} $R_E$ Buffer',ax=ax)

        else:
            n_bins = (calculate_bins(df_out[x_label],bin_width),calculate_bins(df_out[y_label],bin_width))
            h = ax.hist2d(df_out[x_label], df_out[y_label], bins=n_bins, norm=mpl.colors.LogNorm(), cmap='hot')

        if plot_type == 'Compare':
            ax.axline((0, 0), slope=1, color='w', ls=':') # y=x
            ax.set_xlim(right=30)
            ax.set_ylim(top=50)

            if compressions is not None:
                num_ext = np.sum(are_points_above_line(B_imf, B_msh, df_out[y2_col], df_out[y1_col]))
                ax.plot(B_imf, B_msh, color='cyan', ls=':')

                perc_ext = num_ext/len(df_out)*100
                add_figure_title(fig,f'{perc_ext:.2g}%, {num_ext:,} mins',ax=ax)

        elif plot_type == 'Orbit':
            pressures = df_out[df_out.index.isin(df.index)]['p_flow_OMNI']
            pressures = pressures[~np.isnan(pressures)]
            velocities = df_out[df_out.index.isin(df.index)]['v_x_GSE_OMNI']
            velocities = velocities[~np.isnan(velocities)]

            bs_jel = msh_boundaries('jelinek', 'bs', Pd=np.median(pressures), vsw=np.median(velocities))

            bs_x_coords = bs_jel.get('x')
            bs_y_coords = bs_jel.get('rho')

            #ax.set_aspect('equal', adjustable='box')
            ax.plot(bs_x_coords, bs_y_coords, linestyle='-', color='lime')

            create_half_circle_marker(ax, center=(0, 0), radius=1, full=False)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            add_figure_title(fig,f'{len(df_out):,} mins',ax=ax)
            plt.gca().invert_xaxis()

        ax.set_xlabel(x_axis_label,c=black)
        ax.set_ylabel(y_axis_label,c=black)

        cbar = fig.colorbar(h[3], ax=ax)
        cbar.ax.tick_params(colors=black)
        cbar.outline.set_edgecolor(black)

        add_legend(fig, ax,heat=True)
        dark_mode_fig(fig,black,white)
        ax.set_facecolor('k')
        plt.tight_layout()
        save_frame(fig, i, frame_files)
        plt.close()

    save_gif(frame_files, length=frame_time)



