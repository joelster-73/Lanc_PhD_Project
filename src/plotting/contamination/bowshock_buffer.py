# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:41:20 2025

@author: richarj2
"""


def plot_method_sketch(plane='x-rho', **kwargs):

    phi = kwargs.get('phi', np.pi/2)

    bs_colour = kwargs.get('bs_colour','g')
    c1_colour = kwargs.get('c1_colour','r')
    buff_colour = kwargs.get('buff_colour','c')

    if plane in ('x-rho','xy'):
        phi = np.pi/2
    elif plane in ('xz',):
        phi = 0

    ###-------------------BS BOUNDARY-------------------###
    bs_jel = bs_boundaries('jelinek', **kwargs)
    # vsw = -400, Pd=2.056 are defaults

    bs_x_coords = bs_jel.get('x')
    bs_y_coords = bs_jel.get('y')
    bs_z_coords = bs_jel.get('z')
    bs_p_coords = bs_jel.get('rho')

    bd_R0 = bs_jel.get('R0')
    alpha = bs_jel.get('alpha')

    fig, ax = plt.subplots()

    mod = -1
    if plane == 'xy':
        mod = 1
        y_neg = bs_y_coords<0
        ax.plot(bs_x_coords[y_neg], bs_y_coords[y_neg], linestyle='-', color=bs_colour, lw=3, label='Bow shock')
        y_label = r'$Y_{GSE}$ [$R_E$]'
    elif plane == 'xz':
        z_pos = bs_z_coords>0
        ax.plot(bs_x_coords[z_pos], bs_z_coords[z_pos], linestyle='-', color=bs_colour, lw=3, label='Bow shock')
        y_label = r'$Z_{GSE}$ [$R_E$]'
    elif plane == 'x-rho':
        y_neg = bs_y_coords<0
        ax.plot(bs_x_coords[y_neg], bs_p_coords[y_neg], linestyle='-', color=bs_colour, lw=3, label='Bow shock')
        y_label = r'$\sqrt{Y^2+Z^2}$ [$R_E$]'


    ###-------------------STAND-OFF POSITION-------------------###
    stand_off_x = bd_R0*np.cos(alpha)
    stand_off_y = bd_R0*np.sin(alpha)*np.sin(phi)
    stand_off_z = bd_R0*np.sin(alpha)*np.cos(phi)
    stand_off_p = np.sqrt(stand_off_y**2 + stand_off_z**2)
    #stand_off_y = stand_off_z

    if plane == 'xy':
        stand_off_y_pos = stand_off_y
    elif plane == 'xz':
        stand_off_y_pos = stand_off_z
    elif plane == 'x-rho':
        stand_off_y_pos = stand_off_p


    ###-------------------ABERRATED AXIS-------------------###
    ax.plot((0,1.4*stand_off_x),(0,1.4*stand_off_y_pos),ls='--',c='k',zorder=1,lw=1.5)
    ax.text(1.4*stand_off_x,1.4*stand_off_y_pos-mod*0.3,r'$X_{aGSE}$',size=16)

    arrow_properties = dict(arrowstyle='->', color='black', lw=1.5, shrinkA=0, shrinkB=0)
    ax.annotate('', xy=(1.45*stand_off_x, 1.45*stand_off_y_pos), xytext=(1.4*stand_off_x, 1.4*stand_off_y_pos), arrowprops=arrow_properties)


    ax.scatter(stand_off_x, stand_off_y_pos, c='k',zorder=3)
    ax.text(stand_off_x-0.5, stand_off_y_pos-mod*0.2, f'$R_0$ = {bd_R0:.1f} $R_E$', fontsize=10, color='k')
    ax.text(stand_off_x-0.5, stand_off_y_pos-mod*0.8, f'$\\alpha$ = {np.degrees(alpha):.1f}$^\\circ$', fontsize=10, color='k')


    ###-------------------CLUSTER POSITION-------------------###
    target_x = kwargs.get('target_x',9)

    closest_x = bs_x_coords[np.abs(bs_x_coords - target_x).argmin()]
    if plane == 'xy':
        matching_ys = bs_y_coords[bs_x_coords == closest_x]
        target_y = matching_ys[0]
    elif plane == 'x-rho':
        matching_ps = bs_p_coords[bs_x_coords == closest_x]
        target_y = matching_ps[0]

    # Cluster arrow
    scale = 1.25
    arrow_c1 = FancyArrow(
        0, 0, scale*closest_x, scale*target_y,
        width=0.1, head_width=0.6, head_length=0.6,
        facecolor=c1_colour, edgecolor=c1_colour
    )
    ax.add_patch(arrow_c1)
    ax.text(scale*target_x-2, scale*target_y+mod*1, r'$r_{C1}$', ha='center', color=c1_colour)

    triangle_centre = np.array([scale*closest_x+0.7,scale*target_y-mod*0.7])
    triangle_vertices = [triangle_centre+np.array([0,0.5*mod]),
                         triangle_centre+np.array([0.5,-0.5*mod]),
                         triangle_centre+np.array([-0.5,-0.5*mod])]
    triangle = Polygon(
        triangle_vertices,
        closed=True,
        facecolor='w',
        edgecolor=c1_colour,
        linewidth=2,
        label='Cluster'
    )
    ax.add_patch(triangle)

    ###-------------------BOW SHOCK ARROW-------------------###
    arrow_bs = FancyArrow(
        0, 0, 0.95*closest_x, 0.95*target_y,
        width=0.1, head_width=0.6, head_length=0.6,
        facecolor=bs_colour, edgecolor=bs_colour
    )
    ax.add_patch(arrow_bs)
    ax.text(target_x-2, target_y+mod, r'$r_{BS}$', ha='center', color=bs_colour)

    ax.text(3,-mod*0.5,r'$\theta$', ha='center', va='bottom', color=c1_colour, size=40)

    arrow_bs_angle = np.arctan(target_y/closest_x)
    radius=4.8
    if plane == 'xy':
        angle_arc = Arc((0, 0), width=radius, height=radius, theta1=np.degrees(arrow_bs_angle)+1.5, theta2=np.degrees(alpha), color=c1_colour, lw=2)
    elif plane == 'x-rho':
        angle_arc = Arc((0, 0), width=radius, height=radius, theta1=-np.degrees(alpha), theta2=np.degrees(arrow_bs_angle)-1.5, color=c1_colour, lw=2)


    ax.add_patch(angle_arc)
    ax.plot((0,radius*np.cos(alpha)/2),(0,mod*radius*np.sin(alpha)/2),lw=2,c=c1_colour,zorder=1)

    ###-------------------BUFFER ARROW-------------------###
    arrow_buff = FancyArrow(
        closest_x+0.5, target_y+mod*0.5, ((scale-1.15)*closest_x-0.1), ((scale-1.15)*target_y+mod*0.1),
        width=0.1, head_width=0.6, head_length=0.6,
        facecolor=buff_colour, edgecolor=buff_colour
    )
    ax.add_patch(arrow_buff)
    ax.text(closest_x+2, target_y, r'$r_{buff}$', ha='center', color=buff_colour)

    ###-------------------SOLAR WIND-------------------###
    create_half_circle_marker(ax, center=(0, 0), radius=1, full=True)

    start_x, start_y, change_x, change_y = 20, -mod*4.2, -4, 0

    arrow_sw = FancyArrow(
        x=start_x,
        y=start_y,
        dx=change_x,
        dy=change_y,
        width=1,
        head_width=1.75,
        head_length=1,
        edgecolor='k',
        facecolor='w',
        linewidth=2.5
    )
    ax.add_patch(arrow_sw)

    ax.text(start_x, start_y-mod*2.25, r'$v_{sw}\approx-400\,\mathrm{km\,s^{-1}}$', ha='left')
    ax.text(start_x, start_y-mod*1.5, r'$P_d\approx2.056\,\mathrm{nPa}$', ha='left')

    ###-------------------LABELS AND AXES-------------------###
    ax.set_xlabel(r'$X_{GSE}$ [$R_E$]')
    ax.set_ylabel(y_label,labelpad=-20)

    ax.axis('square')
    ax.set_xlim(right=20)

    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    ax.annotate('', xy=(20, 0), xytext=(19, 0), arrowprops=arrow_properties)
    ax.annotate('', xy=(0, -mod*20), xytext=(0, -mod*19), arrowprops=arrow_properties)

    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)


    ax.xaxis.set_inverted(True)
    if plane == 'xy':
        ax.set_ylim(bottom=-20)

        ax.spines['bottom'].set_bounds(-2.5, 19)
        ax.spines['left'].set_bounds(-19, 2.5)

        xticks = ax.get_xticks()
        yticks = ax.get_yticks()

        ax.set_xticks([tick for tick in xticks if tick > 0 and tick < 20])
        ax.set_yticks([tick for tick in yticks if tick < 0 and tick > -20])

        ax.yaxis.set_inverted(True)

    elif plane == 'x-rho':
        ax.set_ylim(top=20)

        ax.spines['bottom'].set_bounds(-2.5, 19)
        ax.spines['left'].set_bounds(-2.5, 19)

        xticks = ax.get_xticks()
        yticks = ax.get_yticks()

        ax.set_xticks([tick for tick in xticks if tick > 0 and tick < 20])
        ax.set_yticks([tick for tick in yticks if tick > 0 and tick < 20])

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.yaxis.set_label_position('right')

    #ax.legend()
    save_figure(fig)
    plt.show()
    plt.close()

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

        bs_jel = bs_boundaries('jelinek', Pd=np.median(pressures), vsw=np.median(velocities))

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

            bs_jel = bs_boundaries('jelinek', Pd=np.median(pressures), vsw=np.median(velocities))

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

def plot_best_buffer(df, r_diff_col, y1_col, y2_col, **kwargs):

    buff_min     = kwargs.get('buff_min',0)
    buff_max     = kwargs.get('buff_max',6)
    scale        = kwargs.get('scale','lin')
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

    def plot_buffers(log_scale=False):
        fig, ax = plt.subplots()

        ###---------- PLOT MINUTES DATA ----------###

        # Add vertical lines
        ax.plot([], [], c=black, marker='o', markersize=6, markerfacecolor='w', markevery=density, label=f'{data_name} Data')
        ax.plot([], [], c='r', marker='^', markersize=6, markerfacecolor='w', markevery=density, label='"Contamination"')

        ax.axvline(reference, ls='-',  c='g', lw=2, label=f'{reference:.2f} $R_E$: {ref_perc:.2f}% $\\cdot$ {int(ref_length):,}')
        ax.axvline(best_buff, ls='--', c='b',       label=f'{best_buff:.2f} $R_E$: {best_perc:.2f}% $\\cdot$ {int(best_length):,}')

        ax.plot(buffers, num_total, c=black, marker='o', markersize=6, markerfacecolor='w', markevery=density)

        # Set axis labels and formatting
        ax.set_xlabel('Buffer [$R_E$]', c=black)
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
        ax2.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))


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
            inset_ax2.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))


            # Add title and final touches
            add_figure_title(fig, 'Dataset "Contamination" against buffer size')
            dark_mode_fig(fig, black, white)
            plt.tight_layout()
            save_figure(fig)
            plt.show()
            plt.close()


    # Plot for the specified scales
    if scale in {'lin', 'both'}:
        plot_buffers(log_scale=False)

    if scale in {'log', 'both'}:
        plot_buffers(log_scale=True)


