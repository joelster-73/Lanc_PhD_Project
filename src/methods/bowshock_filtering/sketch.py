# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 14:05:23 2025

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