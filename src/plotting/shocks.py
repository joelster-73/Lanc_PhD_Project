# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:20:40 2025

@author: richarj2
"""
import numpy as np
import pandas as pd
from datetime import timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

from .additions import create_half_circle_marker, plot_segments
from .formatting import custom_date_formatter, add_legend, array_to_string, add_figure_title, create_label
from .utils import save_figure

from ..processing.speasy.config import speasy_variables, colour_dict, few_spacecraft
from ..processing.speasy.retrieval import retrieve_data, retrieve_datum
from ..processing.omni.config import omni_spacecraft

from ..analysing.shocks.in_sw import is_in_solar_wind
from ..coordinates.boundaries import msh_boundaries


def plot_vertical_line_unc(ax, time, uncertainty, label_info=None, colour='k', uncertainty_tuple=None, return_label=False):

    minutes, seconds = divmod(int(uncertainty), 60) # uncertainty in seconds

    formatted_uncertainty = f'{minutes:02}:{seconds:02}'
    line_label = (
        time.strftime('%H:%M:%S') + r' $\pm$ ' + formatted_uncertainty
    )

    if label_info is not None:
        line_label = f'{label_info}: ' + line_label

    dt = timedelta(seconds=uncertainty)
    if uncertainty_tuple is not None:
        dt_left  = timedelta(seconds=uncertainty_tuple[0])
        dt_right = timedelta(seconds=uncertainty_tuple[1])
    else:
        dt_left, dt_right = dt, dt

    ax.axvspan(time-dt_left, time+dt_right, color=colour, alpha=0.08)

    if return_label:
        ax.axvline(x=time, c=colour, ls='--', lw=0.5)
        return line_label

    ax.axvline(x=time, c=colour, ls='--', lw=0.5, label=line_label)

def plot_all_shocks(shocks, parameter, time=None, time_window=20, position_var='R_GSE', R_E=6370, shock_colour='red', start_printing=None, plot_positions=False, plot_in_sw=False):

    if isinstance(shocks, pd.Series):
        plot_shock_times(shocks, parameter, time_window, position_var, R_E, shock_colour, plot_in_sw)

        if plot_positions:
            plot_shock_positions(shocks, parameter, position_var, R_E, shock_colour)

    elif time is not None:
        time_shock = time
        nearest_idx = shocks.index.searchsorted(time_shock, side='right')
        nearest_time = shocks.index[nearest_idx]
        shock = shocks.loc[nearest_time].copy()

        plot_shock_times(shock, parameter, time_window, position_var, R_E, shock_colour, plot_in_sw)

        if plot_positions:
            plot_shock_positions(shock, parameter, position_var, R_E, shock_colour)

    else:

        for index, shock in shocks.iterrows():

            if start_printing is not None and index < start_printing: # where printing got up to
                continue

            # plot_shock_times(shock, parameter, time_window, position_var, R_E, shock_colour, plot_in_sw)
            # if plot_positions:
            #     plot_shock_positions(shock, parameter, position_var, R_E, shock_colour)

            try:
                plot_shock_times(shock, parameter, time_window, position_var, R_E, shock_colour, plot_in_sw)
                if plot_positions:
                    plot_shock_positions(shock, parameter, position_var, R_E, shock_colour)

            except Exception as e:
                print(f'Issue with shock at time {index}: {e}.')

def plot_shock_times(shock, parameter, time_window=20, position_var='R_GSE', R_E=6370, shock_colour='red', plot_in_sw=False, plot_full_range=False):


    shock_time = shock.name.to_pydatetime()
    shock_time_unc = shock['time_s_unc']
    sc_L1 = shock['spacecraft'].upper()
    source_code = shock['source']
    database = 'CFA' if source_code=='C' else 'Donki'

    start_times   = []
    end_times     = []

    spacecraft_times = {}

    for region in ['L1', 'Earth']:
        for source in few_spacecraft.get(region, []):

            time = shock[f'{source}_time']

            plot_data = False
            plot_vertical = False
            lw = 0.8
            
            if pd.isnull(time):

                if source in ('WIND','ACE','DSC'):
                    time = shock_time
                else:
                    time = shock_time + timedelta(hours=1)

                if source == 'OMNI':
                    plot_data = True
                elif plot_in_sw:

                    approx_start = time-timedelta(minutes=time_window)
                    approx_end   = time+timedelta(minutes=time_window)

                    df_pos = retrieve_data(position_var, source, speasy_variables, approx_start, approx_end, upsample=True)
                    if df_pos.empty:
                        continue
                    in_sw = is_in_solar_wind(source, speasy_variables, approx_start, approx_end, pos_df=df_pos, shock=shock)
                    if np.any(~in_sw):
                        continue
                    plot_data = True
                    lw = 0.6

            else:
                plot_data = True
                plot_vertical = True

            if plot_data:
                spacecraft_times[source] = (time, plot_vertical)

    spacecraft_times = dict(sorted(
        spacecraft_times.items(),
        key=lambda item: (not item[1][1], item[1][0])  # Sort by boolean (True first), then by time
    ))

    ###-------------------SHOCK DATA-------------------###
    fig, ax = plt.subplots(figsize=(12,8))

    plot_vertical_line_unc(ax, shock_time, shock_time_unc, 'Shock Detected')

    min_time = min(value[0] for value in spacecraft_times.values())
    max_time = max(value[0] for value in spacecraft_times.values())

    for source, (time, plot_vertical) in spacecraft_times.items():
        if source == 'OMNI' or plot_full_range:
            start = min_time-timedelta(minutes=time_window)
            end   = max_time+timedelta(minutes=time_window)
        else:
            start = time-timedelta(minutes=time_window)
            end   = time+timedelta(minutes=time_window)
        time_unc = shock[f'{source}_time_unc_s']

        if source in(sc_L1,'OMNI'):
            lw = 1.2

        df_param = retrieve_data(parameter, source, speasy_variables, start, end, downsample=True)
        if df_param.empty:
            continue

        sc_label = f'{source}'
        if source != sc_L1:
            if source == 'OMNI':
                omni_sc = omni_spacecraft.get(int(shock['OMNI_sc']),int(shock['OMNI_sc']))
                sc_label += f' [{omni_sc}]'

            coeff = shock[f'{source}_coeff']
            if ~np.isnan(coeff):
                sc_label += f' ({coeff:.2f})'

            if plot_vertical:
                sc_label = plot_vertical_line_unc(ax, time, time_unc, sc_label, colour_dict.get(source), return_label=True)

        plot_segments(ax, df_param, colour_dict.get(source), sc_label, parameter, lw=lw, marker='.')

        start_times.append(start)
        end_times.append(end)


    # ###-------------------SHOCK ARROW-------------------###
    # y_min, y_max = ax.get_ylim()
    # arrow_height = y_min+0.94*(y_max-y_min)
    # text_height  = y_min+0.96*(y_max-y_min)

    # #shock_duration = timedelta(seconds=int(shock['v_sh']/2))
    # shock_duration = timedelta(seconds=300)
    # ax.annotate('',  xy=(shock_time + shock_duration, arrow_height), xytext=(shock_time, arrow_height),
    #         arrowprops=dict(facecolor=shock_colour, edgecolor=shock_colour, headwidth=8, headlength=10, width=2))
    # shock_speed = ufloat(shock['v_sh'],shock['v_sh_unc'])
    # shock_speed_label = f'${shock_speed:L}$ $\\mathrm{{km\\,s^{{-1}}}}$'
    # ax.text(shock_time+timedelta(seconds=30), text_height, shock_speed_label,
    #     color=shock_colour, fontsize=10, ha='left', va='bottom')


    ###-------------------AXES LABELS AND TICKS-------------------###
    _, param_unit = retrieve_datum(parameter, sc_L1, speasy_variables, shock_time)
    ax.set_ylabel(create_label(parameter, param_unit))

    formatter = FuncFormatter(custom_date_formatter)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(min(start_times), max(end_times))

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    min_time_diff = (pd.Timestamp(min(start_times)) - pd.Timestamp(shock_time)).total_seconds()
    max_time_diff = (pd.Timestamp(max(end_times)) - pd.Timestamp(shock_time)).total_seconds()

    time_range = (max_time_diff - min_time_diff)/60
    if time_range < 60:
        diff_step = 5
    elif time_range > 120:
        diff_step = 20
    else:
        diff_step = 10
    min_time_diff = int(min_time_diff/(diff_step*60))*diff_step
    max_time_diff = int(max_time_diff/(diff_step*60))*diff_step

    time_diff_ticks = np.arange(min_time_diff, max_time_diff+1, diff_step)

    shock_time_numeric = mdates.date2num(shock_time)  # Convert shock_time to matplotlib's numeric format
    time_tick_positions = shock_time_numeric + time_diff_ticks / (24 * 60)  # Convert seconds to days for x-axis

    ax2.set_xticks(time_tick_positions)
    ax2.set_xticklabels([f'{"+" if t > 0 else ""}{t}m' if t != 0 else '0' for t in time_diff_ticks])

    add_legend(fig, ax, loc='upper center', anchor=(0.5,-0.1), cols=3)

    add_figure_title(fig, title=f'Shock recorded by {sc_L1} on {shock_time.strftime("%Y-%m-%d")} ({database})')
    plt.tight_layout()
    #save_figure(fig, file_name=shock_time.strftime('%Y-%m-%d')+'_t', sub_directory='shocks')
    plt.show()
    plt.close()



def plot_shock_positions(shock, parameter, position_var='R_GSE', R_E=6370, shock_colour='red'):

    shock_time = shock.name.to_pydatetime()
    sc_L1 = shock['spacecraft'].upper()
    source_code = shock['source']
    database = 'CFA' if source_code=='C' else 'Donki'

    start_time_orbit = shock_time-timedelta(hours=24)
    end_time_orbit   = shock_time+timedelta(hours=24)

    omni_time = shock['OMNI_time']
    if pd.isnull(omni_time):
        omni_time = shock_time+timedelta(hours=1)

    #-----------POSITIONS IN X,Y,Z-----------#

    pos_L1    = np.array([shock[f'{sc_L1}_r_x_GSE'],shock[f'{sc_L1}_r_y_GSE'],shock[f'{sc_L1}_r_z_GSE']])
    pos_L1_dict = {'x': pos_L1[0], 'y': pos_L1[1], 'z': pos_L1[2]}

    # shock_normal    = np.array([shock['Nx'], shock['Ny'], shock['Nz']])
    # shock_direction = {'x': shock_normal[0], 'y': shock_normal[1], 'z': shock_normal[2]}
    # shock_velocity  = {key: shock['v_sh'] * value/R_E for key, value in shock_direction.items()}

    pos_BS, _ = retrieve_datum('R_GSE', 'OMNI', speasy_variables, omni_time)

    #-----------OMNI PARAMETERS AND SPACECRAFT REGIONS-----------#

    Vsw_OMNI, _  = retrieve_datum('V_GSE', 'OMNI', speasy_variables, omni_time)

    Pd_OMNI, _   = retrieve_datum('P_dyn', 'OMNI', speasy_variables, omni_time)

    spacecraft_positions = {}
    for region in ['L1', 'Earth']:
        for source in few_spacecraft.get(region, []):
            if source in (sc_L1,'OMNI'):
                continue

            time = shock[f'{source}_time']
            coord = shock[[f'{source}_r_x_GSE',f'{source}_r_y_GSE',f'{source}_r_z_GSE']].to_numpy()
            
            if not pd.isnull(time):
                spacecraft_positions[source] = (time,{'x': coord[0], 'y': coord[1], 'z': coord[2]},True)

            else:
                if source in ('WIND','ACE','DSC'):
                    time = shock_time
                else:
                    time = omni_time
                pos, _ = retrieve_datum('R_GSE', source, speasy_variables, time)
                if pos is not None:
                    spacecraft_positions[source] = (time,{'x': pos[0], 'y': pos[1], 'z': pos[2]},False)

    positions = dict(sorted(
        spacecraft_positions.items(),
        key=lambda item: (not item[1][2], item[1][0])  # Sort by boolean (True first), then by time
    ))

    orbits = {}
    for sc in ('C1','THA','THB'):
        orbits[sc] = retrieve_data(position_var, sc, speasy_variables,
                              start_time_orbit, end_time_orbit, downsample=False)

    #-----------LOOP THROUGH PLANES-----------#

    planes = ('xy','xz')
    for plane in planes:

        x_coord = plane[0]
        y_coord = plane[1]

        fig, ax = plt.subplots(figsize=None)

        ###-------------------DETECTOR AND OMNI-------------------###
        x0 = pos_L1_dict.get(x_coord)
        y0 = pos_L1_dict.get(y_coord)

        ax.scatter(x0, y0, marker='x', color=colour_dict.get(sc_L1), label=f'{sc_L1}: {array_to_string(pos_L1)} $R_E$')
        ax.plot([0,x0], [0,y0], color=colour_dict.get(sc_L1), lw=0.5, ls=':', zorder=1)

        if pos_BS is not None:
            pos_BS_dict = {'x': pos_BS[0], 'y': pos_BS[1], 'z': pos_BS[2]}
            ax.scatter(pos_BS_dict.get(x_coord), pos_BS_dict.get(y_coord),
                       marker='x', color=colour_dict.get('OMNI'), label=f'BS Nose: {array_to_string(pos_BS)} $R_E$')

        ###-----SHOCK FRONT-----###
        # num_secs = 300 # 5 minutes of shock travel

        # dx = shock_velocity.get(x_coord) * num_secs
        # dy = shock_velocity.get(y_coord) * num_secs

        # perp_dx, perp_dy = -dy, dx
        # scale = 0.5  # Scale the length of the perpendicular line
        # perp_dx *= scale
        # perp_dy *= scale

        # ax.arrow(x0, y0, dx, dy,
        #          head_width=2, head_length=1, fc='red', ec=shock_colour)

        # ax.plot([x0 - perp_dx, x0 + perp_dx], [y0 - perp_dy, y0 + perp_dy], c=shock_colour, ls='--')

        # speed_info = f'v = {int(shock["v_sh"])} $\\mathrm{{km\\,s^{{-1}}}}$'
        # normal_info = f'$\\boldsymbol{{n}}$ = ({shock_direction["x"]:.1f}, {shock_direction["y"]:.1f}, {shock_direction["z"]:.1f})'
        # ax.text(x0 + dx - 5, y0 + dy, speed_info+'\n'+normal_info,
        #         fontsize=12, color=shock_colour, ha='left', va='center')

        ###-----MAGNETOSHEATH-----###
        for surface, style in zip(('bs','mp'), ('-','--')):

            phi = np.pi/2
            if y_coord == 'z':
                phi = 0

            kwargs = {'phi': phi}
            if Pd_OMNI is not None:
                kwargs['Pd'] = Pd_OMNI
            if Vsw_OMNI is not None:
                kwargs['v_sw_x'], kwargs['v_sw_y'], kwargs['v_sw_z'] = Vsw_OMNI

            surface_coords = msh_boundaries('jelinek', surface, **kwargs)
            surface_x = surface_coords.get(x_coord)
            surface_y = surface_coords.get(y_coord)

            nose = surface_coords.get('nose')
            nose_x = nose[0]
            nose_y = nose[1 if y_coord=='y' else 2]

            ax.plot(surface_x, surface_y, ls=style, lw=0.5, color=colour_dict.get('OMNI'))
            ax.scatter(nose_x, nose_y, s=2, color=colour_dict.get('OMNI'))
            ax.plot([0,nose_x],[0,nose_y], ls=':', lw=0.5, color=colour_dict.get('OMNI'), zorder=1)


        ###-------------------OTHER SPACECRAFT-------------------###

        ind = 1
        for sc, (time,coord,label_number) in positions.items():
            if sc in (sc_L1,'OMNI'):
                continue
            if label_number:
                sc_label = f'{ind}. {sc}'
                ind += 1
            else:
                sc_label = sc

            ax.scatter(coord.get(x_coord), coord.get(y_coord), marker='+', color=colour_dict.get(sc), label=sc_label)

            orbit = orbits.get(sc, None)
            if orbit is None:
                continue

            rx = orbit.get(position_var+f'_{x_coord}')
            ry = orbit.get(position_var+f'_{y_coord}')

            ax.plot(rx, ry, c=colour_dict.get(sc), ls=':', lw=0.4)
            arrow_step = len(rx) // 4
            for i in range(0, len(rx)-1, arrow_step):
                dx = rx.iloc[i + 1] - rx.iloc[i]
                dy = ry.iloc[i + 1] - ry.iloc[i]
                norm = np.sqrt(dx**2 + dy**2)
                if norm != 0:
                    dx /= norm
                    dy /= norm
                ax.quiver(rx.iloc[i], ry.iloc[i], dx, dy, angles='xy', scale_units='xy',
                          scale=0.5, width=0.005, color=colour_dict.get(sc))

        ###-------------------AXES LABELS AND TICKS-------------------###
        create_half_circle_marker(ax, center=(0, 0), radius=1, full=True)

        ax.axhline(y=0, color='grey', ls='--', lw=0.2)
        ax.axvline(x=0, color='grey', ls='--', lw=0.2)

        ax.xaxis.set_inverted(True)
        if y_coord == 'y':
            ax.yaxis.set_inverted(True)
        ax.set_xlabel(x_coord.upper()+r' GSE [$R_E$]')
        ax.set_ylabel(y_coord.upper()+r' GSE [$R_E$]')

        plt.gca().set_aspect('equal')

        add_legend(fig, ax, loc='upper center', anchor=(0.5,-0.25), rows=3)

        add_figure_title(fig, title=f'Shock recorded by {sc_L1} on {shock_time.strftime("%Y-%m-%d")} ({database})', ax=ax)
        plt.tight_layout()
        #save_figure(fig, file_name=shock_time.strftime('%Y-%m-%d')+'_'+plane, sub_directory='shocks')
        plt.show()
        plt.close()