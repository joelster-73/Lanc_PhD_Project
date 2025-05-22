# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:20:40 2025

@author: richarj2
"""
import os
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from uncertainties import ufloat

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

from .additions import create_half_circle_marker, plot_segments
from .formatting import custom_date_formatter, add_legend, array_to_string, add_figure_title, create_label
from .utils import save_figure

from ..processing.speasy.config import speasy_variables, colour_dict, all_spacecraft
from ..processing.speasy.retrieval import retrieve_data, retrieve_datum
from ..processing.omni.config import omni_spacecraft
from ..processing.utils import add_unit

from ..analysing.shocks import is_in_solar_wind, where_shock_intercept, when_shock_intercept, approximate_pressure, sort_positions
from ..coordinates.boundaries import msh_boundaries


def plot_vertical_line_unc(ax, time, uncertainty, label_info=None, colour='k'):

    minutes, seconds = divmod(int(uncertainty), 60) # uncertainty in seconds

    formatted_uncertainty = f'{minutes:02}:{seconds:02}'
    line_label = (
        time.strftime('%H:%M:%S') + r' $\pm$ ' + formatted_uncertainty
    )

    if label_info is not None:
        line_label = f'{label_info}: ' + line_label
    ax.axvline(x=time, c='k', ls='--', lw=0.5, label=line_label)
    dt = timedelta(seconds=uncertainty)
    ax.axvspan(time-dt, time+dt, color=colour, alpha=0.15)

def plot_all_shocks(shocks, parameter, time=None, time_window=20, print_shock=False, R_E=6370, show_verticals=True, shock_colour='red', position_var='R_GSE', start_printing=None):

    if time is not None:
        time_shock = time
        nearest_idx = shocks.index.searchsorted(time_shock, side='right')
        nearest_time = shocks.index[nearest_idx]
        shock = shocks.loc[nearest_time]
        plot_shock(shock, parameter, time_window, print_shock, R_E, show_verticals, shock_colour, position_var)

    else:

        script_dir = os.getcwd() # change to location of script __file__
        file_name = 'Shocks_not_processed.txt'
        file_path = os.path.join(script_dir, 'scripts', file_name)
        if not os.path.exists(file_path):
            with open(file_path, 'w') as my_file:
                my_file.write(f'Log created on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')

        for index, shock in shocks.iterrows():

            if start_printing is not None and index < start_printing: # where printing got up to
                continue

            try:
                plot_shock(shock, parameter, time_window, print_shock, R_E, show_verticals, shock_colour, position_var)

            except Exception as e:
                print(f'Issue with shock at time {index}.')
                with open(file_path, 'a') as my_file:
                    sc = shock['spacecraft'].upper()
                    my_file.write(f'{index.strftime("%Y-%m-%d %H:%M:%S")} not added ({sc}): {e}\n')

def plot_shock(shock, parameter, time_window=20, print_shock=False, R_E=6370, show_verticals=True, shock_colour='red', position_var='R_GSE'):

    shock_speed = shock['v_sh']
    shock_speed_unc = shock['v_sh_unc'] # Likely issue with shock in database
    if pd.isna(shock_speed):
        e = 'Shock speed is NaN.'
        raise Exception(e)
    elif pd.isna(shock_speed_unc):
        e = 'Shock speed uncertainty is NaN.'
        raise Exception(e)
    elif shock_speed <= 0:
        e = f'Shock speed is negative/zero ({shock_speed}).'
        raise Exception(e)
    elif shock_speed <= shock_speed_unc:
        e = f'Shock speed ({shock_speed}) is smaller than uncertainty ({shock_speed_unc}).'
        raise Exception(e)


    plot_shock_downstream(shock, parameter, time_window, print_shock, R_E, show_verticals, shock_colour)
    plot_shock_spacecraft_positions(shock, parameter, time_window, position_var, R_E, shock_colour)

def plot_shock_downstream(shock, parameter, time_window=20, print_shock=False, R_E=6370, show_verticals=True, shock_colour = 'red'):


    shock_time = shock.name.to_pydatetime()
    shock_time_unc = shock['time_s_unc']
    sc_L1 = shock['spacecraft'].upper()
    shock_type = shock['type']

    arrival_time      = shock_time + timedelta(seconds=shock['delay_s'])
    arrival_time_unc  = shock['delay_s_unc']

    start_time_up = shock_time-timedelta(minutes=time_window)
    end_time_up   = shock_time+timedelta(minutes=time_window)
    start_time_dw = arrival_time-timedelta(minutes=time_window)
    end_time_dw   = arrival_time+timedelta(minutes=time_window)

    df_detect = retrieve_data(parameter, sc_L1, speasy_variables, start_time_up, end_time_up)
    try:
        param_unit = df_detect.attrs['units'][parameter]
    except:
        param_unit = add_unit(parameter)

    ###-------------------INTERCEPT TIMES-------------------###

    intercept_times = {}
    for region in ['L1', 'Earth']:
        for sc in all_spacecraft.get(region, []):
            if (region == 'L1' and sc == sc_L1) or (region == 'Earth' and sc == 'OMNI'):
                continue

            coord, time = where_shock_intercept(shock, sc, speasy_variables, R_E)
            if time is None:
                continue

            in_sw = is_in_solar_wind(sc, time, speasy_variables)
            if in_sw:
                intercept_times[sc] = time

    pos_BS, _ = retrieve_datum('R_GSE', 'OMNI', speasy_variables, arrival_time)
    time_at_BS = when_shock_intercept(shock, pos_BS, unit='time')[0]
    intercept_times['OMNI'] = time_at_BS # Even if None, OMNI should be in dictionary

    intercept_times = dict(sorted(intercept_times.items(),
                                  key=lambda item: item[1] if item[1] is not None else datetime.max))


    fig, ax = plt.subplots(figsize=(12,8))

    ###-------------------SHOCK DATA-------------------###
    plot_vertical_line_unc(ax, shock_time, shock_time_unc, 'Shock Detected')
    plot_vertical_line_unc(ax, arrival_time, arrival_time_unc, 'Arrived at Earth')


    # Detector - plotted after for legend purposes
    if df_detect is not None: # In case issue with retrieving data
        plot_segments(ax, df_detect, colour_dict.get(sc_L1), sc_L1, parameter, lw=1.2, marker='.')

    ###-------------------OTHER SPACECRAFT-------------------###

    for ind, (sc, time) in enumerate(intercept_times.items()):

        if sc == 'OMNI':
            start, end = min(start_time_up,start_time_dw), max(end_time_up,end_time_dw)
        elif sc in all_spacecraft['L1']:
            start, end = start_time_up, end_time_up
        elif sc in all_spacecraft['Earth']:
            start, end = start_time_dw, end_time_dw

        df_sc = retrieve_data(parameter, sc, speasy_variables, start, end, sc!='OMNI')
        sc_label = f'{ind+1}. {sc}'

        if sc=='OMNI':
            sc_OMNI = []
            sc_OMNI_id = np.unique(df_sc['spacecraft'])
            for sc_id in sc_OMNI_id:
                if sc_id != 99: # bad data
                    sc_OMNI.append(omni_spacecraft.get(int(sc_id)))
            if len(sc_OMNI)>0:
                sc_label += f' [{", ".join(sc_OMNI)}]'

        if df_sc is not None and len(df_sc)>0:
            lw = 1.2 if sc=='OMNI' else 0.7
            plot_segments(ax, df_sc, colour_dict.get(sc), sc_label, parameter, lw=lw, marker='.')
            if show_verticals and time is not None:
                ax.axvline(time, c=colour_dict.get(sc), ls=':')

        elif show_verticals and time is not None:
            ax.axvline(time, c=colour_dict.get(sc), ls=':', label=sc_label)


    ###-------------------SHOCK ARROW-------------------###
    y_min, y_max = ax.get_ylim()
    arrow_height = y_min+0.94*(y_max-y_min)
    text_height  = y_min+0.96*(y_max-y_min)

    #shock_duration = timedelta(seconds=int(shock['v_sh']/2))
    shock_duration = timedelta(seconds=300)
    ax.annotate('',  xy=(shock_time + shock_duration, arrow_height), xytext=(shock_time, arrow_height),
            arrowprops=dict(facecolor=shock_colour, edgecolor=shock_colour, headwidth=8, headlength=10, width=2))
    shock_speed = ufloat(shock['v_sh'],shock['v_sh_unc'])
    shock_speed_label = f'${shock_speed:L}$ $\\mathrm{{km\\,s^{{-1}}}}$'
    ax.text(shock_time+timedelta(seconds=30), text_height, shock_speed_label,
        color=shock_colour, fontsize=10, ha='left', va='bottom')


    ###-------------------AXES LABELS AND TICKS-------------------###
    ax.set_ylabel(create_label(parameter, param_unit))

    formatter = FuncFormatter(custom_date_formatter)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlim(min(start_time_up,start_time_dw)-timedelta(minutes=5),
                max(end_time_up,end_time_dw)+timedelta(minutes=5))

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    min_time_diff = (pd.Timestamp(min(start_time_up,start_time_dw)) - pd.Timestamp(shock_time)).total_seconds()
    max_time_diff = (pd.Timestamp(max(end_time_up,end_time_dw)) - pd.Timestamp(shock_time)).total_seconds()

    min_time_diff = round(min_time_diff/600)*10
    max_time_diff = round(max_time_diff/600)*10

    time_diff_ticks = np.arange(min_time_diff, max_time_diff+1, 10)

    shock_time_numeric = mdates.date2num(shock_time)  # Convert shock_time to matplotlib's numeric format
    time_tick_positions = shock_time_numeric + time_diff_ticks / (24 * 60)  # Convert seconds to days for x-axis

    ax2.set_xticks(time_tick_positions)
    ax2.set_xticklabels([f'{"+" if t > 0 else ""}{t}m' if t != 0 else '0' for t in time_diff_ticks])

    add_legend(fig, ax, loc='upper center', anchor=(0.5,-0.1), rows=2)

    add_figure_title(fig, title=f'{shock_type} Shock recorded by {sc_L1} on {shock_time.strftime("%Y-%m-%d")}')
    plt.tight_layout()
    save_figure(fig, file_name=shock_time.strftime('%Y-%m-%d'), sub_directory='shocks')
    #plt.show()
    plt.close()



def plot_shock_spacecraft_positions(shock, parameter, time_window=20, position_var='R_GSE', R_E=6370, shock_colour='red'):

    shock_time = shock.name.to_pydatetime()
    sc_L1 = shock['spacecraft'].upper()
    shock_type = shock['type']

    arrival_time = shock_time + timedelta(seconds=shock['delay_s'])
    start_time_orbit = shock_time-timedelta(hours=24)
    end_time_orbit   = shock_time+timedelta(hours=24)

    #-----------POSITIONS IN X,Y,Z-----------#

    pos_L1    = np.array([shock['r_x_GSE'],shock['r_y_GSE'],shock['r_z_GSE']])
    pos_L1_dict = {'x': pos_L1[0], 'y': pos_L1[1], 'z': pos_L1[2]}

    shock_normal    = np.array([shock['Nx'], shock['Ny'], shock['Nz']])
    shock_direction = {'x': shock_normal[0], 'y': shock_normal[1], 'z': shock_normal[2]}
    shock_velocity  = {key: shock['v_sh'] * value/R_E for key, value in shock_direction.items()}

    pos_BS, _ = retrieve_datum('R_GSE', 'OMNI', speasy_variables, arrival_time)
    time_at_BS = when_shock_intercept(shock, pos_BS, unit='time')[0]

    #-----------OMNI PARAMETERS AND SPACECRAFT REGIONS-----------#

    Vsw_OMNI, _  = retrieve_datum('V_GSE', 'OMNI', speasy_variables, time_at_BS)
    if Vsw_OMNI is None:
        Vsw_OMNI = np.array([shock['v_x_GSE_dw'],shock['v_y_GSE_dw'],shock['v_z_GSE_dw']])
    vx, vy, vz   = Vsw_OMNI

    Pd_OMNI, _   = retrieve_datum('P_dyn', 'OMNI', speasy_variables, time_at_BS)
    if Pd_OMNI is None:
        Ni_OMNI = shock['ni_dw']
        Pd_OMNI = approximate_pressure(Ni_OMNI, Vsw_OMNI)

    intercept_pos = {}
    intercept_reg = {}
    for region in ['L1', 'Earth']:
        for sc in all_spacecraft.get(region, []):
            if (region == 'L1' and sc == sc_L1) or (region == 'Earth' and sc == 'OMNI'):
                continue

            coord, time = where_shock_intercept(shock, sc, speasy_variables, R_E)
            if coord is None:
                if region == 'L1':
                    coord, _ = retrieve_datum('R_GSE', sc, speasy_variables, shock_time)
                elif region == 'Earth':
                    coord, _ = retrieve_datum('R_GSE', sc, speasy_variables, arrival_time)

                if coord is None: # Doesn't intercept/no data
                    continue

            if time is None:
                time = when_shock_intercept(shock, coord)

            coord = {'x': coord[0], 'y': coord[1], 'z': coord[2]}

            intercept_pos[sc] = coord
            in_sw = is_in_solar_wind(sc, time, speasy_variables, Pd=Pd_OMNI, Vsw=Vsw_OMNI, pos=coord)
            intercept_reg[sc] = in_sw

    positions = sort_positions(intercept_pos, intercept_reg, shock)

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

        if pos_BS is not None:
            pos_BS_dict = {'x': pos_BS[0], 'y': pos_BS[1], 'z': pos_BS[2]}
            ax.scatter(pos_BS_dict.get(x_coord), pos_BS_dict.get(y_coord),
                       marker='x', color=colour_dict.get('OMNI'), label=f'BS Nose: {array_to_string(pos_BS)} $R_E$')

        ###-----SHOCK FRONT-----###
        num_secs = 300 # 5 minutes of shock travel

        dx = shock_velocity.get(x_coord) * num_secs
        dy = shock_velocity.get(y_coord) * num_secs

        perp_dx, perp_dy = -dy, dx
        scale = 0.5  # Scale the length of the perpendicular line
        perp_dx *= scale
        perp_dy *= scale

        ax.arrow(x0, y0, dx, dy,
                 head_width=2, head_length=1, fc='red', ec=shock_colour)

        ax.plot([x0 - perp_dx, x0 + perp_dx], [y0 - perp_dy, y0 + perp_dy], c=shock_colour, ls='--')

        speed_info = f'v = {int(shock["v_sh"])} $\\mathrm{{km\\,s^{{-1}}}}$'
        normal_info = f'$\\boldsymbol{{n}}$ = ({shock_direction["x"]:.1f}, {shock_direction["y"]:.1f}, {shock_direction["z"]:.1f})'
        ax.text(x0 + dx - 5, y0 + dy, speed_info+'\n'+normal_info,
                fontsize=12, color=shock_colour, ha='left', va='center')

        ###-----MAGNETOSHEATH-----###
        for surface, style in zip(('bs','mp'), ('-','--')):

            phi = np.pi/2
            if y_coord == 'z':
                phi = 0

            surface_coords = msh_boundaries('jelinek', surface, Pd=Pd_OMNI,
                                            v_sw_x=vx, v_sw_y=vy, v_sw_z=vz, phi=phi)
            surface_x = surface_coords.get(x_coord)
            surface_y = surface_coords.get(y_coord)

            nose = surface_coords.get('nose')
            nose_x = nose[0]
            nose_y = nose[1 if y_coord=='y' else 2]

            ax.plot(surface_x, surface_y, ls=style, lw=0.5, color=colour_dict.get('OMNI'))
            ax.scatter(nose_x, nose_y, s=2, color=colour_dict.get('OMNI'))
            ax.plot([0,nose_x],[0,nose_y], ls=':', lw=0.5, color=colour_dict.get('OMNI'), zorder=1)


        ###-------------------OTHER SPACECRAFT-------------------###

        for ind, (sc, coord) in enumerate(positions.items()):
            if sc in (sc_L1,'OMNI'):
                continue

            sc_label = sc
            in_sw = intercept_reg[sc]
            if in_sw:
                sc_label = f'{ind+1}. {sc}'

            ax.scatter(coord.get(x_coord), coord.get(y_coord), marker='x', color=colour_dict.get(sc), label=sc_label)

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

        add_figure_title(fig, title=f'{shock_type} Shock recorded by {sc_L1} on {shock_time.strftime("%Y-%m-%d")}', ax=ax)
        plt.tight_layout()
        save_figure(fig, file_name=shock_time.strftime('%Y-%m-%d')+'_'+plane, sub_directory='shocks')
        #plt.show()
        plt.close()

