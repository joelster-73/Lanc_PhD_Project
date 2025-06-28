# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""

import numpy as np
import matplotlib.pyplot as plt

from uncertainties import ufloat
from uncertainties import unumpy as unp
from datetime import timedelta

from .intercepts import find_propagation_time
from ..fitting import straight_best_fit

from ...plotting.utils import save_figure
from ...processing.speasy.config import colour_dict, speasy_variables
from ...processing.speasy.retrieval import retrieve_modal_omni_sc


# %% Training Functions

def train_algorithm_buffers(df_shocks, event_list, buffer_up_range=range(25,36), buffer_dw_range=range(10,26), coeff_lim=0.7):

    buffer_up_values = list(buffer_up_range)
    buffer_dw_values = list(buffer_dw_range)

    slopes = np.zeros((len(buffer_dw_values), len(buffer_up_values)))
    counts = np.zeros((len(buffer_dw_values), len(buffer_up_values)))
    r2_val = np.zeros((len(buffer_dw_values), len(buffer_up_values)))
    diffs  = np.zeros((len(buffer_dw_values), len(buffer_up_values)))

    for j, buffer_up in enumerate(buffer_up_values):
        for i, buffer_dw in enumerate(buffer_dw_values):

            events_data = analyse_all_events(df_shocks, event_list, buffer_up, buffer_dw)
            if events_data is None:
                for struc in (slopes,counts,r2_val,diffs):
                    struc[i, j] = np.nan
                continue

            helsinki_delays, correlated_delays, correlated_uncs, coefficients = events_data


            ###-------------------CLOSEST-------------------###
            coeff_mask = coefficients>=coeff_lim

            x_vals     = helsinki_delays[coeff_mask]
            y_vals     = correlated_delays[coeff_mask]
            y_uncs     = correlated_uncs[coeff_mask]

            slope, intercept, r2 = straight_best_fit(x_vals,y_vals,y_uncs,detailed=True)

            slopes[i, j] = slope.n
            counts[i, j] = len(x_vals)
            r2_val[i, j] = r2

    return slopes, counts, r2_val


def train_algorithm_param(df_shocks, event_list, vary='buffer_up', vary_array=None, buffer_up=28, buffer_dw=34, dist_buff=60, min_ratio_change=0.8, coeff_lim=0.7):

    if (vary=='buffer_up') and (vary_array is None):
        vary_array = range(25,36)
    elif (vary=='buffer_dw') and (vary_array is None):
        vary_array = range(20,41)
    elif (vary=='dist_buff') and (vary_array is None):
        vary_array = np.arange(10,71,5)
    elif (vary=='min_ratio') and (vary_array is None):
        vary_array = np.arange(0.5,0.91,0.01)
    else:
        raise Exception(f'Not valid parameter for vary: {vary}')

    num_shocks = np.zeros(len(vary_array))
    slope_R2   = np.zeros(len(vary_array))
    slope_fit  = np.zeros(len(vary_array))

    for i, ind in enumerate(vary_array):

        if vary=='buffer_up':
            buffer_up = ind
        elif vary=='buffer_dw':
            buffer_dw = ind
        elif vary=='dist_buff':
            dist_buff = ind
        elif vary=='min_ratio':
            min_ratio = ind

        helsinki_delays, correlated_delays, coefficients, shock_times, detectors, interceptors, modal_omni_sc = analyse_all_events_more_info(df_shocks, event_list, buffer_up, buffer_dw, distance_buff=dist_buff, min_ratio_change=min_ratio)

        ###-------------------MINIMUM CROSS-CORR-------------------###

        coeff_mask = coefficients>=coeff_lim

        xs      = unp.nominal_values(helsinki_delays[coeff_mask])
        ys      = unp.nominal_values(correlated_delays[coeff_mask])
        ys_unc  = unp.std_devs(correlated_delays[coeff_mask])

        slope, intercept, r2 = straight_best_fit(xs,ys,ys_unc,detailed=True)

        num_shocks[i]= len(xs)
        slope_R2[i] = r2
        slope_fit[i] = slope.n

    return slope_fit, num_shocks, slope_R2, vary_array



# %% Analysis_functions


def analyse_all_events(df_shocks, event_list, buffer_up, buffer_dw):

    correlated_delays = []
    helsinki_delays   = []
    coefficients      = []
    correlated_uncs   = []

    for event in event_list:

        for upstream in ('ACE','WIND','DSC'):
            up_time_u = event.get(upstream,None)
            if up_time_u is None:
                continue
            up_time, up_unc = up_time_u
            # consider removing position vector in the shock df
            up_pos  = df_shocks.loc[up_time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
            for downstream in ('OMNI','C1','C3','C4'):
                dw_time_u = event.get(downstream,None)
                if dw_time_u is None:
                    continue
                dw_time, dw_unc = dw_time_u

                dw_pos  = df_shocks.loc[dw_time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
                if np.sum(np.abs(dw_pos)>=9999)>1: # Bad data flag
                    dw_pos = None

                helsinki_delay = (dw_time-up_time).total_seconds()

                best_lag = find_propagation_time(up_time, upstream, downstream, 'B_mag', position=up_pos, buffer_up=buffer_up, buffer_dw=buffer_dw, intercept_pos=dw_pos)
                if best_lag is None:
                    continue
                delay, coeff = best_lag

                delay_unc = np.sqrt(up_unc**2+(delay.s)**2)
                correlated_delays.append(delay.n)
                correlated_uncs.append(delay_unc)
                helsinki_delays.append(helsinki_delay)
                coefficients.append(coeff)

    if len(correlated_delays)<=2:
        return None

    helsinki_delays   = np.array(helsinki_delays)/60
    correlated_delays = np.array(correlated_delays)/60
    correlated_uncs   = np.array(correlated_uncs)/60
    coefficients      = np.array(coefficients)

    return helsinki_delays, correlated_delays, correlated_uncs, coefficients

def analyse_all_events_more_info(df_shocks, event_list, buffer_up, buffer_dw, **kwargs):


    correlated_delays = []
    helsinki_delays   = []
    coefficients      = []

    shock_times      = []
    detectors        = []
    interceptors     = []
    modal_omni_sc    = []

    for event in event_list:

        for upstream in ('ACE','WIND','DSC'):
            up_time_u = event.get(upstream,None)
            if up_time_u is None:
                continue
            up_time, up_unc = up_time_u
            # consider removing position vector in the shock df
            up_pos  = df_shocks.loc[up_time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
            for downstream in ('OMNI','C1','C3','C4'):
                dw_time_u = event.get(downstream,None)
                if dw_time_u is None:
                    continue
                dw_time, dw_unc = dw_time_u

                dw_pos = df_shocks.loc[dw_time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
                if np.sum(np.abs(dw_pos)>=9999)>1: # Bad data flag
                    dw_pos = None

                best_lag = find_propagation_time(up_time, upstream, downstream, 'B_mag', position=up_pos, buffer_up=buffer_up, buffer_dw=buffer_dw, intercept_pos=dw_pos, **kwargs)
                if best_lag is None:
                    continue

                shock_times.append(up_time_u)

                helsinki_delay     = (dw_time-up_time).total_seconds()
                helsinki_delay_unc = np.sqrt(up_unc**2+dw_unc**2)
                helsinki_delays.append(ufloat(helsinki_delay,helsinki_delay_unc))

                delay, coeff = best_lag
                delay       -= ufloat(0,up_unc)
                correlated_delays.append(delay)

                coefficients.append(coeff)

                omni_sc = retrieve_modal_omni_sc(speasy_variables, dw_time-timedelta(minutes=60), dw_time+timedelta(minutes=60))

                detectors.append(upstream)
                interceptors.append(downstream)
                modal_omni_sc.append(omni_sc)

    if len(correlated_delays)<=2:
        raise Exception(f'Not enough delays found: {len(correlated_delays)}')

    helsinki_delays   = np.array(helsinki_delays)/60
    correlated_delays = np.array(correlated_delays)/60
    coefficients      = np.array(coefficients)

    shock_times    = np.array(shock_times)
    detectors      = np.array(detectors)
    interceptors   = np.array(interceptors)
    modal_omni_sc  = np.array(modal_omni_sc)

    return helsinki_delays, correlated_delays, coefficients, shock_times, detectors, interceptors, modal_omni_sc


# %% Plotting_procedures

def plot_buffer_training(structures, limits=None, buffer_up_range=range(25,36), buffer_dw_range=range(30,41), coeff_lim=0.7, num_events=0):


    buffer_up_values = list(buffer_up_range)
    buffer_dw_values = list(buffer_dw_range)


    if limits is not None:

        limits_ind = ((buffer_dw_values.index(limits[1][0]),buffer_dw_values.index(limits[1][1])),
                      (buffer_up_values.index(limits[0][0]),buffer_up_values.index(limits[0][1])))

        up_lims = range(limits[0][0],limits[0][1])
        dw_lims = range(limits[1][0],limits[1][1])

    else:
        up_lims = buffer_up_values
        dw_lims = buffer_dw_values

    for name, struc in structures.items():
        # Plot heatmap
        fig, ax = plt.subplots()
        ax.set_facecolor('k')

        structure = struc
        if limits is not None:
            structure = struc[limits_ind[0][0]:limits_ind[0][1],limits_ind[1][0]:limits_ind[1][1]].copy()

        struc_zeros = structure.copy()
        struc_zeros[np.isnan(struc_zeros)] = 0
        best = np.max(struc_zeros)

        max_coords = np.unravel_index(np.argmax(struc_zeros), struc_zeros.shape)
        x_coord = up_lims[max_coords[1]]+0.5
        y_coord = dw_lims[max_coords[0]]+0.5

        heat = ax.imshow(structure, cmap='Blues_r', aspect='auto',
                  extent=[min(up_lims), max(up_lims)+1,
                          max(dw_lims)+1, min(dw_lims)])

        if limits is not None:
            min_val = np.min(structure[~np.isnan(structure)])
            max_val = np.max(structure[~np.isnan(structure)])
            mid_val = (min_val+max_val)/2

            for i in range(len(structure)):
                for j in range(len(structure[0])):
                    value = structure[i, j]

                    x = up_lims[j]+0.5
                    y = dw_lims[i]+0.5

                    if np.isnan(value):
                        tc = 'k'
                    elif value == best:
                        tc = 'r'
                    elif value < mid_val:
                        tc = 'w'
                    else:
                        tc = 'k'

                    ax.text(x, y, f'{value:.4g}', color=tc, ha='center', va='center', fontsize=12)
        else:
            ax.scatter(x_coord, y_coord, color='red', label=f"Max: {best:.2f}")


        cbar = plt.colorbar(heat)
        name_label = name if name!='R2' else r'$R^2$'
        cbar.set_label(name_label)
        ax.invert_yaxis()

        ax.set_xticks(np.array(up_lims)+0.5)
        ax.set_xticklabels(up_lims)

        ax.set_yticks(np.array(dw_lims)+0.5)
        ax.set_yticklabels(dw_lims)

        ax.get_xticklabels()[max_coords[1]].set_color('red')
        ax.get_yticklabels()[max_coords[0]].set_color('red')


        if limits is not None:
            ax.set_xlim(limits[0])
            ax.set_ylim(limits[1])

        ax.set_aspect('equal')

        ax.set_xlabel('Buffer up [mins]')
        ax.set_ylabel('Buffer dw [mins]')
        ax.set_title(f'Best fit {name_label} of {num_events} events; $\\rho\\geq${coeff_lim}')

        plt.tight_layout()
        save_figure(fig, file_name=f'{name}_up_{min(buffer_up_values)}_{max(buffer_up_values)}_dw_{min(buffer_dw_values)}_{max(buffer_dw_values)}')
        plt.show()
        plt.close()

def plot_comparison(xs_u, ys_u, coeffs, sc_ups, sc_dws, **kwargs):

    coeff_lim       = kwargs.get('coeff_lim',0.7)
    colouring       = kwargs.get('colouring','coeff')
    modal_omni      = kwargs.get('modal_omni',None)
    title_info_dict = kwargs.get('title_info_dict',{})

    title_info_dict['coeff_lim'] = coeff_lim
    title_info_dict['colouring'] = colouring

    coeff_mask = coeffs>=coeff_lim

    xs      = unp.nominal_values(xs_u[coeff_mask])
    xs_unc  = unp.std_devs(xs_u[coeff_mask])
    ys      = unp.nominal_values(ys_u[coeff_mask])
    ys_unc  = unp.std_devs(ys_u[coeff_mask])
    coeffs  = coeffs[coeff_mask]

    sc_ups  = sc_ups[coeff_mask]
    sc_dws  = sc_dws[coeff_mask]

    omni_mode = np.char.upper(modal_omni[coeff_mask]) if modal_omni is not None else None

    slope, intercept, r2 = straight_best_fit(xs,ys,ys_unc,detailed=True)


    fig, ax = plt.subplots()

    ax.errorbar(xs, ys, xerr=xs_unc, yerr=ys_unc, fmt='.', ms=0, ecolor='k', capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

    if colouring=='coeff':
        scatter = ax.scatter(xs, ys, c=coeffs, s=20, marker='x', cmap='plasma_r', vmin=coeff_lim, vmax=1)
        ax.scatter([], [], c='k', s=20, marker='x', label=f'{len(xs)}')

        cbar = plt.colorbar(scatter)
        cbar.set_label('Correlation Coeff')
    elif colouring=='sc':
        marker_dict = {'WIND': 'x', 'ACE': '+', 'DSC': '^'}

        for sc_up in ('WIND','ACE','DSC'):
            for sc_dw in ('OMNI','Cluster'):
                if sc_dw=='Cluster':
                    sc_mask = (sc_ups==sc_up)&(np.isin(sc_dws, ('C1','C3','C4')))
                    sc_c    = colour_dict['C1']
                elif sc_dw=='OMNI':
                    sc_mask = (sc_ups==sc_up)&(sc_dws==sc_dw)
                    sc_c    = colour_dict[sc_dw]
                    if omni_mode is not None:
                        sc_c = np.where(omni_mode[sc_mask]==sc_up,colour_dict[sc_dw],'r')
                count = np.sum(sc_mask)
                if count==0:
                    continue
                ax.scatter(xs[sc_mask], ys[sc_mask], c=sc_c, s=20, marker=marker_dict[sc_up], label=f'{sc_up} | {sc_dw}: {count}')


    ax.axline(slope=1,xy1=[0,0],c='k',ls=':')
    ax.axline([0,intercept.n],slope=slope.n,c='r',ls='--',lw=1)

    ax.axhline(y=0,c='grey',lw=0.2, ls='--')
    ax.axvline(x=0,c='grey',lw=0.2, ls='--')

    if intercept.n<0:
        sign = '-'
    else:
        sign = '+'

    low_lim = min(ax.get_xlim()[0],ax.get_ylim()[0])
    high_lim = max(ax.get_xlim()[1],ax.get_ylim()[1])

    middle = (low_lim+high_lim)/2
    height = ax.get_ylim()[1]-10

    ax.text(middle,height,f'$\\Delta t_c$ = (${slope:L}$)$\\cdot$$\\Delta t_H$\n{sign} (${abs(intercept):L}$) mins, $R^2$={r2:.3f}', ha='center',va='center')

    ax.set_xlabel('Helsinki delays [mins]')
    ax.set_ylabel('Correlated delays [mins]')

    ax.set_xlim(low_lim,high_lim)
    ax.set_ylim(low_lim,high_lim)

    ax.legend(loc='upper left', fontsize=8, title=f'{np.sum(coeff_mask)} times')
    ax.set_aspect('equal')

    title = create_title('Comparing Delay Times', title_info_dict)
    ax.set_title(title)
    plt.tight_layout()

    file_name = create_file_name('Comparing_Delay_Times', title_info_dict)
    save_figure(fig, file_name=file_name)
    plt.show()
    plt.close()

import matplotlib.dates as mdates

def plot_differences_over_time(xs_u, ys_u, times, coeffs, sc_ups, sc_dws, **kwargs):

    coeff_lim       = kwargs.get('coeff_lim',0.7)
    colouring       = kwargs.get('colouring','coeff')
    modal_omni      = kwargs.get('modal_omni',None)
    title_info_dict = kwargs.get('title_info_dict',{})

    title_info_dict['coeff_lim'] = coeff_lim
    title_info_dict['colouring'] = colouring

    marker_dict = {'WIND': 'x', 'ACE': '+', 'DSC': '^'}

    coeff_mask = coeffs>=coeff_lim

    diffs   = (ys_u-xs_u)[coeff_mask]
    ds      = unp.nominal_values(diffs)
    ds_unc  = unp.std_devs(diffs)
    ts      = np.array([t[0] for t in times[coeff_mask]])
    ts_unc  = np.array([t[1] for t in times[coeff_mask]])

    ts_unc = np.array(ts_unc) / (24 * 3600) # converts from s to days
    ts = mdates.date2num(ts)

    coeffs  = coeffs[coeff_mask]

    sc_ups  = sc_ups[coeff_mask]
    sc_dws  = sc_dws[coeff_mask]

    omni_mode = np.char.upper(modal_omni[coeff_mask]) if modal_omni is not None else None

    fig, ax = plt.subplots()

    ax.errorbar(ts, ds, yerr=ds_unc, fmt='.', ms=0, ecolor='k', capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

    if colouring=='coeff':
        scatter = ax.scatter(ts, ds, c=coeffs, s=20, marker='x', cmap='plasma_r', vmin=coeff_lim, vmax=1)
        ax.scatter([], [], c='k', s=20, marker='x', label=f'{len(ds)}')

        cbar = plt.colorbar(scatter)
        cbar.set_label('Correlation Coeff')
    elif colouring=='sc':

        for sc_up in ('WIND','ACE','DSC'):
            for sc_dw in ('OMNI','Cluster'):
                if sc_dw=='Cluster':
                    sc_mask = (sc_ups==sc_up)&(np.isin(sc_dws, ('C1','C3','C4')))
                    sc_c    = colour_dict['C1']
                elif sc_dw=='OMNI':
                    sc_mask = (sc_ups==sc_up)&(sc_dws==sc_dw)
                    sc_c    = colour_dict[sc_dw]
                    if omni_mode is not None:
                        sc_c = np.where(omni_mode[sc_mask]==sc_up,colour_dict[sc_dw],'r')
                count = np.sum(sc_mask)
                if count==0:
                    continue
                ax.scatter(ts[sc_mask], ds[sc_mask], c=sc_c, s=20, marker=marker_dict[sc_up], label=f'{sc_up} | {sc_dw}: {count}')


    ax.axhline(y=0,c='k',ls=':')

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    ax.set_xlabel('time')
    ax.set_ylabel('Correlated - Helsinki delays [mins]')

    ax.legend(loc='upper left', fontsize=8, title=f'{np.sum(coeff_mask)} times')

    title = create_title('Comparing Delay Times', title_info_dict)
    ax.set_title(title)
    plt.tight_layout()

    file_name = create_file_name('Comparing_Delay_Differences', title_info_dict)
    save_figure(fig, file_name=file_name)
    plt.show()
    plt.close()

def plot_single_param_vary(independent, slopes_fit=None, counts=None, slopes_R2=None, ind_var='Independent Variable', title_info_dict={}):

    if slopes_fit is None and counts is None and slopes_R2 is None:
        raise Exception('No data to plot.')

    if ind_var=='buffer_up':
        x_label = 'Buffer up [mins]'
    elif ind_var=='buffer_dw':
        x_label = 'Buffer dw [mins]'
    elif ind_var=='dist_buff':
        x_label = 'Distances [Re]'
    elif ind_var=='min_ratio':
        x_label = r'B-Ratio_{dw} $\geq x\cdot$ B-Ratio_{up}'


    fig, ax = plt.subplots()

    if slopes_fit is not None:
        ax.plot(independent,slopes_R2,c='b',marker='o',label='R2')
    if slopes_R2 is not None:
        ax.plot(independent,slopes_fit,c='r',marker='o',label='grad')

    if counts is not None:
        ax2 = ax.twinx()
        ax2.plot(independent,counts,c='g',marker='o',label='num')

        ax.set_xlabel(x_label)
        ax.set_ylabel('R2 & Gradient')
        ax2.set_ylabel('Num shocks')

        ax2_ticks = ax2.get_yticks()
        ax2.set_yticks(ax2_ticks)
        new_y_labels = [f'{int(tick)} ({(tick / counts[0]) * 100:.0f}%)' if tick >= 0 else '' for tick in ax2_ticks]
        ax2.set_yticklabels(new_y_labels)

    title = create_title(f'Varying  {ind_var}', title_info_dict)
    ax.set_title(title)
    plt.tight_layout()

    file_name = create_file_name(f'Vary_{ind_var}', title_info_dict)
    save_figure(fig, file_name=file_name)

    plt.show()
    plt.close()

# %%

def create_title(start, title_info_dict):
    title = start + ';'

    for i, (key, value) in enumerate(title_info_dict.items()):

        if key=='buffer_up':
            title += f' Buffer up: {value} mins;'
        elif key=='buffer_dw':
            title += f' Buffer dw: {value} mins;'
        elif key=='coeff_lim':
            title += f' $\\rho\\geq${value};'
        elif key=='dist_buff':
            title += f' Distance: {value} Re;'
        elif key=='min_ratio':
            title += f' Min Ratio: {value};'

        if i==1:
            title += '\n'

    return title[:-1]

def create_file_name(start, title_info_dict):
    file_name = start

    for key, value in title_info_dict.items():
        if key=='buffer_up':
            file_name += f'_Buffer_up_{value}'
        elif key=='buffer_dw':
            file_name += f'_Buffer_dw_{value}'
        elif key=='coeff_lim':
            file_name += f'_Coeff_lim_{value}'
        elif key=='dist_buff':
            file_name += f'_Dist_buff_{value}'
        elif key=='min_ratio':
            file_name += f'_Min_ratio_{value}'
        elif key=='colouring':
            file_name += f'_colour_{value}'

    return file_name