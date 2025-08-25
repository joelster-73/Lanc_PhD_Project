# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from uncertainties import ufloat
from uncertainties import unumpy as unp
import itertools as it

from .intercepts import find_propagation_time

from ...analysing.fitting import fit_function

from ...processing.speasy.config import colour_dict
from ...processing.speasy.retrieval import retrieve_omni_value

from ...plotting.utils import save_figure
from ...plotting.additions import plot_error_region
from ...plotting.formatting import add_legend


# %% Analysis_functions

def analyse_all_events(helsinki_events, **kwargs):

    # Shocks contains the known shock times used in training
    # Dataframe to store shock times in shocks_intercepts


    df_all_events = pd.DataFrame(columns=['event_num','shock_time','shock_time_unc','detector','interceptor','helsinki_delay','correlated_delay','corr_coeff'])

    for eventID_str, event in helsinki_events.groupby(by='eventNum'):

        eventID = int(eventID_str)
        if len(event)<=1:
            continue

        find_shock_times_training(eventID, event, df_all_events, **kwargs)

    return df_all_events


def find_shock_times_training(eventID, event, df_all, **kwargs):

    parameter = kwargs.get('parameter','B_mag')

    ###-------------------INITIAL CHECKS-------------------###

    # Info for all shocks in this event
    times       = event.index.tolist()
    uncs        = event['time_unc'].tolist()
    spacecraft   = event['spacecraft'].tolist()
    spacecraft_dict = dict(zip(spacecraft,list(zip(times,uncs))))

    ###-------------------FIND WHEN SHOCK ARRIVES AT BSN ACCORDING TO OMNI-------------------###

    # Shocks in the event recorded by spacecraft used by OMNI
    if 'OMNI' in spacecraft:
        spacecraft.remove('OMNI')
        omni_event = event[event['spacecraft']=='OMNI']

        omni_time, omni_unc = spacecraft_dict['OMNI']
        omni_sc = retrieve_omni_value(omni_time, 'OMNI_sc')
        omni_pos = omni_event.iloc[0][['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()

        if omni_sc in spacecraft:

            detect_event = event[event['spacecraft']==omni_sc]

            detect_time, detect_unc = spacecraft_dict[omni_sc]
            detect_pos = detect_event.iloc[0][['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()

            # Find time lag
            time_lag = find_propagation_time(detect_time, omni_sc, 'OMNI', parameter, detect_pos, intercept_pos=omni_pos, **kwargs)
            if time_lag is not None:

                delay, coeff = time_lag

                time_diff = (omni_time-detect_time).total_seconds()
                time_diff = ufloat(time_diff,omni_unc) - ufloat(0,detect_unc)

                # Found suitable lag
                df_all.loc[len(df_all)] = [eventID, detect_time, detect_unc, omni_sc, 'OMNI', time_diff, delay, coeff]

    # Need time to check against
    if len(spacecraft)>1:

        ###-------------------FIND WHEN SHOCKS INTERCEPT DOWNSTREAM SPACECRAFT-------------------###

        # These are the monitors we will be investigating
        sw_detectors = ('ACE','WIND','DSC')

        for (sc_A, sc_B) in it.combinations(spacecraft, 2):

            if not (sc_A in sw_detectors or sc_B in sw_detectors):
                continue
            elif sc_B not in sw_detectors:
                detector, interceptor = sc_A, sc_B
            elif sc_A not in sw_detectors:
                detector, interceptor = sc_B, sc_A
            else:
                time_A = spacecraft_dict[sc_A][0]
                time_B = spacecraft_dict[sc_B][0]

                if time_A < time_B:
                    detector, interceptor = sc_A, sc_B
                else:
                    detector, interceptor = sc_B, sc_A

            interceptor_event = event[event['spacecraft']==interceptor]
            interceptor_time, interceptor_unc = spacecraft_dict[interceptor]
            interceptor_pos = interceptor_event.iloc[0,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()

            detector_event = event[event['spacecraft']==detector]
            detector_time, detector_unc = spacecraft_dict[detector]
            detector_pos = detector_event.iloc[0,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()

            # Find how long shocks takes to intercept spacecraft
            time_lag = find_propagation_time(detector_time, detector, interceptor, parameter, detector_pos, intercept_pos=interceptor_pos, **kwargs)

            if time_lag is not None:

                corr_delay, corr_coeff = time_lag

                time_diff = (interceptor_time-detector_time).total_seconds()
                time_diff = ufloat(time_diff,interceptor_unc) - ufloat(0,detector_unc)

                # Found suitable lag
                df_all.loc[len(df_all)] = [eventID, detector_time, detector_unc, detector, interceptor, time_diff, corr_delay, corr_coeff]


# %% Compare_correlated_helsinki

def plot_comparison(database, correlated, coeffs, sc_ups, sc_dws, **kwargs):

    coeff_lim       = kwargs.get('coeff_lim',0.7)
    colouring       = kwargs.get('colouring','coeff')
    title_info_dict = kwargs.get('title_info_dict',{})
    simple_title    = kwargs.get('simple_title',True)
    event_nums      = kwargs.get('event_nums',None)

    title_info_dict['coeff_lim'] = coeff_lim
    title_info_dict['colouring'] = colouring

    marker_dict = {'WIND': 'x', 'ACE': '+', 'DSC': '^', 'Cluster': 'o'}

    coeff_mask = coeffs>=coeff_lim

    xs      = unp.nominal_values(database[coeff_mask])
    xs_unc  = unp.std_devs(database[coeff_mask])
    ys      = unp.nominal_values(correlated[coeff_mask])
    ys_unc  = unp.std_devs(correlated[coeff_mask])
    coeffs  = coeffs[coeff_mask]

    sc_ups  = sc_ups[coeff_mask]
    sc_dws  = sc_dws[coeff_mask]

    fit_dict = fit_function(xs,ys,fit_type='straight',ys_unc=ys_unc)
    slope, intercept, r2 = fit_dict['params']['m'], fit_dict['params']['c'], fit_dict['R2']

    fig, ax = plt.subplots()

    ax.errorbar(xs, ys, xerr=xs_unc, yerr=ys_unc, fmt='.', ms=0, ecolor='k', capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

    if colouring=='coeff':
        scatter = ax.scatter(xs, ys, c=coeffs, s=20, marker='x', cmap='plasma_r', vmin=coeff_lim, vmax=1)
        ax.scatter([], [], c='k', s=20, marker='x', label=f'{len(xs)}')

        cbar = plt.colorbar(scatter)
        cbar.set_label('Correlation Coeff')
    elif colouring=='sc':

        for (sc_up, sc_dw) in it.permutations(('WIND','ACE','DSC','Cluster','OMNI'), 2):

            if sc_dw=='Cluster':
                sc_mask = (sc_ups==sc_up)&(np.isin(sc_dws, ('C1','C3','C4')))
                sc_c    = colour_dict['C1']
            elif sc_up=='Cluster':
                sc_mask = (np.isin(sc_ups, ('C1','C3','C4')))&(sc_dws==sc_dw)
                sc_c    = colour_dict[sc_dw]
            else:
                sc_mask = (sc_ups==sc_up)&(sc_dws==sc_dw)
                sc_c    = colour_dict[sc_dw]
            count = np.sum(sc_mask)
            if count==0:
                continue
            ax.scatter(xs[sc_mask], ys[sc_mask], c=sc_c, s=20, marker=marker_dict[sc_up], label=f'{sc_up} | {sc_dw}: {count}')


    ax.axline(slope=1,xy1=[0,0],c='grey',ls=':')
    ax.axline([0,intercept.n],slope=slope.n,c='k',ls='--',lw=1.25)

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

    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal')

    if simple_title:
        if event_nums is not None:
            num_shocks = len(np.unique(event_nums))
            title = f'Comparing {np.sum(coeff_mask)} times from {num_shocks} shocks'
        else:
            title = 'Comparing Correlated Times with Helsinki Database'
    else:
        title = create_title('Comparing Delay Times', title_info_dict)
    ax.set_title(title)
    plt.tight_layout()

    file_name = create_file_name('Comparing_Delay_Times', title_info_dict)
    save_figure(fig, file_name=file_name)
    plt.show()
    plt.close()


# %% Vary_one_param

def train_algorithm_param(df_events, vary='buffer_up', vary_array=None, coeff_lim=0.7, **kwargs):

    if vary_array is None:
        if (vary=='buffer_up'):
            vary_array = range(25,41,1)
        elif (vary=='buffer_dw'):
            vary_array = range(20,51,5)
        elif (vary=='dist_buff'):
            vary_array = np.arange(10,101,5)
        elif (vary=='min_ratio'):
            vary_array = np.arange(0.5,0.91,0.01)
        else:
            raise Exception(f'Not valid parameter for vary: {vary}')

    df_trained_params = pd.DataFrame(columns=[vary,'num_shocks','fit_R2','fit_slope','fit_intercept'], index=np.arange(len(vary_array)))

    key_map = {
        'buffer_up': 'buffer_up',
        'buffer_dw': 'buffer_dw',
        'dist_buff': 'distance_buff',
        'min_ratio': 'min_ratio_change'
    }

    param_key = key_map[vary]

    for i, ind in enumerate(vary_array):
        kwargs[param_key] = ind

        df_all_events = analyse_all_events(df_events, **kwargs)

        ###-------------------MINIMUM CROSS-CORR-------------------###

        coeff_mask = df_all_events.loc[:,'corr_coeff']>=coeff_lim

        xs      = df_all_events[coeff_mask,'helsinki_delay'] # get nominal value
        ys      = df_all_events[coeff_mask,'correlated_delay'] # get nominal value
        ys_unc  = df_all_events[coeff_mask,'correlated_delay'] # get standard deviation

        fit_dict = fit_function(xs,ys,fit_type='straight',ys_unc=ys_unc)
        slope, intercept, r2 = fit_dict['params']['m']. fit_dict['params']['c'], fit_dict['R2']

        df_trained_params.iloc[i] = [ind, len(xs),r2,slope,intercept]

    return df_trained_params

def plot_single_param_vary(independent, **kwargs):

    slopes_fit      = kwargs.get('slopes_fit', None)
    slopes_int      = kwargs.get('slopes_int', None)
    counts          = kwargs.get('counts', None)
    slopes_R2       = kwargs.get('slopes_R2', None)
    ind_var         = kwargs.get('ind_var', 'Independent Variable')
    title_info_dict = kwargs.get('title_info_dict',{})
    simple_title    = kwargs.get('simple_title',True)
    coeff_lim       = kwargs.get('coeff_lim',None)

    if slopes_fit is None and counts is None and slopes_R2 is None:
        raise Exception('No data to plot.')

    if ind_var=='buffer_up':
        x_label = r'Data Time ($\Delta t_\mathrm{data}$) [mins]'
    elif ind_var=='buffer_dw':
        x_label = r'Near Time ($t_\mathrm{near}$) [mins]'
    elif ind_var=='dist_buff':
        x_label = r'Distances ($d_\mathrm{diff}$) [$\mathrm{R_E}$]'
    elif ind_var=='min_ratio':
        x_label = r'$B_{\mathrm{ratio,2}}\geq x\cdot B_{\mathrm{ratio,1}}$'
    else:
        x_label = ind_var

    if ind_var in title_info_dict:
        del title_info_dict[ind_var]

    if counts is not None:
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax = fig.add_subplot(gs[0])
        histx_ax = fig.add_subplot(gs[1], sharex=ax)
        plt.setp(histx_ax.get_xticklabels(), visible=False)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots()

    y_label = ''
    if slopes_fit is not None:
        plot_error_region(ax, independent, unp.nominal_values(slopes_fit), unp.std_devs(slopes_fit), c='r', marker='x', label='Gradient')

        y_label += 'Fitted Gradient'
        ax.axhline(y=1, c='grey', ls=':', lw='1', alpha=0.1)

    if slopes_R2 is not None:
        ax.plot(independent, slopes_R2, c='b', marker='o', label=r'$R^2$')
        if y_label=='':
            y_label = r'Fitted $R^2$'
        else:
            y_label += r' & $R^2$'

    if slopes_int is not None:
        ax2 = ax.twinx()
        plot_error_region(ax2, independent, unp.nominal_values(slopes_int), unp.std_devs(slopes_int), c='g', marker='^', label='Intercept')
        ax2.axhline(y=0, c='grey', ls=':', lw='1', alpha=0.1)

        if ind_var=='min_ratio':
            ax2_lims = ax2.get_ylim()
            ax2.set_ylim(ax2_lims[0],2.5*ax2_lims[1])

        ax2.set_ylabel('Fitted Intercept')
        add_legend(fig, ax2, loc='upper right')

    if counts is not None:
        label = 'Count'
        if coeff_lim is not None:
            label = f'$N$, $\\rho\\geq{coeff_lim}$'
        histx_ax.plot(independent, counts, c='k', marker='|', label=label)

        #hist_ticks = histx_ax.get_yticks()
        #histx_ax.set_yticks(hist_ticks)
        #new_y_labels = [f'{int(tick)} ({(tick / counts[0]) * 100:.0f}%)' if tick >= 0 else '' for tick in hist_ticks]
        #histx_ax.set_yticklabels(new_y_labels)
        histx_ax.set_ylabel('Number of Shocks')
        loc = 'upper left'
        if ind_var=='min_ratio':
            loc = 'upper right'

        add_legend(fig, histx_ax, loc=loc)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    add_legend(fig, ax, loc='upper left')

    if simple_title:
        title = 'Algorithm\'s Performance by Varying Parameter'
    else:
        title = create_title(f'Varying {ind_var}', title_info_dict)
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