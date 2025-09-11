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

from ...plotting.utils import save_figure
from ...plotting.config import black, white
from ...plotting.additions import plot_error_region
from ...plotting.formatting import add_legend, add_figure_title, dark_mode_fig
from ...plotting.comparing.parameter import compare_series

from ...processing.speasy.retrieval import retrieve_omni_value
from ...processing.dataframes import add_df_units



# %% Analysis_functions

def analyse_all_events(helsinki_events, **kwargs):

    # Shocks contains the known shock times used in training
    # Dataframe to store shock times in shocks_intercepts

    df_all_events = pd.DataFrame(columns=['event_num','shock_time','shock_time_unc','detector','interceptor','helsinki_delay','correlated_delay','corr_coeff'])

    for eventID, event in helsinki_events.iterrows():

        if len(event['detectors'])<=1:
            continue

        find_shock_times_training(eventID, event, df_all_events, **kwargs)

    add_df_units(df_all_events)
    df_all_events.attrs['units']['detector'] = 'STRING'
    df_all_events.attrs['units']['interceptor'] = 'STRING'
    df_all_events.attrs['units']['event_num'] = 'STRING'

    return df_all_events


def find_shock_times_training(eventID, event, df_all, **kwargs):

    parameter = kwargs.get('parameter','B_mag')

    ###-------------------FIND WHEN SHOCK ARRIVES AT BSN ACCORDING TO OMNI-------------------###

    spacecraft = event['detectors'].copy()

    if 'OMNI' in spacecraft:
        spacecraft.remove('OMNI')

        # Shocks in the event recorded by spacecraft used by OMNI
        omni_time, omni_unc = event[['OMNI_time','OMNI_time_unc_s']]
        if np.isnan(omni_unc):
            omni_unc = 0

        omni_sc = retrieve_omni_value(omni_time, 'OMNI_sc')
        omni_pos = event[[f'OMNI_r_{comp}_GSE' for comp in ('x','y','z')]].to_numpy()

        if omni_sc in spacecraft:
            detect_time, detect_unc = event[[f'{omni_sc}_time',f'{omni_sc}_time_unc_s']]
            if np.isnan(detect_unc):
                detect_unc = 0

            detect_pos = event[[f'{omni_sc}_r_{comp}_GSE' for comp in ('x','y','z')]].to_numpy()

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
                time_A = event[f'{sc_A}_time']
                time_B = event[f'{sc_B}_time']

                if time_A < time_B:
                    detector, interceptor = sc_A, sc_B
                else:
                    detector, interceptor = sc_B, sc_A

            interceptor_time, interceptor_unc = event[[f'{interceptor}_time',f'{interceptor}_time_unc_s']]
            interceptor_pos = event[[f'{interceptor}_r_{comp}_GSE' for comp in ('x','y','z')]].to_numpy()

            detector_time, detector_unc = event[[f'{detector}_time',f'{detector}_time_unc_s']]
            detector_pos = event[[f'{detector}_r_{comp}_GSE' for comp in ('x','y','z')]].to_numpy()

            # Find how long shocks takes to intercept spacecraft
            time_lag = find_propagation_time(detector_time, detector, interceptor, parameter, detector_pos, intercept_pos=interceptor_pos, **kwargs)

            if time_lag is not None:

                corr_delay, corr_coeff = time_lag

                time_diff = (interceptor_time-detector_time).total_seconds()
                time_diff = ufloat(time_diff,interceptor_unc) - ufloat(0,detector_unc)

                # Found suitable lag
                df_all.loc[len(df_all)] = [eventID, detector_time, detector_unc, detector, interceptor, time_diff, corr_delay, corr_coeff]


# %% Compare_correlated_helsinki

def plot_comparison(df, **kwargs):

    coeff_lim  = kwargs.get('coeff_lim',0.8)
    colouring  = kwargs.get('colouring','coeff')

    print(f'Number in total: {np.sum(df.loc[:,"corr_coeff"]>=0)}')
    coeff_mask = df.loc[:,'corr_coeff']>=coeff_lim

    hel_delays = df.loc[coeff_mask,'helsinki_delay'].apply(lambda x: x.nominal_value/60)
    hel_errors = df.loc[coeff_mask,'helsinki_delay'].apply(lambda x: x.std_dev/60)
    cor_delays = df.loc[coeff_mask,'correlated_delay'].apply(lambda x: x.nominal_value/60)
    cor_errors = df.loc[coeff_mask,'correlated_delay'].apply(lambda x: x.std_dev/60)

    detectors    = df.loc[coeff_mask,'detector']
    interceptors = df.loc[coeff_mask,'interceptor']

    hel_delays.attrs = {'units': {'helsinki_delay': 'mins'}}
    cor_delays.attrs = {'units': {'correlated_delay': 'mins'}}

    if kwargs.get('data1_name',None) is None:
        kwargs['data1_name'] = 'Helsinki Delays'

    if kwargs.get('data2_name',None) is None:
        kwargs['data2_name'] = 'Correlated Delays'

    if kwargs.get('brief_title',None) is None:
        event_nums = df.loc[coeff_mask,'event_num']
        kwargs['brief_title'] = f'Comparing times from {len(np.unique(event_nums))} shocks: $\\rho\\geq${coeff_lim:.1g}'

    fig, _ = compare_series(hel_delays, cor_delays, xs_unc=hel_errors, ys_unc=cor_errors, display='scatter_dict', sc_ups=detectors, sc_dws=interceptors, fit_type='straight', as_text=True, return_objs=True, **kwargs)


    file_info_dict = kwargs.get('file_info_dict',{})
    file_info_dict['coeff_lim'] = coeff_lim
    file_info_dict['colouring'] = colouring

    file_name = create_file_name('Comparing_Delay_Times', file_info_dict)
    save_figure(fig, file_name=file_name)
    plt.show()
    plt.close()


# %% Vary_one_param

def train_algorithm_param(df_events, vary='buffer_up', vary_array=None, coeff_lim=0.8, **kwargs):

    if vary_array is None:
        if (vary=='buffer_up'):
            vary_array = range(28,45,1)
        elif (vary=='buffer_dw'):
            vary_array = range(30,39,1)
        elif (vary=='dist_buff'):
            vary_array = np.arange(35,86,5)
        elif (vary=='min_ratio'):
            vary_array = np.arange(0.7,0.96,0.01)
        else:
            raise Exception(f'Not valid parameter for vary: {vary}')

    df_trained_params = pd.DataFrame(columns=[vary,'num_times','fit_R2','fit_slope','fit_intercept'], index=list(range(len(vary_array))))

    for i, param_val in enumerate(vary_array):
        kwargs[vary] = param_val

        df_all_events = analyse_all_events(df_events, **kwargs)

        ###-------------------MINIMUM CROSS-CORR-------------------###

        coeff_mask = df_all_events['corr_coeff']>=coeff_lim
        if np.sum(coeff_mask)==0:
            raise Exception('No times above correlation coefficient limit.')

        xs      = df_all_events.loc[coeff_mask,'helsinki_delay'].apply(lambda x: x.nominal_value/60)
        ys      = df_all_events.loc[coeff_mask,'correlated_delay'].apply(lambda x: x.nominal_value/60)
        ys_unc  = df_all_events.loc[coeff_mask,'correlated_delay'].apply(lambda x: x.std_dev/60)

        fit_dict = fit_function(xs, ys, fit_type='straight', ys_unc=ys_unc)
        slope, intercept, r2 = fit_dict['params']['m'], fit_dict['params']['c'], fit_dict['R2']

        df_trained_params.iloc[i] = [param_val, np.sum(coeff_mask), r2, slope, intercept]

    return df_trained_params

def plot_grid_param_vary(df_events, *independent, **kwargs):

    show_count = kwargs.get('show_count',True)

    if len(independent)==0:
        print('No varying parameters')

    num_params = len(independent)

    if num_params <= 3:
        cols = num_params
        rows = 1
    elif num_params == 4:
        cols = 2
        rows = 2
    else:
        cols = 3
        rows = (num_params + 2) // 3  # for fallback if more than 4

    if show_count:
        fig = plt.figure(figsize=(8 * cols, 6.25 * rows))
        height_ratios = []
        for _ in range(rows):
            height_ratios.extend([3, 1])

        gs = fig.add_gridspec(2 * rows, cols, height_ratios=height_ratios)

        axes = []
        hist_axes = []
        for i in range(num_params):
            r = i // cols
            c = i % cols

            main_row = r * 2 if rows > 1 else 0
            hist_row = main_row + 1

            ax = fig.add_subplot(gs[main_row, c])
            hist_ax = fig.add_subplot(gs[hist_row, c], sharex=ax)
            hist_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

            axes.append(ax)
            hist_axes.append(hist_ax)
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if isinstance(axes, np.ndarray):
            axes = axes.flatten().tolist()
        else:
            axes = [axes]

    fig.tight_layout()

    label_dict = {0: '(a)', 1: '(b)', 2: '(c)', 3: '(d)'}

    for i, param in enumerate(independent):
        df_params = train_algorithm_param(df_events, vary=param, **kwargs)

        ax = axes[i]

        if show_count:
            hist_ax = hist_axes[i]
            axs = [ax,hist_ax]
        else:
            axs = [ax]

        kwargs['show_left_lab']  = (i % cols)==0
        kwargs['show_right_lab'] = (i % cols)==(cols-1)
        kwargs['want_legend']    = (i==0)

        if num_params>1:
            kwargs['title_letter'] = label_dict[i]

        _ = plot_single_param_vary(df_params, param, fig=fig, axs=axs, return_objs=True, **kwargs)


    save_figure(fig)
    plt.show()
    plt.close()



def plot_single_param_vary(df_vary, param, **kwargs):

    coeff_lim      = kwargs.get('coeff_lim',None)
    want_legend    = kwargs.get('want_legend',False)
    show_left_lab  = kwargs.get('show_left_lab',True)
    show_right_lab = kwargs.get('show_right_lab',True)
    title_letter   = kwargs.get('title_letter',None)

    fig = kwargs.get('fig',None)
    axs = kwargs.get('axs',None)
    return_objs = kwargs.get('return_objs',True)

    if param=='buffer_up':
        x_label = r'$\Delta t_\mathrm{data}$ [mins]'
        title = r'Varying Data Time'
    elif param=='buffer_dw':
        x_label = r'$t_\mathrm{near}$ [mins]'
        title = r'Varying Near Time'
    elif param=='dist_buff':
        x_label = r'$d_\mathrm{diff}$ [$\mathrm{R_E}$]'
        title = r'Varying Distance'
    elif param=='min_ratio':
        x_label = r'$B_{\mathrm{thresh}}$'
        title = r'Varying Minimum Compression'
    else:
        x_label = param
        title = f'Varying {param}'

    if title_letter is not None:
        title = title_letter + ' ' + title

    brief_title = kwargs.get('brief_title',title)

    if fig is None or axs is None:
        if 'num_times' in df_vary:
            fig = plt.figure(figsize=(12, 10))
            gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
            ax = fig.add_subplot(gs[0])
            histx_ax = fig.add_subplot(gs[1], sharex=ax)
            plt.setp(histx_ax.get_xticklabels(), visible=False)
            fig.tight_layout()
            axs = [ax,histx_ax]
        else:
            fig, ax = plt.subplots()
            axs = [ax]
    else:
        ax, histx_ax = axs

    vary_values    = df_vary.loc[:,param].to_numpy()
    fit_slopes     = df_vary.loc[:,'fit_slope'].to_numpy()
    fit_R2s        = df_vary.loc[:,'fit_R2'].to_numpy()
    fit_intercepts = df_vary.loc[:,'fit_intercept'].to_numpy()
    counts         = df_vary.loc[:,'num_times'].to_numpy()

    if fit_slopes is None and counts is None and fit_R2s is None:
        raise Exception('No data to plot.')

    ###-------------------FITTED GRADIENT-------------------###
    y_label = ''
    if fit_slopes is not None:
        plot_error_region(ax, vary_values, unp.nominal_values(fit_slopes), unp.std_devs(fit_slopes), c='r', marker='x', label='Gradient')

        y_label += 'Fitted Gradient'
        ax.axhline(y=1, c='grey', ls=':', lw='1', alpha=0.1)

    if fit_R2s is not None:
        ax.plot(vary_values, fit_R2s, c='b', marker='o', label=r'$R^2$')
        if y_label=='':
            y_label = r'Fitted $R^2$'
        else:
            y_label += r' & $R^2$'

    ###-------------------FITTED INTERCEPT-------------------###
    if fit_intercepts is not None:
        ax2 = ax.twinx()
        axs.append(ax2)

        plot_error_region(ax2, vary_values, unp.nominal_values(fit_intercepts), unp.std_devs(fit_intercepts), c='darkcyan', marker='^', label='Intercept')
        ax2.axhline(y=0, c='grey', ls=':', lw='1', alpha=0.1)

        if param=='min_ratio':
            ax2_lims = ax2.get_ylim()
            ax2.set_ylim(ax2_lims[0],2.5*ax2_lims[1])

        if show_right_lab:
            ax2.set_ylabel('Fitted Intercept [mins]')
        add_legend(fig, ax2, loc='upper right', legend_on=want_legend)

    ###-------------------FITTED COUNTS-------------------###
    if counts is not None:
        label = 'Count'
        if coeff_lim is not None:
            label = f'$N$, $\\rho\\geq{coeff_lim}$'
        histx_ax.plot(vary_values, counts, c='k', marker='|', label=label)

        if show_left_lab:
            histx_ax.set_ylabel('Count')

        loc = 'upper left'
        if param=='min_ratio':
            loc = 'upper right'
        add_legend(fig, histx_ax, loc=loc, legend_on=want_legend)

    ###---------------LABELLING AND FINISHING TOUCHES---------------###
    ax.set_xlabel(x_label, c=black)
    if show_left_lab:
        ax.set_ylabel(y_label, c=black)

    if brief_title=='':
        brief_title = 'Algorithm\'s Performance by Varying Parameter'

    add_legend(fig, ax, loc='upper left', legend_on=want_legend)
    add_figure_title(fig, black, brief_title, ax=ax)
    dark_mode_fig(fig,black,white)
    plt.tight_layout();

    if return_objs:
        return fig, axs

    file_name = create_file_name(f'Vary_{param}')
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