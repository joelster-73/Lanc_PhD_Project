# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""


# %%
import numpy as np

def find_closest(shocks):

    if 'closest' not in shocks.columns:
        shocks['closest'] = None

    sc_labels = [col.split('_')[0] for col in shocks_intercepts if '_coeff' in col]

    for index, shock in shocks.iterrows():

        omni_pos = np.array([shock[f'OMNI_r_{comp}_GSE'] for comp in ('x','y','z')])
        for comp in omni_pos:
            if np.isnan(comp):
                continue

        sc_distances = {}
        for sc in sc_labels:
            if sc == 'OMNI' or sc==shock['spacecraft']:
                continue

            sc_pos = np.array([shock[f'{sc}_r_{comp}_GSE'] for comp in ('x','y','z')])
            for comp in sc_pos:
                if np.isnan(comp):
                    continue
            sc_distances[sc] = np.linalg.norm(sc_pos - omni_pos)

        shocks.at[index,'closest'] = min(sc_distances, key=sc_distances.get)

# %% Importing
from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data


shocks_intercepts = import_processed_data(PROC_SHOCKS_DIR)
find_closest(shocks_intercepts)


# %%
from uncertainties import unumpy as unp
from uncertainties import ufloat

def vec_mag(vec):

    try:
        return np.linalg.norm(vec)
    except:
        return unp.sqrt(np.sum(vec**2))
    return np.nan


def get_position_u(shock, sc):

    x = shock[f'{sc}_r_x_GSE']
    y = shock[f'{sc}_r_y_GSE']
    z = shock[f'{sc}_r_z_GSE']
    for comp in (x,y,z):
        if np.isnan(comp):
            return None

    x_u = shock[f'{sc}_r_x_GSE_unc']
    y_u = shock[f'{sc}_r_x_GSE_unc']
    z_u = shock[f'{sc}_r_x_GSE_unc']
    for unc in (x_u,y_u,z_u):
        if np.isnan(unc):
            unc = 0
    return unp.uarray([x,y,z],[x_u,y_u,z_u])



# %%
from src.processing.speasy.config import colour_dict

from src.analysing.fitting import gaussian, gaussian_fit

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

database_colour_dict = {'CFA': 'b', 'Donki': 'r'}

def plot_time_differences(shocks, coeff_lim=0.7, selection='all', x_axis='dist', colouring='spacecraft', show_best_fit=True, show_errors=True, max_dist=100):

    # selection = closest, all
    # x_axis    = dist, x_comp, earth_sun
    # colouring = coeff, spacecraft, detector, none

    distances     = []
    distances_unc = []
    times         = []
    times_unc     = []
    coeffs        = []
    spacecrafts   = []
    detectors     = []
    databases     = []

    sc_labels = [col.split('_')[0] for col in shocks_intercepts if '_coeff' in col]

    for index, shock in shocks.iterrows():
        detector = shock['spacecraft']

        BS_time     = shock['OMNI_time']
        if pd.isnull(BS_time):
            continue


        BS_pos = get_position_u(shock,'OMNI')
        if BS_pos is None and x_axis!='earth_sun':
            continue

        for sc in sc_labels:
            if (selection=='closest' and sc!=shock['closest']) or sc in ('OMNI',detector):
                continue

            corr_coeff = shock[f'{sc}_coeff']
            if isinstance(corr_coeff, (pd.Series, pd.DataFrame)) and len(corr_coeff) > 1:
                corr_coeff = corr_coeff.iloc[0]  # Get the first value
            else:
                corr_coeff = corr_coeff

            if np.isnan(corr_coeff) or corr_coeff<coeff_lim or corr_coeff>1:
                #1.1 indicates exact matches
                continue

            if x_axis=='earth_sun':
                L1_pos = get_position_u(shock,detector)
                if L1_pos is None:
                    continue
                L1_rho = unp.sqrt(L1_pos[1]**2+L1_pos[2]**2)
                distances.append(L1_rho.n)
                distances_unc.append(L1_rho.s)

            elif x_axis=='x_comp':
                sc_x = ufloat(shock[f'{sc}_r_x_GSE'],shock[f'{sc}_r_x_GSE_unc'])
                if np.isnan(sc_x.n):
                    continue
                bs_x = ufloat(shock['OMNI_r_x_GSE'] ,shock['OMNI_r_x_GSE_unc'])
                distances.append((sc_x-bs_x).n)
                distances_unc.append((sc_x-bs_x).s)

            elif x_axis=='dist':
                sc_pos = get_position_u(shock,sc)
                if sc_pos is None:
                    continue
                dist_diff = vec_mag(sc_pos-BS_pos)
                distances.append(unp.nominal_values(dist_diff))
                distances_unc.append(unp.std_devs(dist_diff))

            elif x_axis=='signed_dist':
                sc_pos = get_position_u(shock,sc)
                if sc_pos is None:
                    continue
                sign = np.sign(shock[f'{sc}_r_x_GSE'])
                dist_diff = vec_mag(sc_pos-BS_pos)
                distances.append(sign*unp.nominal_values(dist_diff))
                distances_unc.append(unp.std_devs(dist_diff))
            else:
                raise Exception(f'{x_axis} not valid choice of "x_axis".')

            detectors.append(detector.upper())
            spacecrafts.append(sc)
            databases.append('CFA' if shock['source']=='C' else 'Donki')


            time_diff     = (shock[f'{sc}_time'] - BS_time).total_seconds()
            time_diff_unc = ufloat(time_diff,shock[f'{sc}_time_unc_s']) - ufloat(0,shock['OMNI_time_unc_s'])
            times.append(time_diff)
            times_unc.append(time_diff_unc.s)
            coeffs.append(corr_coeff)


    distances     = np.array(distances)
    times         = np.array(times)
    distances_unc = np.array(distances_unc)
    times_unc     = np.array(times_unc)
    coeffs        = np.array(coeffs)
    spacecrafts   = np.array(spacecrafts)
    detectors     = np.array(detectors)
    databases     = np.array(databases)


    closish = abs(distances)<max_dist

    xs = distances[closish]
    ys = times[closish]/60

    xs_unc = distances_unc[closish]
    ys_unc = times_unc[closish]/60

    fig, ax = plt.subplots(figsize=(8,4),dpi=300)
    if show_errors:
        error_colour = 'k'

    if show_errors:
        error_colour = 'k' if colouring in ('coeff','spacecraft','detector') else 'r'
        ax.errorbar(xs, ys, xerr=xs_unc, yerr=ys_unc, fmt='.', ms=0, ecolor=error_colour, capsize=0.5, capthick=0.2, lw=0.2, zorder=1)


    ax.axhline(0,c='grey',ls=':')

    if colouring == 'coeff':

        scatter = ax.scatter(xs, ys, c=coeffs[closish], cmap='coolwarm', vmin=coeff_lim, vmax=1, s=1)

        cbar = plt.colorbar(scatter)
        cbar.set_label('correlation coefficient')

    elif colouring in ('spacecraft','detector','database'):
        plot_colour_dict = colour_dict
        if colouring == 'spacecraft':
            spacecraft_counts = Counter(spacecrafts[closish])
            colours = pd.Series(spacecrafts[closish]).map(colour_dict).fillna('k').to_numpy()
        elif colouring == 'detector':
            spacecraft_counts = Counter(detectors[closish])
            colours = pd.Series(detectors[closish]).map(colour_dict).fillna('k').to_numpy()
        elif colouring == 'database':
            spacecraft_counts = Counter(databases[closish])
            colours = pd.Series(databases[closish]).map(database_colour_dict).fillna('k').to_numpy()
            plot_colour_dict = database_colour_dict

        scatter = ax.scatter(xs, ys, c=colours, s=1)

        legend_elements = [Line2D([0], [0], marker='o', color=colour, label=f'{label}: {spacecraft_counts.get(label, 0)}', markersize=1,
                              linestyle='None')
                       for label, colour in plot_colour_dict.items() if spacecraft_counts.get(label, 0) > 0
        ]
        plt.legend(handles=legend_elements, fontsize=6, loc='upper left', bbox_to_anchor=(1.01, 1.0))

    else:
        ax.scatter(distances[closish], times[closish]/60, c='k', s=1)

    from sklearn.metrics import r2_score

    if show_best_fit:
        slope, intercept = np.polyfit(xs, ys, 1, w=1/ys_unc)
        y_pred = slope * xs + intercept
        ax.plot(xs,y_pred,c='k',lw=1,ls='--')

        if intercept<0:
            sign = '-'
        else:
            sign = '+'
        r2 = r2_score(ys, y_pred)
        middle = (np.max(xs)+np.min(xs))/2
        ax.text(middle,np.max(ys),f'$\\Delta t$ = {slope:.2f}$\\Delta r$ {sign} {abs(intercept):.2f} mins, $R^2$={r2:.3f}',horizontalalignment='center')


    if x_axis=='dist':
        ax.set_xlabel(r'|$r_{SC}$ - $r_{BSN}$| [$R_E$]')
    elif x_axis=='signed_dist':
        ax.set_xlabel(r'sgn(x) $\cdot$ |$r_{SC}$ - $r_{BSN}$| [$R_E$]')
        ax.invert_xaxis()
    elif x_axis=='earth_sun':
        ax.set_xlabel(r'$\rho_{L1}$ [$R_E$]')
    elif x_axis=='x_comp':
        ax.set_xlabel(r'$X_{sc}$ - $X_{BSN}$ [$R_E$]')
        ax.invert_xaxis()
    ax.set_ylabel(r'$t_{SC}$ - $t_{OMNI}$ [mins]')
    ax.set_title(f'{selection.title()} spacecraft: $\\rho\\geq${coeff_lim:.1f}, $R<${max_dist}; N={np.sum(closish):,}')

    plt.show()
    plt.close()


def plot_time_histogram(shocks, coeff_lim=0.7, selection='all', show_best_fit=False, show_errors=True, colouring='none'):

    # selection = closest, all
    # colouring = spacecraft, detector, none

    times         = []
    times_unc     = []
    spacecrafts   = []
    detectors     = []

    sc_labels = [col.split('_')[0] for col in shocks_intercepts if '_coeff' in col]

    for index, shock in shocks.iterrows():
        detector = shock['spacecraft']

        BS_time     = shock['OMNI_time']
        if pd.isnull(BS_time):
            continue

        BS_coeff = shock['OMNI_coeff']
        if np.isnan(BS_coeff) or BS_coeff<coeff_lim or BS_coeff>1:
            #1.1 indicates exact matches
            continue

        for sc in sc_labels:
            if (selection=='closest' and sc!=shock['closest']) or sc in ('OMNI',detector):
                continue

            corr_coeff = shock[f'{sc}_coeff']
            if isinstance(corr_coeff, (pd.Series, pd.DataFrame)) and len(corr_coeff) > 1:
                corr_coeff = corr_coeff.iloc[0]  # Get the first value
            else:
                corr_coeff = corr_coeff

            if np.isnan(corr_coeff) or corr_coeff<coeff_lim or corr_coeff>1:
                #1.1 indicates exact matches
                continue

            sc_time = shock[f'{sc}_time']
            if pd.isnull(sc_time):
                continue

            time_diff     = (shock[f'{sc}_time'] - BS_time).total_seconds()
            time_diff_unc = ufloat(time_diff,shock[f'{sc}_time_unc_s']) - ufloat(0,shock['OMNI_time_unc_s'])
            times.append(time_diff)
            times_unc.append(time_diff_unc.s)

            spacecrafts.append(sc)
            detectors.append(detector.upper())

    times       = np.array(times)/60
    times_unc   = np.array(times_unc)
    spacecrafts = np.array(spacecrafts)
    detectors   = np.array(detectors)

    fig, ax = plt.subplots()

    step = 5
    bin_edges = np.arange(np.floor(np.min(times)/step)*step,np.ceil(np.max(times/step)*step),step)

    counts, bins = np.histogram(times, bin_edges)
    mids = 0.5*(bins[1:]+bins[:-1])

    if colouring in ('spacecraft','detector'):
        if colouring=='spacecraft':
            sc_array = spacecrafts
        elif colouring=='detector':
            sc_array = detectors
        else:
            raise Exception(f'{colouring} not valid choice for "colouring".')

        grouped_counts = np.zeros((len(sc_labels), len(bin_edges) - 1))

        for i, sc in enumerate(sc_labels):
            grouped_counts[i], _ = np.histogram(times[sc_array == sc], bins=bin_edges)
        bottom = np.zeros(len(bin_edges) - 1)

        for i, sc in enumerate(sc_labels):
            plt.bar(mids, grouped_counts[i], width=np.diff(bin_edges), bottom=bottom, color=colour_dict[sc], label=sc)
            bottom += grouped_counts[i]
    else:
        ax.hist(times, bin_edges, color='k')



    ax.axvline(x=np.median(times),ls='--',lw=1,c='c',label=f'Median: {np.median(times):.3g} mins')
    ax.axvline(x=0,ls=':',c='w',lw=1)

    if show_best_fit:

        A, mu, sig = gaussian_fit(mids,counts,detailed=True)
        x_values = np.linspace(min(times), max(times), 1000)
        y_values = gaussian(x_values, A.n, mu.n, sig.n)

        ax.plot(x_values,y_values,c='r',label=f'Mean: ${mu:L}$ mins')

    ax.set_xlabel(f'Time differences for {selection} spacecraft [mins]')
    ax.set_ylabel('Counts / 5mins')
    ax.set_title(f'Frequency histogram of {len(times)} measurements')

    #ax.legend()

    plt.show()

# %%


plot_time_differences(shocks_intercepts, coeff_lim=0.7, selection='all', x_axis='x_comp', colouring='spacecraft')
plot_time_differences(shocks_intercepts, coeff_lim=0.7, selection='closest', x_axis='x_comp', colouring='coeff')

plot_time_differences(shocks_intercepts, coeff_lim=0.7, selection='all', x_axis='x_comp', colouring='database')
plot_time_differences(shocks_intercepts, coeff_lim=0.7, selection='closest', x_axis='x_comp', colouring='database')


plot_time_differences(shocks_intercepts, coeff_lim=0.7, selection='all', x_axis='signed_dist', colouring='spacecraft')
plot_time_differences(shocks_intercepts, coeff_lim=0.7, selection='closest', x_axis='signed_dist', colouring='coeff')


plot_time_histogram(shocks_intercepts, coeff_lim=0.7, selection='all', show_best_fit=False, show_errors=True, colouring='none')
plot_time_histogram(shocks_intercepts, coeff_lim=0.7, selection='closest', show_best_fit=False, show_errors=True, colouring='none')


# %%
from src.plotting.shocks import plot_shock_times, plot_shock_positions
from src.config import PROC_CFA_DIR
from src.processing.reading import import_processed_data

cfa_shocks = import_processed_data(PROC_CFA_DIR)

from src.processing.cfa.donki import combine_cfa_donki

shocks = combine_cfa_donki(cfa_shocks)
# %%
from src.analysing.shocks.intercepts import find_all_shocks
from datetime import timedelta

coeff_lim = 0.7
sc_labels = [col.split('_')[0] for col in shocks_intercepts if '_coeff' in col]

indices = []
for index, shock in shocks_intercepts.iterrows():
    if index.year < 2009:
        continue
    detector = shock['spacecraft']

    BS_time     = shock['OMNI_time']
    if pd.isnull(BS_time):
        continue
    BS_coeff = shock['OMNI_coeff']
    if np.isnan(BS_coeff) or BS_coeff<coeff_lim or BS_coeff>1:
        #1.1 indicates exact matches
        continue
    print_shock = False
    for sc in sc_labels:
        if sc in ('OMNI',detector):
            continue

        corr_coeff = shock[f'{sc}_coeff']
        if isinstance(corr_coeff, (pd.Series, pd.DataFrame)) and len(corr_coeff) > 1:
            corr_coeff = corr_coeff.iloc[0]  # Get the first value
        else:
            corr_coeff = corr_coeff

        if np.isnan(corr_coeff) or corr_coeff<coeff_lim or corr_coeff>1:
            #1.1 indicates exact matches
            continue

        sc_time = shock[f'{sc}_time']
        if pd.isnull(sc_time):
            continue
        time_diff     = (shock[f'{sc}_time'] - BS_time).total_seconds()
        if time_diff>=(45*60):
            print_shock = True

    if print_shock:
        indices.append(index)
        #plot_shock_times(shock, 'B_mag', time_window=30)

        test_shock = find_all_shocks(shocks, 'field', time=index-timedelta(minutes=5))
        plot_shock_times(test_shock, 'B_mag', time_window=30)
        #plot_shock_positions(shock, 'B_mag')


# %%

from datetime import datetime

#chosen_time = None
#chosen_time = datetime(1996,2,6)
chosen_time = datetime(2007,3,18)

# test_shock = find_all_shocks(shocks, 'B_mag', time=chosen_time)
# plot_shock_times(test_shock, 'B_mag', time_window=30)
# plot_shock_positions(test_shock, 'B_mag')
