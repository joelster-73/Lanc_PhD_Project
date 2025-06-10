# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""


# %%
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
import numpy as np
import matplotlib.pyplot as plt

columns = [col for col in shocks_intercepts if '_coeff' in col]

all_coeffs = shocks_intercepts[columns].to_numpy().flatten()
all_coeffs = all_coeffs[~np.isnan(all_coeffs)]
all_coeffs = all_coeffs[(all_coeffs>=0)&(all_coeffs<=1)]

bin_width=0.01
bins = np.arange(start=0, stop=1+bin_width, step=bin_width)

fig, ax = plt.subplots()

ax.hist(all_coeffs, bins=bins, color='b')
ax.set_xlabel('Cross Correlation [0,1)')
ax.set_ylabel('Count')
ax.set_title(f'{len(all_coeffs)} coefficients')

plt.show()

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


colour_dict = {
    'OMNI': 'orange',
    'C1':   'blue',
    'C2':   'cornflowerblue',
    'C3':   'lightskyblue',
    'C4':   'lightblue',
    'THA':  'forestgreen',
    'THB':  'seagreen',
    'THC':  'mediumseagreen',
    'THD':  'lightgreen',
    'THE':  'palegreen',
    'ACE':  'darkviolet',
    'DSC':  'deeppink',
    'GEO':  'teal',
    'IMP8': 'crimson',
    'WIND': 'magenta'
}

# %%




import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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
            else:
                raise Exception(f'{x_axis} not valid choice of "x_axis".')

            detectors.append(detector.upper())
            spacecrafts.append(sc)


            time_diff     = (shock[f'{sc}_time'] - BS_time).total_seconds()
            time_diff_unc = ufloat(time_diff,shock[f'{sc}_time_unc_s']) - ufloat(0,shock['OMNI_time_unc_s'])
            times.append(time_diff)
            times_unc.append(time_diff_unc.s)
            coeffs.append(corr_coeff)

    distances = np.array(distances)
    times = np.array(times)
    distances_unc = np.array(distances_unc)
    times_unc = np.array(times_unc)
    coeffs = np.array(coeffs)
    spacecraft = np.array(spacecrafts)
    detector = np.array(detectors)


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

    elif colouring in ('spacecraft','detector'):
        if colouring == 'spacecraft':
            spacecraft_counts = Counter(spacecraft[closish])
            spacecraft_series = pd.Series(spacecraft[closish])
        else:
            spacecraft_counts = Counter(detector[closish])
            spacecraft_series = pd.Series(detector[closish])

        colours = spacecraft_series.map(colour_dict).fillna('k').to_numpy()
        scatter = ax.scatter(xs, ys, c=colours, s=1)

        legend_elements = [Line2D([0], [0], marker='o', color=colour, label=f'{label}: {spacecraft_counts.get(label, 0)}', markersize=1,
                              linestyle='None')
                       for label, colour in colour_dict.items() if spacecraft_counts.get(label, 0) > 0
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
    elif x_axis=='earth_sun':
        ax.set_xlabel(r'$\rho_{L1}$ [$R_E$]')
    elif x_axis=='x_comp':
        ax.set_xlabel(r'$X_{sc}$ - $X_{BSN}$ [$R_E$]')
        ax.invert_xaxis()
    ax.set_ylabel(r'$t_{SC}$ - $t_{OMNI}$ [mins]')
    ax.set_title(f'{selection.title()} spacecraft: $\\rho\\geq${coeff_lim:.1f}, $R<${max_dist}; N={np.sum(closish):,}')

    plt.show()
    plt.close()

# %%
import numpy as np
from src.analysing.fitting import gaussian, gaussian_fit


def plot_time_histogram(shocks, coeff_lim=0.7, selection='all', show_best_fit=False, show_errors=True):

    # selection = closest, all

    times         = []
    times_unc     = []

    sc_labels = [col.split('_')[0] for col in shocks_intercepts if '_coeff' in col]

    for index, shock in shocks.iterrows():
        detector = shock['spacecraft']

        BS_time     = shock['OMNI_time']
        if pd.isnull(BS_time):
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

    times = np.array(times)
    times_unc = np.array(times_unc)


    fig, ax = plt.subplots()

    times_mins = times/60

    step = 5
    bin_edges = np.arange(np.floor(np.min(times_mins)/step)*step,np.ceil(np.max(times_mins/step)*step),step)
    counts, bins, _ = ax.hist(times_mins,bin_edges,color='k')
    mids = 0.5*(bins[1:]+bins[:-1])

    ax.axvline(x=np.median(times_mins),ls='--',lw=1,c='c',label=f'Median: {np.median(times_mins):.3g} mins')
    ax.axvline(x=0,ls=':',c='w',lw=1)

    if show_best_fit:

        A, mu, sig = gaussian_fit(mids,counts,detailed=True)
        x_values = np.linspace(min(times_mins), max(times_mins), 1000)
        y_values = gaussian(x_values, A.n, mu.n, sig.n)

        ax.plot(x_values,y_values,c='r',label=f'Mean: ${mu:L}$ mins')

    ax.set_xlabel(f'Time differences for {selection} spacecraft [mins]')
    ax.set_ylabel('Counts / 5mins')
    ax.set_title(f'Frequency histogram of {len(times_mins)} measurements')

    ax.legend()

    plt.show()

# %%

plot_time_differences(shocks_intercepts, coeff_lim=0.7, selection='all', x_axis='x_comp', colouring='coeff')

plot_time_histogram(shocks_intercepts, coeff_lim=0.7, selection='closest', show_best_fit=False, show_errors=True)