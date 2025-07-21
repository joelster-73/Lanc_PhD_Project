# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 15:27:14 2025

@author: richarj2
"""

from src.processing.reading import import_processed_data
from src.config import PROC_INTERCEPTS_DIR

shocks_with_intercepts = import_processed_data(PROC_INTERCEPTS_DIR)

# %% Scatter_and_hist
from src.plotting.shocks import plot_time_differences

plot_time_differences(shocks_with_intercepts, selection='all', x_axis='x_comp', colouring='spacecraft', histograms=True)
plot_time_differences(shocks_with_intercepts, selection='earth', x_axis='x_comp', colouring='spacecraft', histograms=True)

# %% Comparing_with_OMNI_Timeshift

from src.processing.speasy.retrieval import retrieve_omni_value
from src.processing.speasy.config import speasy_variables
import pandas as pd
import numpy as np
from uncertainties import ufloat, unumpy as unp

shock_time_lag = []
omni_time_lag  = []
detectors      = []


for index, shock in shocks_with_intercepts.iterrows():
    omni_time = shock['OMNI_time']
    if pd.isnull(omni_time):
        continue

    try:
        detector = retrieve_omni_value(speasy_variables,omni_time,'OMNI_sc')
    except:
        detector = shock['detectors'].split(',')[0]
    if detector is None or detector=='BAD DATA':
        continue

    sc_time = shock[f'{detector}_time']
    if pd.isnull(sc_time):
        continue
    shock_time_diff = (omni_time - sc_time).total_seconds()/60
    the_timeshift = retrieve_omni_value(speasy_variables, omni_time)/60
    if the_timeshift is None:
        continue
    shock_time_unc = (ufloat(shock_time_diff,shock[f'{detector}_time_unc_s']) - ufloat(0,shock['OMNI_time_unc_s'])).s/60

    shock_time_lag.append(ufloat(shock_time_diff,shock_time_unc))
    omni_time_lag.append(the_timeshift)
    detectors.append(detector)


shock_time_lag = np.array(shock_time_lag)
omni_time_lag  = np.array(omni_time_lag)
detectors      = np.array(detectors)

# %% Comapring_with_OMNI_Timeshift

import matplotlib.pyplot as plt
from src.processing.speasy.config import colour_dict
from src.plotting.config import save_fig
from src.analysing.fitting import straight_best_fit

fig, ax = plt.subplots()

colours = pd.Series(detectors).map(colour_dict).fillna('k').to_numpy()

xs = omni_time_lag
ys = unp.nominal_values(shock_time_lag)
ys_unc = unp.std_devs(shock_time_lag)

ax.errorbar(omni_time_lag, ys, yerr=ys_unc, fmt='.', ms=0, ecolor='k', capsize=0.5, capthick=0.2, lw=0.2, zorder=1)

ax.scatter(xs, ys, s=2, c=colours)
ax.axline([0,0],slope=1,c='k',ls=':')

slope, intercept, r2 = straight_best_fit(xs,ys,ys_unc,detailed=True)

ax.axline([0,intercept.n],slope=slope.n,c='r',ls='--',lw=1)
if intercept.n<0:
    sign = '-'
else:
    sign = '+'
middle = (np.max(xs)+np.min(xs))/2
location = np.max(np.abs(ys))
ax.text(middle,location,f'$\\Delta t$ = (${slope:L}$)$\\Delta r$ {sign} (${abs(intercept):L}$) mins\n$R^2$={r2:.3f}',
        ha='center',va='center')

ax.set_xlabel('OMNI timeshift [mins]')
ax.set_ylabel('Shock timelag [mins]')
ax.set_title('Comparing Shock Timelag against OMNI Timeshift')

plt.show()


