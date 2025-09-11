# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""

from src.config import CROSSINGS_DIR
from src.processing.reading import import_processed_data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



crossings = import_processed_data(CROSSINGS_DIR)
cross_labels = crossings.attrs['crossings']

msh_times = crossings.loc[crossings['loc_num']==10].copy()
msh_times.loc[:,'end_time'] = msh_times.index + pd.to_timedelta(msh_times.loc[:,'region_duration'], unit='s')


# %% Times
duration_hours = msh_times.loc[:,'region_duration']/3600

#import matplotlib.pyplot as plt
#import numpy as np

fig, ax = plt.subplots()

ax.hist(duration_hours, bins=np.arange(int(np.max(duration_hours)+1)), label=f'{np.sum(duration_hours):,} hours')
ax.legend()

plt.show()
plt.close()

# %% Setup

from src.processing.speasy.retrieval import retrieve_data, retrieve_datum
from src.analysing.calculations import calc_mean_error


sources = ('C1','OMNI')
params = {'C1': ('R_GSE','B_mag'), 'OMNI': ('B_mag','MA')}
params = {'C1': ('N_tot','V_GSE'), 'OMNI': ('N_tot','V_GSE')}
params = {'C1': ('B_GSE','V_mag','E_mag','E_GSE'), 'OMNI': ('B_GSE','V_mag','E_mag','E_GSE')}
params = {'C1': ('S_mag','S_GSE'), 'OMNI': ('S_mag','S_GSE')}

# incorporate sigma in B for OMNI?
# think about calculation of beta for cluster

for source in sources:

    for param in params.get(source):

        if '_GSE' in param:
            vec = param.split('_')[0]
            coords = param.split('_')[1]

            if param=='R_GSE':

                for label in ('start','end'):
                    for comp in ('x','y','z'):
                        msh_times.loc[:,f'{vec}_{comp}_{coords}_{label}_{source}'] = np.nan

            else:

                for comp in ('x','y','z'):
                    msh_times.loc[:,f'{vec}_{comp}_{coords}_{source}'] = np.nan
                    msh_times.loc[:,f'{vec}_{comp}_{coords}_unc_{source}'] = np.nan

                msh_times.loc[:,f'{param}_count_{source}'] = np.nan

        else:
            msh_times.loc[:,f'{param}_{source}'] = np.nan
            msh_times.loc[:,f'{param}_unc_{source}'] = np.nan
            msh_times.loc[:,f'{param}_count_{source}'] = np.nan

# %% Retrieve

### THERE ARE SOME ISSUES WITH 2011-05-23 and 2019-01-22 for magnetic field
### ISSUES WITH 2015-01-23 FOR PLASMA
### ISSUES WITH 2022-02-17 FOR ELECTRIC FIELD

from datetime import datetime
from src.analysing.calculations import calc_average_vector
from uncertainties import unumpy as unp

try:
    for start, interval in msh_times.iterrows():

        if start < datetime(2022,2,18):
            continue

        print(start)

        end = interval['end_time']

        for source in sources:
            for param in params.get(source):

                if '_GSE' in param:
                    vec = param.split('_')[0]
                    coords = param.split('_')[1]

                    if param=='R_GSE':

                        for label, time in zip(('start','end'),(start,end)):
                            position, _ = retrieve_datum(param, source, time)

                            if position is not None:
                                msh_times.loc[start,[f'{vec}_{comp}_{coords}_{label}_{source}' for comp in ('x','y','z')]] = position
                    else:

                        spz_data = retrieve_data(param, source, start, end).dropna()

                        if len(spz_data)>0:

                            vector = calc_average_vector(spz_data, param)

                            msh_times.loc[start,[f'{vec}_{comp}_{coords}_{source}' for comp in ('x','y','z')]] = unp.nominal_values(vector)
                            msh_times.loc[start,[f'{vec}_{comp}_{coords}_unc_{source}' for comp in ('x','y','z')]] = unp.std_devs(vector)
                            msh_times.at[start,f'{param}_count_{source}'] = len(spz_data)

                            if (param=='E_GSE' and source=='OMNI') or (param=='S_GSE'):

                                mean = calc_mean_error(spz_data.loc[:,f'{vec}_mag'])

                                msh_times.at[start,f'{vec}_mag_{source}'] = mean.n
                                msh_times.at[start,f'{vec}_mag_unc_{source}'] = mean.s
                                msh_times.at[start,f'{vec}_mag_count_{source}'] = len(spz_data)

                else:

                    if (param=='E_mag' and source=='OMNI') or (param=='S_mag'):
                        continue
                        # handled with vector above

                    spz_data = retrieve_data(param, source, start, end).dropna()

                    if not spz_data.empty:

                        mean = calc_mean_error(spz_data.loc[:,param])

                        try:
                            msh_times.at[start,f'{param}_{source}'] = mean.n
                            msh_times.at[start,f'{param}_unc_{source}'] = mean.s
                            msh_times.at[start,f'{param}_count_{source}'] = len(spz_data)
                        except:
                            msh_times.at[start,f'{param}_count_{source}'] = 0

                    else:
                        msh_times.at[start,f'{param}_count_{source}'] = 0

                        # Account for omni's error

except KeyboardInterrupt:
    print('\nManual interrupt detected. Returning date looking at...')
    print(start)




# %%

msh_times.to_csv('msh_data_update.csv',index=True)

# Mkae note of different parameters that can be used to filter (quality) and things like should all averages be fore same time/giving more weight to some things than others (e.g. if one interval is 5 entries in GRMB)


# In paper (create draft) state that this isn't exahsutive due to explanation in Richardson & Wild (2025)


# Create plot of B_MSH vs B_SW colour coded by mach number
# Also think about limiting to durations of certain length
# Also plot against spatial position

# As further, could create plot of ratios like Nyugen


# %% Data

from uncertainties import unumpy as unp


parameter = 'B_mag'
parameter = 'N_tot'
parameter = 'E_mag'
parameter = 'V_mag'

if parameter=='B_mag':
    unit = 'nT'
elif parameter=='N_tot':
    unit = r'cm$^{-3}$'
elif parameter=='E_mag':
    unit = r'mV m$^{-1}$'
elif parameter=='V_mag':
    unit = r'km s$^{-1}$'

param_plot = (f'{parameter}_OMNI',f'{parameter}_C1',f'{parameter}_unc_OMNI',f'{parameter}_unc_C1','MA_OMNI')
valid_times = np.ones(len(msh_times),dtype=bool)
for param in param_plot:
    valid_times &= (~np.isnan(msh_times.loc[:,param]))
if parameter=='B_mag':
    valid_times &= (msh_times.loc[:,f'{parameter}_C1']<100)
elif parameter=='N_tot':
    valid_times &= (msh_times.loc[:,f'{parameter}_C1']<10*msh_times.loc[:,f'{parameter}_OMNI'])
elif parameter=='E_mag':
    valid_times &= (msh_times.loc[:,f'{parameter}_C1']<4*msh_times.loc[:,f'{parameter}_OMNI'])
elif parameter=='V_mag':
    valid_times &= (msh_times.loc[:,f'{parameter}_C1']<1.5*msh_times.loc[:,f'{parameter}_OMNI'])


# %% Scatter

xs = msh_times.loc[valid_times,f'{parameter}_OMNI']
ys = msh_times.loc[valid_times,f'{parameter}_C1']
zs = msh_times.loc[valid_times,'MA_OMNI']

xs_unc = msh_times.loc[valid_times,f'{parameter}_unc_OMNI']
ys_unc = msh_times.loc[valid_times,f'{parameter}_unc_C1']


zmin   = np.min(zs)
zmax   = np.max(zs)

fig, ax = plt.subplots(dpi=400)

ax.errorbar(xs, ys, xerr=xs_unc, yerr=ys_unc, fmt='.', ms=0, ecolor='k', capsize=0.5, capthick=0.2, lw=0.2, zorder=1)
scatter = ax.scatter(xs, ys, c=zs, cmap='cool', vmin=1, vmax=10, s=1, alpha=0.9)

#res = fit_continuous_origin_to_horizontal_vectorized(xs, ys)
#plot_fit_continuous(ax, xs, ys, res)

cbar = plt.colorbar(scatter)
cbar.set_label(r'$M_A$')


ax.set_xlabel(f'OMNI {parameter.split("_")[0]}$_{{sw}}$ [{unit}]')
ax.set_ylabel(f'C1 {parameter.split("_")[0]}$_{{msh}}$ [{unit}]')
ax.set_title(f'N = {np.sum(valid_times):,}')

plt.show()


# %% Mach

from src.analysing.calculations import average_of_averages

fig, ax = plt.subplots(dpi=400)


MA_max = 10
step = 0.5

for MA in np.arange(1,MA_max+step,step):
    mask = valid_times.copy()

    if MA==MA_max:
        mask &= msh_times.loc[:,'MA_OMNI']>=MA
    else:
        mask &= (msh_times.loc[:,'MA_OMNI']>=MA) & (msh_times.loc[:,'MA_OMNI']<(MA+step))

    if np.sum(mask)<=2:
        print(f'MA of {MA}: {np.sum(mask)} valid times.')
        continue

    xs = msh_times.loc[mask,f'{parameter}_OMNI']
    ys = msh_times.loc[mask,f'{parameter}_C1']
    zs = msh_times.loc[mask,'MA_OMNI']

    xs_unc = msh_times.loc[mask,f'{parameter}_unc_OMNI']
    ys_unc = msh_times.loc[mask,f'{parameter}_unc_C1']

    xs_counts = msh_times.loc[mask,f'{parameter}_count_OMNI']
    ys_counts = msh_times.loc[mask,f'{parameter}_count_C1']
    zs_counts = msh_times.loc[mask,'MA_count_OMNI']

    x = average_of_averages(xs, xs_counts, xs_unc)
    y = average_of_averages(ys, ys_counts, ys_unc)
    z = average_of_averages(zs, zs_counts)

    ax.errorbar(x.n, y.n, xerr=x.s, yerr=y.s, fmt='.', ms=0, ecolor='k', capsize=0.5, capthick=0.2, lw=0.2, zorder=1)
    if MA==MA_max:
        ax.scatter(x.n, y.n, c='orange', s=12, label=r'$M_A\geq$'+f'{MA_max}')
    else:
        scatter = ax.scatter(x.n, y.n, c=z.n, cmap='cool', vmin=1, vmax=MA_max, s=12)


cbar = plt.colorbar(scatter)
cbar.set_label(r'$M_A$')

ax.legend(loc='upper left')

ax.set_xlabel(f'OMNI {parameter.split("_")[0]}$_{{sw}}$ [{unit}]')
ax.set_ylabel(f'C1 {parameter.split("_")[0]}$_{{msh}}$ [{unit}]')
ax.set_title(f'N = {np.sum(valid_times):,}')

plt.show()




# %% Regimes


fig, ax = plt.subplots(dpi=400)

if parameter=='V_mag':
    step = 50
    maximum = 1000 + step
else:
    step = 1
    maximum = 50 + step

for X_sw in np.arange(0,maximum,step):
    X_mask = valid_times.copy()

    X_mask &= (msh_times.loc[:,f'{parameter}_OMNI']>=X_sw) & (msh_times.loc[:,f'{parameter}_OMNI']<(X_sw+step))

    MA_thresh = 5
    for group, colour in zip(('high','low'),('magenta','cyan')):
        mask = X_mask.copy()
        if group=='low':
            mask &= (msh_times.loc[:,'MA_OMNI']<MA_thresh)
        elif group=='high':
            mask &= (msh_times.loc[:,'MA_OMNI']>=MA_thresh)

        if np.sum(mask)<=1:
            continue

        xs = msh_times.loc[mask,f'{parameter}_OMNI']
        ys = msh_times.loc[mask,f'{parameter}_C1']

        xs_unc = msh_times.loc[mask,f'{parameter}_unc_OMNI']
        ys_unc = msh_times.loc[mask,f'{parameter}_unc_C1']

        xs_counts = msh_times.loc[mask,f'{parameter}_count_OMNI']
        ys_counts = msh_times.loc[mask,f'{parameter}_count_C1']

        x = average_of_averages(xs, xs_counts, xs_unc)
        y = average_of_averages(ys, ys_counts, ys_unc)


        ax.errorbar(x.n, y.n, xerr=x.s, yerr=y.s, fmt='.', ms=0, ecolor='k', capsize=0.5, capthick=0.2, lw=0.2, zorder=1)
        ax.scatter(x.n, y.n, c=colour, s=10)


ax.scatter([], [], c='cyan', s=10, label=r'$M_A<$'+f'{MA_thresh}')
ax.scatter([], [], c='magenta', s=10, label=r'$M_A\geq$'+f'{MA_thresh}')

ax.legend()

ax.set_xlabel(f'OMNI {parameter.split("_")[0]}$_{{sw}}$ [{unit}]')
ax.set_ylabel(f'C1 {parameter.split("_")[0]}$_{{msh}}$ [{unit}]')
ax.set_title(f'N = {np.sum(valid_times):,}')

plt.show()

# %% Time

duration_hours = msh_times.loc[valid_times,'region_duration']/3600

fig, ax = plt.subplots()

ax.hist(duration_hours, bins=np.arange(int(np.max(duration_hours)+1),step=0.5), label=f'{int(np.sum(duration_hours)):,} hours', color='b')
ax.legend()

ax.set_xlabel('MSH Interval Duration [hours]')
ax.set_ylabel(f'Counts ({parameter})')

plt.show()
plt.close()


