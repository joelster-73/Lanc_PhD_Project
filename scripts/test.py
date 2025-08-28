# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""
from src.config import CROSSINGS_DIR
from src.processing.reading import import_processed_data

import pandas as pd


crossings = import_processed_data(CROSSINGS_DIR)
cross_labels = crossings.attrs['crossings']

msh_times = crossings.loc[crossings['loc_num']==10].copy()
msh_times.loc[:,'end_time'] = msh_times.index + pd.to_timedelta(msh_times.loc[:,'region_duration'], unit='s')


# %%
duration_hours = msh_times.loc[:,'region_duration']/3600

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

ax.hist(duration_hours, bins=np.arange(int(np.max(duration_hours)+1)), label=f'{np.sum(duration_hours):,} hours')
ax.legend()

plt.show()
plt.close()

# %%

from src.processing.speasy.retrieval import retrieve_data, retrieve_datum
from src.analysing.calculations import calc_mean_error


sources = ('C1','OMNI')
params = {'C1': ('R_GSE','B_mag'), 'OMNI': ('B_mag','MA')}

for source in sources:

    for param in params.get(source):

        if param=='R_GSE':
            vec = param.split('_')[0]
            coords = param.split('_')[1]

            for label in ('start','end'):
                for comp in ('x','y','z'):
                    msh_times.loc[:,f'{vec}_{comp}_{coords}_{label}_{source}'] = np.nan

        else:
            msh_times.loc[:,f'{param}_{source}'] = np.nan
            msh_times.loc[:,f'{param}_unc_{source}'] = np.nan
            msh_times.loc[:,f'{param}_count_{source}'] = np.nan

# %%
for start, interval in msh_times.iterrows():

    end = interval['end_time']

    for source in sources:
        for param in params.get(source):

            if param=='R_GSE':
                vec = param.split('_')[0]
                coords = param.split('_')[1]

                for label, time in zip(('start','end'),(start,end)):
                    position, _ = retrieve_datum(param, source, time)

                    if position is not None:
                        msh_times.loc[start,[f'{vec}_{comp}_{coords}_{label}_{source}' for comp in ('x','y','z')]] = position

            else:

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


# %%

# In paper (create draft) state that this isn't exahsutive due to explanation in Richardson & Wild (2025)


# Create plot of B_MSH vs B_SW colour coded by mach number
# Also think about limiting to durations of certain length
# Also plot against spatial position