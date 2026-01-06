# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:42:09 2025

@author: richarj2
"""


# %%
from src.processing.reading import import_processed_data
from src.processing.mag.plotting import plot_mag_data, plot_magnetometer_map

omni = import_processed_data('omni', resolution='1min')
omni = omni.shift(freq='17min')

THL      = import_processed_data('supermag', dtype='THL', resolution='gse')
indices  = import_processed_data('indices', file_name='combined_1min')

# %%

year, month, day_min, day_max = (2008, 5, 10, 11) # Quiet day
#year, month, day_min, day_max = (2015, 3, 17, 18) # Max PCN


thl_sub  = THL.loc[(THL.index.year==year)&(THL.index.month==month)&(THL.index.day>=day_min)&(THL.index.day<day_max)]
omni_sub = omni.loc[(omni.index.year==year)&(omni.index.month==month)&(omni.index.day>=day_min)&(omni.index.day<day_max)]
indices_sub = indices.loc[(indices.index.year==year)&(indices.index.month==month)&(indices.index.day>=day_min)&(indices.index.day<day_max)]


# %% plot

import itertools as it

for (ind, coords, quantity) in it.product(('ER','Ey'),('GSE','GSM'),('mag','phi','tr')):

    if coords=='GSM' and quantity=='mag':
        continue
    elif ind=='Ey' and quantity!='phi':
        continue
    elif ind=='ER' and quantity=='phi':
        continue

    ### add to function to save into file active/quiet_YYMMDD
    plot_mag_data(thl_sub, omni_sub, indices_sub, coords=coords, quantity=quantity, ind=ind)


# %% Map
plot_magnetometer_map(thl_sub, df_sw=omni_sub, coords='GSE')

