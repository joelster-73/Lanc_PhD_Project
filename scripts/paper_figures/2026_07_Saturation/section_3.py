# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:22:14 2026

@author: richarj2
"""

# %% lags

from src.methods.saturation.plotting_lags import plot_lags_saturation

lags = (15,16,17,18,19)

for sc, reg in (('omni','sw'),('combined','sw'),('combined','msh')):

    plot_lags_saturation('E_R', 'THL', lags, spacecraft=sc, region=reg, resolution='1min', restrict=True)


lags = (0,5,10,15,20,25,30)

for sc, reg in (('omni','sw'),('combined','sw'),('combined','msh')):

    plot_lags_saturation('E_R', 'THL', lags, spacecraft=sc, region=reg, resolution='5min', restrict=True)

# %% resolutions

from src.methods.saturation.plotting_resolutions import plot_resolutions_saturation

resolutions = ('1min','5min','15min')

for sc, reg in (('omni','sw'),('combined','sw'),('combined','msh')):

    plot_resolutions_saturation('E_R', 'THL', resolutions, spacecraft=sc, region=reg, lag=17, restrict=True)


