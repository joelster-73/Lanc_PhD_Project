# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:22:14 2026

@author: richarj2
"""

# %% lags

from src.methods.saturation.plotting_lags import plot_lags_saturation

lags = (15,16,17,18,19)

plot_lags_saturation('E_R', 'THL', lags, spacecraft='omni', region='sw', resolution='1min', restrict=True)

plot_lags_saturation('E_R', 'THL', lags, spacecraft='combined', region='sw', resolution='1min', restrict=True)

plot_lags_saturation('E_R', 'THL', lags, spacecraft='combined', region='msh', resolution='1min', restrict=True)

# %% resolutions

from src.methods.saturation.plotting_resolutions import plot_resolutions_saturation

resolutions = ('1min','5min','15min')

plot_resolutions_saturation('E_R', 'THL', resolutions, spacecraft='omni', region='sw', lag=17, restrict=True)

plot_resolutions_saturation('E_R', 'THL', resolutions, spacecraft='combined', region='sw', lag=17, restrict=True)

plot_resolutions_saturation('E_R', 'THL', resolutions, spacecraft='combined', region='msh', lag=17, restrict=True)


