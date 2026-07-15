# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:22:14 2026

@author: richarj2
"""

# %% lags

from src.methods.saturation.plotting_lags import plot_different_lags_saturation

plot_different_lags_saturation('E_R', 'THL', (15,16,17,18,19), spacecraft='omni', region='sw', resolution='1min', restrict=True)

plot_different_lags_saturation('E_R', 'THL', (15,16,17,18,19), spacecraft='combined', region='sw', resolution='1min', restrict=True)

plot_different_lags_saturation('E_R', 'THL', (15,16,17,18,19), spacecraft='combined', region='msh', resolution='1min', restrict=True)

# %% resolutions