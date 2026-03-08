# -*- coding: utf-8 -*-
'''
Created on Sat Feb 28 14:22:13 2026

@author: richarj2
'''

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')


delay = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
time_bins = ['01-07', '07-13', '13-19', '19-01']
x = np.arange(len(time_bins))

nov_feb = np.array([
    [0.619, 0.651, 0.647, 0.622],
    [0.638, 0.680, 0.678, 0.647],
    [0.659, 0.709, 0.707, 0.675],
    [0.674, 0.720, 0.721, 0.689],
    [0.677, 0.716, 0.717, 0.689],
    [0.674, 0.706, 0.703, 0.683],
    [0.670, 0.696, 0.688, 0.676],
    [0.663, 0.688, 0.675, 0.669],
    [0.658, 0.682, 0.665, 0.663],
])

mar_apr_sep_oct = np.array([
    [0.684, 0.679, 0.652, 0.663],
    [0.704, 0.705, 0.676, 0.685],
    [0.725, 0.733, 0.702, 0.711],
    [0.739, 0.749, 0.718, 0.726],
    [0.742, 0.747, 0.719, 0.729],
    [0.739, 0.735, 0.709, 0.725],
    [0.733, 0.721, 0.697, 0.718],
    [0.727, 0.708, 0.683, 0.709],
    [0.724, 0.697, 0.670, 0.700],
])

may_aug = np.array([
    [0.673, 0.653, 0.554, 0.692],
    [0.697, 0.672, 0.584, 0.718],
    [0.722, 0.692, 0.618, 0.745],
    [0.738, 0.704, 0.641, 0.761],
    [0.742, 0.703, 0.646, 0.765],
    [0.739, 0.696, 0.635, 0.759],
    [0.733, 0.685, 0.614, 0.748],
    [0.726, 0.673, 0.589, 0.737],
    [0.717, 0.662, 0.565, 0.725],
])

data = np.stack([nov_feb, mar_apr_sep_oct, may_aug])


X, Y = np.meshgrid(x, delay)

season_names = ['Winter', 'Equinox', 'Summer']
colour_maps  = ['winter', 'spring', 'summer']
season_index = 0  # change to 1 or 2

# %% plot
fig, ax = plt.subplots(figsize=(6,5), dpi=300)

cbar = plt.contourf(X, Y, data[season_index], levels=10, cmap=colour_maps[season_index])
_    = plt.contour(X, Y, data[season_index], levels=10, colors='black', linewidths=0.5)

cbar = plt.colorbar(cbar)
cbar.ax.set_ylabel('R', rotation=0, labelpad=15)

ax.set_xticks(x, time_bins)
ax.set_xlabel('Time [UT]')
ax.set_ylabel('Delay [mins]')
plt.title(season_names[season_index])
plt.tight_layout()
plt.show()

# %%

