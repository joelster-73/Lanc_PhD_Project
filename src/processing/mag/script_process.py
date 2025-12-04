# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:42:09 2025

@author: richarj2
"""

from src.processing.mag.handling import process_supermag_data


THL = process_supermag_data('THL')

# %%
from matplotlib import pyplot as plt
import numpy as np

fig, ax = plt.subplots()

year = THL.loc[THL.index.year==2024]
ax.scatter(year.index,np.linalg.norm(year[['B_n_GEO','B_e_GEO','B_z_GEO']],axis=1))

plt.show()
