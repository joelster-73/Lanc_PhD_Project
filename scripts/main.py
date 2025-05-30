# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""


# %% Importing
from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data

shocks_intercepts = import_processed_data(PROC_SHOCKS_DIR)

# %%
import numpy as np
import matplotlib.pyplot as plt

columns = [col for col in shocks_intercepts if '_coeff' in col]

all_coeffs = shocks_intercepts[columns].to_numpy().flatten()
all_coeffs = all_coeffs[~np.isnan(all_coeffs)]
all_coeffs = all_coeffs[(all_coeffs>0)&(all_coeffs<1)]

bin_width=0.01
bins = np.arange(start=0, stop=1+bin_width, step=bin_width)

fig, ax = plt.subplots()

ax.hist(all_coeffs, bins=bins)
plt.show()