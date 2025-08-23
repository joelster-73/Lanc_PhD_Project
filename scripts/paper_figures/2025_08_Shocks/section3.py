# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 08:43:58 2025

@author: richarj2
"""

# %% Imports
from src.config import HELSINKI_DIR
from src.processing.reading import import_processed_data
helsinki_shocks = import_processed_data(HELSINKI_DIR)


# %% Compression

from src.plotting.shocks import plot_compressions_both

plot_compressions_both(helsinki_shocks, plot_type='hist')


# Restructure files
# Create "paper figures" scripts
# Make other comparison side by side
# Also make the plot for showing performance
# Probably make one grid plot for performance of changing all parameters