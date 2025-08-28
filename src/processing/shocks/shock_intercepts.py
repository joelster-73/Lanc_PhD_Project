# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 14:52:14 2025

@author: richarj2
"""

from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data

all_processed_shocks = import_processed_data(PROC_SHOCKS_DIR, file_name='all_shocks.cdf')

# %% Initialise
from src.methods.shock_intercepts.intercepts import find_all_shocks
shocks_intercepted = find_all_shocks(all_processed_shocks, 'B_mag')


#shocks_intercepted = find_all_shocks(all_processed_shocks, 'B_mag', shocks_intercepts_started=shocks_intercepted, starting_ID=1402)

# usually gets stuck at #1402

# %%
import os
from src.processing.writing import write_to_cdf
output_file = os.path.join(PROC_SHOCKS_DIR, 'shocks_with_intercepts.cdf')

write_to_cdf(shocks_intercepted, output_file, attributes={'time_col': 'none'})

# %%

