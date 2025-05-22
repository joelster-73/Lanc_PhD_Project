# -*- coding: utf-8 -*-
"""
Created on Mon May 19 17:09:10 2025

@author: richarj2
"""
import os
from datetime import datetime

from src.config import PROC_CFA_DIR
from src.processing.reading import import_processed_data
from src.plotting.shocks import plot_all_shocks


shocks = import_processed_data(PROC_CFA_DIR)
shocks.loc[(shocks['spacecraft']=='dscover') | (shocks['spacecraft']=='dscovr'),'spacecraft'] = 'dsc' # consistency with rest of naming conventions
#shocks.attrs

script_dir = os.getcwd() # change to location of script __file__

create_csv = False
if create_csv:
    file_name = 'data_distribution.csv'
    file_path = os.path.join(script_dir, 'scripts', file_name)
    shocks.to_csv(file_path, index=True)


# %%
time_shock = datetime(2013,3,17)

plot_all_shocks(shocks, parameter='B_mag', time=None, start_printing=datetime(2017,9,7))


# %%

