# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""

from src.processing.themis.config import LUNA_THEMIS_DIRECTORIES, PROC_THEMIS_DIRECTORIES, THEMIS_VARIABLES_FGM, THEMIS_VARIABLES_STATE
from src.processing.themis.handling import process_themis_files

# %%
all_spacecraft = ('tha','thb','thc','thd','the')

for spacecraft in all_spacecraft:

    process_themis_files(spacecraft, LUNA_THEMIS_DIRECTORIES, PROC_THEMIS_DIRECTORIES, THEMIS_VARIABLES_FGM, THEMIS_VARIABLES_STATE,
                     sub_folders=True)

