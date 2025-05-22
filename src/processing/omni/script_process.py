# -*- coding: utf-8 -*-
"""
Created on Mon May 19 23:02:24 2025

@author: joels
"""

# Process all the data from the CDF files and save to a new CDF file
from src.processing.omni.config import omni_variables
from src.processing.omni.handling import process_omni_files

from src.config import LUNA_OMNI_DIR, PROC_OMNI_DIR

process_omni_files(LUNA_OMNI_DIR, PROC_OMNI_DIR, omni_variables)