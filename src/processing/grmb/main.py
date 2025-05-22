# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:31:34 2025

@author: richarj2
"""

# Process crossing data and save to a new CDF file
from handling_crossings import process_crossing_file
process_crossing_file(luna_cros_dir, proc_cros_dir, crossing_variables)

from handling_crossings import generate_bs_df
generate_bs_df(None, omni, proc_scbs_dir, 'C1', None, df_sc=cluster1)
