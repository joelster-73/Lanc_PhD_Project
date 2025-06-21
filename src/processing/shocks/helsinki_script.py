# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 16:56:39 2025

@author: richarj2
"""

# %%
import os
from ...config import HELSINKI_DIR
from ..writing import write_to_cdf
from .helsinki import process_helsinki_shocks

helsinki_shocks = process_helsinki_shocks(HELSINKI_DIR, 'Helsinki_database.dat')

output_file = os.path.join(HELSINKI_DIR, 'helsinki_shocks.cdf')
write_to_cdf(helsinki_shocks,output_file,reset_index=True)

