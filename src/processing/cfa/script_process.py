# -*- coding: utf-8 -*-
"""
Created on Mon May 12 11:36:12 2025

@author: richarj2
"""

# %% Scraping

from src.processing.cfa.config import HTML_TAG_STRINGS, HTML_TAG_LABELS, HTML_TAGS_UNCERTAINTY, HTML_TAGS_UP_DW
from src.processing.cfa.handling import get_all_shocks

from src.config import PROC_CFA_DIR

get_all_shocks(HTML_TAG_STRINGS, HTML_TAG_LABELS, HTML_TAGS_UNCERTAINTY, HTML_TAGS_UP_DW, PROC_CFA_DIR)


# %% Importing
from src.config import PROC_CFA_DIR
from src.processing.reading import import_processed_data

cfa_shocks = import_processed_data(PROC_CFA_DIR)
# %% Donki
from src.processing.cfa.donki import combine_cfa_donki

shocks = combine_cfa_donki(cfa_shocks)
# %% Intercept spacecraft

import os
from src.config import PROC_SHOCKS_DIR
from src.analysing.shocks.intercepts import find_all_shocks
from src.processing.writing import write_to_cdf

output_file = os.path.join(PROC_SHOCKS_DIR, 'cfa_donki_shocks.cdf')

shocks_with_intercepts = find_all_shocks(shocks,'B_mag')

write_to_cdf(shocks_with_intercepts,output_file,reset_index=True)


# %%
from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data
from src.plotting.shocks import plot_all_shocks

from datetime import datetime

shocks_intercepts = import_processed_data(PROC_SHOCKS_DIR)

plot_all_shocks(shocks_intercepts, 'B_mag', plot_in_sw=True, plot_positions=True, start_printing=datetime(2011,12,31))

# %%