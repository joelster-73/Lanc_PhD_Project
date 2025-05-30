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

shocks = import_processed_data(PROC_CFA_DIR)

# %% Intercept spacecraft

import os
from src.config import PROC_SHOCKS_DIR
from src.analysing.shocks.intercepts import find_all_shocks
from src.processing.writing import write_to_cdf

output_file = os.path.join(PROC_SHOCKS_DIR, 'cfa_shocks_intercepts.cdf')
shocks_with_intercepts = find_all_shocks(shocks,'B_mag')
shocks_with_intercepts.reset_index(inplace=True)

write_to_cdf(shocks_with_intercepts,output_file)

# %%
from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data
from src.plotting.shocks import plot_all_shocks

shocks_intercepts = import_processed_data(PROC_SHOCKS_DIR)

plot_all_shocks(shocks_intercepts, 'B_mag', plot_in_sw=True, plot_positions=True)