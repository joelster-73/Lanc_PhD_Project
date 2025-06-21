# -*- coding: utf-8 -*-
"""
Created on Mon May 12 11:36:12 2025

@author: richarj2
"""

# %% Scraping

from .config import HTML_TAG_STRINGS, HTML_TAG_LABELS, HTML_TAGS_UNCERTAINTY, HTML_TAGS_UP_DW
from .handling import get_all_shocks

from ...config import PROC_CFA_DIR

get_all_shocks(HTML_TAG_STRINGS, HTML_TAG_LABELS, HTML_TAGS_UNCERTAINTY, HTML_TAGS_UP_DW, PROC_CFA_DIR)


# %% Importing
from ...config import PROC_CFA_DIR
from ..processing.reading import import_processed_data

cfa_shocks = import_processed_data(PROC_CFA_DIR)
# %% Donki
from ..processing.shocks.donki import combine_cfa_donki

shocks = combine_cfa_donki(cfa_shocks)

# %%
import os
from ...config import PROC_SHOCKS_DIR
from ...analysing.shocks.intercepts import find_all_shocks
from ...writing import write_to_cdf

output_file = os.path.join(PROC_SHOCKS_DIR, 'cfa_donki_shocks.cdf')

shocks_with_intercepts = find_all_shocks(shocks,'field')

write_to_cdf(shocks_with_intercepts,output_file,reset_index=True)

