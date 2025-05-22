# -*- coding: utf-8 -*-
"""
Created on Mon May 12 11:36:12 2025

@author: richarj2
"""

from src.processing.cfa.config import HTML_TAG_STRINGS, HTML_TAG_LABELS, HTML_TAGS_UNCERTAINTY
from src.processing.cfa.handling import get_all_shocks

from src.config import PROC_CFA_DIR

get_all_shocks(HTML_TAG_STRINGS, HTML_TAG_LABELS, HTML_TAGS_UNCERTAINTY, PROC_CFA_DIR)
