# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:42:09 2025

@author: richarj2
"""

from src.processing.mag.indices import build_averaged_index


for index in ('Dst','SMR','AE','SME','SYM','ASY','PC','THL','SVS','ALE','NRD','DMH','RES'):

    if index in ('Dst','SMR','AE','SME','SYM','ASY','PC'):
        continue # skips already processed ones

    build_averaged_index(index)
