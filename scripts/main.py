# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""

from src.analysing.shocks.discontinuities import approx_normal_vector
from datetime import datetime


detector='WIND'
shock_time = datetime(2021,8,29,19,45,28)
shock_vector = approx_normal_vector(shock_time, detector)
print(shock_vector)