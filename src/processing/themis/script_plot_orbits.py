# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 10:56:14 2025

@author: richarj2
"""

from src.processing.themis.plot_orbits import plot_themis_position
from src.processing.themis.config import PROC_THEMIS_DIRECTORIES

all_spacecraft = ('tha','thb','thc','thd','the')
non_artemis    = ('tha','thd','the')
artemis        = ('thb','thc')

plot_themis_position(PROC_THEMIS_DIRECTORIES, probes=all_spacecraft, year=(2008,2009))
plot_themis_position(PROC_THEMIS_DIRECTORIES, probes=non_artemis)
plot_themis_position(PROC_THEMIS_DIRECTORIES, probes=artemis)


