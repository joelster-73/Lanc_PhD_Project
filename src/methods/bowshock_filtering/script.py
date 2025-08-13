# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 10:42:58 2025

@author: richarj2
"""
from src.config import PROC_CLUS_DIR_1MIN, C1_RANGE, PROC_OMNI_DIR, PROC_SCBS_DIR

from src.methods.bowshock_filtering.jelinek import generate_bs_df
from src.coordinates.spatial import insert_car_coords
from src.processing.reading import import_processed_data

cluster1 = import_processed_data(PROC_CLUS_DIR_1MIN)
insert_car_coords(cluster1, field='B', coords='GSM', r_col='B_mag', th_col='B_pitch', ph_col='B_clock') # Undoing circular averaging for B

omni = import_processed_data(PROC_OMNI_DIR, date_range=C1_RANGE)

generate_bs_df(cluster1, omni, PROC_SCBS_DIR, 'C1')
