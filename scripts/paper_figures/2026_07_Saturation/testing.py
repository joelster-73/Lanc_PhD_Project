# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


from src.processing.reading import import_processed_data, import_processed_spacecraft

df_sc = import_processed_data('sw', dtype='plasma', resolution='5min', file_name='sw_times_combined')


df_m1 = import_processed_spacecraft('mms1', populations=['state','fgm','fpi'], resolution='15min')
df_te = import_processed_spacecraft('the', populations=['STATE','FGM','msh'], resolution='15min')



