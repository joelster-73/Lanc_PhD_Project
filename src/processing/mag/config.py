# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 12:52:37 2025

@author: richarj2
"""

PC_times = (15,16,17,18,19)
AE_times = (50,51,52,53,54,55)

PC_STATIONS = ('THL','SVS','ALE','NRD','DMH','RES')

lagged_indices = {'AE':  AE_times, 'AEc': AE_times,     'SME': AE_times,
                  'PCN': PC_times, 'PCC': PC_times,
                  'AA': (60,)}

for station in PC_STATIONS:
    for col in (station,f'{station}_unc',f'{station}_y_GSE',f'{station}_y_GSM'):
        lagged_indices[col] = PC_times