# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:23:26 2025

@author: richarj2
"""

wind_variables = { # 1 minute intervals (at midpoints, i.e. 30 seconds)
    'epoch' : 'Epoch', # epoch time in milliseconds (scalar).
    'r'     : 'PGSE',  # position in Re=6738km from centre of Earth (vector).
    'B_avg' : 'BF1',   # fgm nT (magnitude).
    'B_GSE' : 'BGSE',  # fgm nT (vector).
    'B_GSM' : 'BGSM',  # fgm nT (vector).
}

wind_variables_3_sec = { # at 3 second intervals (at midpoints, i.e. 1.5 seconds)
    'epoch' : 'Epoch3', # No 3-sec res on position
    'B_avg' : 'B3F1',
    'B_GSE' : 'B3GSE',
    'B_GSM' : 'B3GSM',
}

wind_variables_1_hour = { # at 1 hour intervals (at midpoints, i.e. 30 minutes)
    'epoch' : 'Epoch1',
    'r'     : 'P1GSE',
    'B_avg' : 'B1F1',
    'B_GSE' : 'B1GSE',
    'B_GSM' : 'B1GSM',
}