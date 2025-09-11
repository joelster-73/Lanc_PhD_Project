# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 16:56:14 2025

@author: richarj2
"""

import numpy as np

def kan_lee_field(df):

    # Kan and Lee Electric Field
    # E_R = |V| * B_T * sin^2 (clock/2)

    if 'V_flow' in df:
        V = df.loc[:,'V_flow'].copy()
    elif 'V_mag' in df:
        V = df.loc[:,'V_mag'].copy()
    else:
        raise Exception('No velocity column')
    V *= 1e3

    if 'B_y_GSE' in df and 'B_z_GSE' in df:
        BT = np.sqrt(df.loc[:,'B_y_GSE']**2+df.loc[:,'B_z_GSE']**2)
    elif 'B_y_GSM' in df and 'B_z_GSM' in df:
        BT = np.sqrt(df.loc[:,'B_y_GSM']**2+df.loc[:,'B_z_GSM']**2)
    else:
        raise Exception('No magnetic field column')
    BT *= 1e-9

    try:
        clock = df.loc[:,'B_clock']
    except:
        raise Exception('No clock angle column')

    df['E_R'] = (V * BT * (np.sin(clock/2))**2) * 1e3