# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 11:37:36 2026

@author: richarj2
"""

import numpy as np

def calc_omni_uncertainty(df_omni, column, dt_err_col='rms_timeshift'):
    """
    Estimate the uncertainty on a column in df_omni due to uncertainty in the applied time shift:
        sigma_X = |dX/dt| * sigma_dt
    """
    X = df_omni[column].values
    t_sec = df_omni.index.astype('int64')/1e9 # seconds since epoch

    dX_dt = np.gradient(X, t_sec)
    sigma_dt = df_omni[dt_err_col].values

    return np.abs(dX_dt) * sigma_dt