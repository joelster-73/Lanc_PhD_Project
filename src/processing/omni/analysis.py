# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 11:37:36 2026

@author: richarj2
"""

import numpy as np

from scipy.constants import m_p, physical_constants
m_a = physical_constants['alpha particle mass'][0]

from .config import indices_columns

from ...coordinates.magnetic import calc_GSE_to_GSM_angles
from ...analysing.calculations import calc_angle_between_vecs


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


def update_omni(df, drop_cols=indices_columns):

    # Rename temperature for consistency, assuming isothermal
    df.rename(columns={'P_flow': 'P_p', 'T_p': 'T_tot'}, inplace=True)
    df.attrs['units']['P_p'] = 'nPa'

    df.loc[df['T_tot']>=9999999, 'T_tot'] = np.nan # replace fills
    df['T_tot'] /= 1e6 #convert to MK
    df.attrs['units']['T_tot'] = 'MK'

    # OMNI defines pressure as rhoV^2 for just the protons, so halving for consistency
    df['P_flow'] = 0.5 * (df['n_p']*m_p + (1+df['na_np_ratio'])*m_a) * df['V_flow']**2 * 1e21

    df.loc[df['M_A']>100,'M_A'] = np.nan

    df['N_tot'] = df['n_p'] * (1+df['na_np_ratio'])
    df.attrs['units']['N_tot'] = 'n/cc'

    # Theta Bn angle - quasi-perp/quasi-para
    df['theta_Bn'] = calc_angle_between_vecs(df, 'B_GSE', 'R_BSN')

    # restrict to be between 0 and 90 degrees
    df.loc[df['theta_Bn']>np.pi/2,'theta_Bn'] = np.pi - df.loc[df['theta_Bn']>np.pi/2,'theta_Bn']
    df.attrs['units']['theta_Bn'] = 'rad'

    df['gse_to_gsm_angle'] = calc_GSE_to_GSM_angles(df, ref='B')

    # Drop index columns
    df.drop(columns=drop_cols,inplace=True,errors='ignore')