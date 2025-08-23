# -*- coding: utf-8 -*-
"""
Created on Tue May 27 14:47:11 2025

@author: richarj2
"""

import numpy as np
import pandas as pd

from ...coordinates.boundaries import calc_msh_r_diff
from ...processing.speasy.retrieval import retrieve_data, retrieve_datum

def in_solar_wind(spacecraft, time, **kwargs):

    Pd     = kwargs.get('Pd',None)
    Vsw    = kwargs.get('Vsw',None)
    pos    = kwargs.get('pos',None)
    buffer = kwargs.get('buffer',0)


    if pos is None:
        pos, _ = retrieve_datum('R_GSE', spacecraft, time)
        pos = {'x': pos[0], 'y': pos[1], 'z': pos[2]}

    pos_df = pd.DataFrame([[pos.get('x'),pos.get('y'),pos.get('z')]], index=[time],
                          columns=[f'r_x_GSE_{spacecraft}',f'r_y_GSE_{spacecraft}',f'r_z_GSE_{spacecraft}'])

    if Vsw is None:
        Vsw, _ = retrieve_datum('V_GSE', 'OMNI', time)

    if Vsw is not None:
        pos_df[['v_x_GSE_OMNI','v_y_GSE_OMNI','v_z_GSE_OMNI']] = Vsw



    if Pd is None:
        Pd, _ = retrieve_datum('P_dyn', 'OMNI', time)

    if Pd is not None:
        pos_df['p_flow_OMNI'] = Pd


    r_BS = calc_msh_r_diff(pos_df, 'BS', position_key=spacecraft, data_key='OMNI', time_col='index')
    r_BS = r_BS.iloc[0]

    return r_BS['r_BS_diff'] > buffer

def is_in_solar_wind(spacecraft, speasy_variables, start, end, **kwargs):

    Pd      = kwargs.get('Pd',None)
    Vsw     = kwargs.get('Vsw',None)
    pos_df  = kwargs.get('pos_df',None)
    buffer  = kwargs.get('buffer',0)
    pos_var = kwargs.get('position_var','R_GSE')
    shock   = kwargs.get('shock',None)

    shock = None # temporary fix as not passing shock info in

    if pos_df is None:
        pos_df = retrieve_data(pos_var, spacecraft, start, end, upsample=True)
    if pos_df is None:
        return False # no position data with which to determine if in solar wind

    pos_df = pos_df.rename(columns={f'{pos_var}_x':f'r_x_GSE_{spacecraft}',
                                    f'{pos_var}_y':f'r_y_GSE_{spacecraft}',
                                    f'{pos_var}_z':f'r_z_GSE_{spacecraft}'})

    # Using upstream values (i.e. before shock passes) as default
    # Quieter conditions so bow shock less compressed so more conservative approach
    if shock is not None:
        def_v_x = shock['v_x_GSE_up']
        def_v_y = shock['v_y_GSE_up']
        def_v_z = shock['v_z_GSE_up']

        velocity = np.sqrt(def_v_x**2+def_v_y**2+def_v_z**2)
        def_p_d = approximate_pressure(shock['ni_up'],velocity)
    else:
        def_v_x = -400
        def_v_y, def_v_z = 0,0
        def_p_d = np.float32(2.056)

    if Vsw is None:
        Vsw = retrieve_data('V_GSE', 'OMNI', start, end)

    if Vsw is not None:
        for comp, default in zip(('x','y','z'),(def_v_x,def_v_y,def_v_z)):
            try:
                pos_df[f'v_{comp}_GSE_OMNI'] = Vsw[f'V_GSE_{comp}'].interpolate(method='linear', limit_direction='forward', axis=0)
                pos_df[f'v_{comp}_GSE_OMNI'] = pos_df[f'V_GSE_{comp}'].fillna(default)
            except Exception:
                pos_df[f'v_{comp}_GSE_OMNI'] = Vsw[f'V_GSE_{comp}'].fillna(default)

    if Pd is None:
        Pd = retrieve_data('P_dyn', 'OMNI', start, end)

    if Pd is not None:
        try:
            pos_df['p_flow_OMNI'] = Pd['P_dyn'].interpolate(method='linear', limit_direction='forward', axis=0)
            pos_df['p_flow_OMNI'] = pos_df['P_dyn'].fillna(def_p_d)
        except Exception:
            pos_df['p_flow_OMNI'] = Pd['P_dyn'].fillna(def_p_d)

    r_BS = calc_msh_r_diff(pos_df, 'BS', position_key=spacecraft, data_key='OMNI', time_col='index')
    r_BS = r_BS.iloc[0]

    return r_BS['r_bs_diff'] > buffer

def approximate_pressure(number_density, velocity):
    """
    If no pressure is provided by OMNI
    Use the downstream conditions of the bulk velocity and density to approximate pressure
    In line with OMNI's definition, P = (2*10**-6)*Np*Vp**2 nPa (N in cm**-3, Vp in km/s)
    """
    m = 2*10**(-27)
    n = number_density * 1e6 # n/cc to n/m3
    v = np.linalg.norm(velocity) * 1e3 # km/s to m/s
    p = m * n * v**2 / 1e-9 # nPa
    return p
