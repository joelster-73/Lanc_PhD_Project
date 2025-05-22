# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:20:40 2025

@author: richarj2
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from scipy.interpolate import interp1d
from scipy.optimize import root_scalar

from uncertainties import unumpy as unp
from uncertainties import ufloat

from ..coordinates.spatial import calc_bs_pos
from ..processing.speasy.retrieval import retrieve_data, retrieve_datum

def where_shock_intercept(shock, spacecraft, speasy_variables, R_E=6370):

    # Add uncertainties to this consideration

    v_sh = np.array(shock['v_sh']) / R_E # km/s -> RE/s
    n_sh = np.array(shock[['Nx','Ny','Nz']])

    t0 = shock.name.to_pydatetime()
    r_L1 = shock[['r_x_GSE','r_y_GSE','r_z_GSE']] # RE

    arrival_time = t0 + timedelta(seconds=shock['delay_s'])
    start_time   = arrival_time - timedelta(minutes=20)
    end_time     = arrival_time + timedelta(minutes=20)

    try:
        r_SC = retrieve_data('R_GSE', spacecraft, speasy_variables, start_time, end_time)
        r_SC_times = (r_SC.index - t0).total_seconds()  # Convert timestamps to seconds since t0
        r_SC_interp = [interp1d(r_SC_times, r_SC[col], kind='linear', fill_value='extrapolate') for col in r_SC.columns]
    except:
        return None, None

    def spacecraft_position(t):
        return np.array([interp(t) for interp in r_SC_interp])

    def distance_to_plane(t):
        sc_t = spacecraft_position(t)
        return sc_t.dot(n_sh) - r_L1.dot(n_sh) - v_sh * t

    # Find the interception time
    try:
        result = root_scalar(distance_to_plane, bracket=(0, (end_time - t0).total_seconds()), method='brentq')
        if result.converged:
            t_intercept = result.root
            intercept_time = t0 + timedelta(seconds=t_intercept)
            intercept_position = spacecraft_position(t_intercept)
            return intercept_position, intercept_time
    except ValueError:
        #print(f'Root finding failed for {spacecraft}: {e}')
        pass  # No root found

    return None, None  # No interception

def when_shock_intercept(shock, sc_pos, R_E=6370, unit='s'):
    # r_0 is the point where the shock is measured (at L1) in RE
    # r_1 is the point of interception (Cluster or BS nose) in RE
    # n is the normal
    # v is the speed in km/s

    if sc_pos is None:
        if unit == 'time':
            return None, None
        return None

    v_sh = ufloat(shock['v_sh'], shock['v_sh_unc']) / R_E # km/s -> RE/s
    n_sh = unp.uarray([shock['Nx'], shock['Ny'], shock['Nz']],
                      [shock['Nx_unc'], shock['Ny_unc'], shock['Nz_unc']])

    # Spacecraft positions haven't been accounted for but are incredibly small
    # Because L1 spacecraft move very slowly and Earth spacecraft slow at apogee

    # approximate percentage uncertainty
    r_0 = np.array([shock['r_x_GSE'], shock['r_y_GSE'], shock['r_z_GSE']]) # RE
    r_0_unc = 0.0001
    r_0_u = unp.uarray(r_0, np.abs(r_0_unc*r_0))

    r_1 = np.array(sc_pos)
    r_1_unc = 0.01
    r_1_u = unp.uarray(r_1, np.abs(r_1_unc*r_1))

    time = np.dot(n_sh, (r_1_u - r_0_u)) / v_sh # in s
    time -= ufloat(0, shock['time_s_unc']) # Dominant uncertainty

    if unit == 'm':
        time /= 60
    elif unit == 'h':
        time /= 3600
    elif unit == 'time':
        t0 = shock.name.to_pydatetime()
        return t0 + timedelta(seconds=time.nominal_value), time.std_dev
    return time

def sort_positions(positions, in_solar_wind, shock):

    shock_position = np.array([shock['r_x_GSE'], shock['r_y_GSE'], shock['r_z_GSE']])
    normal = np.array([shock['Nx'], shock['Ny'], shock['Nz']])

    def calculate_dot_product(item):
        try:
            in_sw = in_solar_wind.get(item[0])
            position = np.array([item[1]['x'], item[1]['y'], item[1]['z']])
            dot_product = np.dot(normal, position - shock_position)
            if dot_product is None or np.isnan(dot_product):
                return float('inf')  # Move None/NaN to the end
            if dot_product < 0 or not in_sw:
                return float('inf') # Move negatives and non-sw to the end
            return dot_product
        except KeyError:
            return float('inf')  # Handle missing keys

    sorted_positions = sorted(positions.items(), key=calculate_dot_product)
    return {sc: pos for sc, pos in sorted_positions}


def is_in_solar_wind(spacecraft, time, speasy_variables, **kwargs):

    Pd     = kwargs.get('Pd',None)
    Vsw    = kwargs.get('Vsw',None)
    pos    = kwargs.get('pos',None)
    buffer = kwargs.get('buffer',0)


    if pos is None:
        pos, _ = retrieve_datum('R_GSE', spacecraft, speasy_variables, time)
        pos = {'x': pos[0], 'y': pos[1], 'z': pos[2]}

    pos_df = pd.DataFrame([[pos.get('x'),pos.get('y'),pos.get('z')]], index=[time],
                          columns=[f'r_x_GSE_{spacecraft}',f'r_y_GSE_{spacecraft}',f'r_z_GSE_{spacecraft}'])

    if Vsw is None:
        Vsw, _ = retrieve_datum('V_GSE', 'OMNI', speasy_variables, time)

    if Vsw is not None:
        pos_df[['v_x_GSE_OMNI','v_y_GSE_OMNI','v_z_GSE_OMNI']] = Vsw



    if Pd is None:
        Pd, _ = retrieve_datum('P_dyn', 'OMNI', speasy_variables, time)

    if Pd is not None:
        pos_df['p_flow_OMNI'] = Pd


    r_BS = calc_bs_pos(pos_df, sc_key=spacecraft, time_col='index')
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