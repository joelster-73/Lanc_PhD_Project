# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:24:20 2025

@author: richarj2
"""

omni_variables = ('time','satellite','B_field','pressure','velocity','density','bow_shock_nose','propagation','electric_field','mach_number','auroral_electrojet')

omni_spacecraft = {
    71: 'Ace',
    60: 'Geo',
    50: 'IMP8',
    51: 'Wind',
    52: 'Wind-v2',
    99: 'Bad Data'
}

omni_columns = ['Year', 'Day', 'Hour', 'Minute', 'imf_sc', 'plasma_sc', 'imf_counts', 'plasma_counts', 'percent_interp', 'prop_time_s', 'rms_timeshift', 'rms_minvar', 'dt_seconds', 'B_avg', 'B_x_GSE', 'B_y_GSE', 'B_z_GSE', 'B_y_GSM', 'B_z_GSM', 'B_avg_rms', 'B_vec_rms', 'V_flow', 'V_x_GSE', 'V_y_GSE', 'V_z_GSE', 'n_p', 'T_p', 'na_np_ratio', 'P_flow', 'E_y', 'beta', 'M_A', 'M_ms', 'R_x_GSE', 'R_y_GSE', 'R_z_GSE', 'R_x_BSN', 'R_y_BSN', 'R_z_BSN', 'AE', 'AL', 'AU', 'SYM_D', 'SYM_H', 'ASY_D', 'ASY_H']

omni_columns_5min = omni_columns + ['PSI_P_10', 'PSI_P_30', 'PSI_P_60']

lagged_indices = {'AE': (53,), 'AEc': (53,), 'PCN': (17,), 'PCC': (17,), 'AA': (17,), 'SME': (53,) }

imf_bad_cols = ['B_avg', 'B_x_GSE', 'B_y_GSE', 'B_z_GSE', 'B_y_GSM', 'B_z_GSM', 'B_avg_rms', 'B_vec_rms', 'R_x_BSN', 'R_y_BSN', 'R_z_BSN', 'prop_time_s', 'E_y', 'M_A', 'beta']

plasma_bad_cols = ['P_flow', 'n_p', 'T_p', 'na_np_ratio', 'V_flow', 'V_x_GSE', 'V_y_GSE', 'V_z_GSE', 'R_x_GSE', 'R_y_GSE', 'R_z_GSE', 'E_y', 'M_A', 'M_ms', 'beta']

column_units = {
    'Year'          : 'yr',
    'Day'           : 'day',
    'Hour'          : 'hr',
    'Minute'        : 'min',
    'imf_sc'        : '1',
    'plasma_sc'     : '1',
    'imf_counts'    : '1',
    'plasma_counts' : '1',
    'percent_interp': '%',
    'prop_time_s'   : 's',
    'rms_timeshift' : 's',
    'rms_minvar'    : 'nT',
    'dt_seconds'    : 's',
    'B_avg'         : 'nT',
    'B_x_GSE'       : 'nT',
    'B_y_GSE'       : 'nT',
    'B_z_GSE'       : 'nT',
    'B_y_GSM'       : 'nT',
    'B_z_GSM'       : 'nT',
    'B_avg_rms'     : 'nT',
    'B_vec_rms'     : 'nT',
    'V_flow'        : 'km/s',
    'V_x_GSE'       : 'km/s',
    'V_y_GSE'       : 'km/s',
    'V_z_GSE'       : 'km/s',
    'V_y_GSM'       : 'km/s',
    'V_z_GSM'       : 'km/s',
    'E_x_GSM'       : 'mV/m',
    'E_y_GSM'       : 'mV/m',
    'E_z_GSM'       : 'mV/m',
    'S_mag'         : 'uW/m2',
    'S_x_GSM'       : 'uW/m2',
    'S_y_GSM'       : 'uW/m2',
    'S_z_GSM'       : 'uW/m2',
    'n_p'           : 'n/cc',
    'T_p'           : 'K',
    'na_np_ratio'   : '1',
    'P_flow'        : 'nPa',
    'E_y'           : 'mV/m',
    'beta'          : '1',
    'M_A'           : '1',
    'M_ms'          : '1',
    'R_x_GSE'       : 'Re',
    'R_y_GSE'       : 'Re',
    'R_z_GSE'       : 'Re',
    'R_x_BSN'       : 'Re',
    'R_y_BSN'       : 'Re',
    'R_z_BSN'       : 'Re',
    'AE'            : 'nT',
    'AL'            : 'nT',
    'AU'            : 'nT',
    'SYM_D'         : 'nT',
    'SYM_H'         : 'nT',
    'ASY_D'         : 'nT',
    'ASY_H'         : 'nT',
    'PSI_P_10'      : '1/scm2sr',
    'PSI_P_30'      : '1/scm2sr',
    'PSI_P_60'      : '1/scm2sr'
}

