# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 13:43:31 2025

@author: richarj2
"""
import warnings

import numpy as np
import re
import operator
from collections import defaultdict

from ....plotting.formatting import create_label, data_string

minimum_counts = {'1min': 100, '5min': 50, '15min': 25}

minimum_sample = {'1min': 5, '5min': 25, '15min': 125} # at least 5 counts per min for stats

imf_cols = ['B_avg', 'B_x_GSE', 'B_y_GSE', 'B_z_GSE', 'B_y_GSM', 'B_z_GSM', 'B_avg_rms', 'B_vec_rms', 'R_x_BSN', 'R_y_BSN', 'R_z_BSN', 'prop_time_s', 'E_y', 'M_A', 'beta']

plasma_cols = ['P_flow', 'n_p', 'T_p', 'na_np_ratio', 'V_flow', 'V_x_GSE', 'V_y_GSE', 'V_z_GSE', 'R_x_GSE', 'R_y_GSE', 'R_z_GSE', 'E_y', 'M_A', 'M_ms', 'beta', 'E_mag', 'E_x_GSE', 'E_y_GSM', 'E_z_GSM', 'S_mag', 'S_x_GSE', 'S_y_GSM', 'S_z_GSM', 'E_R']

def shift_angular_data(df, *cols, inplace=True):
    # Shift angular data to centre lies at +-180 rather than 0

    if not inplace:
        df = df.copy()

    for to_shift in cols:
        shift_unit = df.attrs.get('units',{}).get(to_shift,'')
        if shift_unit == 'rad':
            df[to_shift] = (df[to_shift] + 2*np.pi) % (2*np.pi)
        elif shift_unit in ('deg','°'):
            df[to_shift] = (df[to_shift] + 360) % 360

    if not inplace:
        return df

def mask_df(df, col, limits=None, count_col=None, resolution='1min', condition=None, cond_col=None, invert=False):

        mask = ~df[col].isna()

        if limits is not None:

            if limits[0] is not None:
                if limits[0]==0:
                    mask &= ((df[col] > 0) | ((df[col] == 0) & ~np.signbit(df[col])))
                else:
                    mask &= df[col] >= limits[0]

            if limits[-1] is not None:
                if limits[1]==0:
                    mask &= ((df[col] < 0) | ((df[col] == 0) & np.signbit(df[col])))
                else:
                    mask &= df[col] <= limits[1]

        if count_col is not None:

            if count_col not in df:

                warnings.warn(f'Column {count_col} not in dataframe.')

            else:
                min_count = minimum_sample.get(resolution, 5)
                initial = np.sum(mask)
                mask &= df[count_col] >= min_count
                print(f'Removing low counts (below {min_count}); removed {initial-np.sum(mask):,} ({col}).')

        if condition is not None:

            cond_col = col if cond_col is None else cond_col

            operation, value = parse_condition(condition)

            mask &= operation(df[cond_col], value)

            #print(f'Applied condition: {cond_col} {condition}.')

        df_masked = df.loc[mask].copy()

        if invert:
            df_masked[col] *= -1
            df_masked.rename(columns={col: f'-{col}'}, inplace=True)
            df_masked.attrs.get('units',{})[f'-{col}'] = df_masked.attrs.get('units',{}).get(col)

        return df_masked

def parse_condition(condition):

    ops = {
        '<': operator.lt,
        '>': operator.gt,
        '<=': operator.le,
        '>=': operator.ge,
        '==': operator.eq,
        '!=': operator.ne
    }

    match = re.fullmatch(
        r'\s*(<=|>=|==|!=|<|>)\s*(-?\d+(?:\.\d+)?)\s*',
        condition
    )

    if not match:
        raise ValueError(f'Invalid condition: {condition}')

    op, value = match.groups()

    return ops[op], float(value)

# %% names

def def_param_names(df, variable, source=None):

    ###-----Error-----###

    if variable in ('RES','THL'): # extend to do full list
        var_err = '_'.join((variable,'mag','unc'))

    else:
        var_err = '_'.join((variable,'unc'))

    if source is not None:
        var_err = '_'.join((var_err,source))

    if var_err not in df:
        var_err = None

    ###-----Count-----###

    if source=='omni':
        if variable in plasma_cols:
            var_count   = 'plasma_counts'
        elif variable in imf_cols:
            var_count   = 'imf_counts'

    elif '_GS' in variable:
        parts  = variable.split('_')
        field  = parts[0]

        var_count = '_'.join((field,'mag','count'))

    elif variable in ('RES','THL'):
        var_count = '_'.join((variable,'mag','count'))

    else:
        var_count = '_'.join((variable,'count'))

    if source is not None:
        var_count = '_'.join((var_count,source))

    if var_count not in df:
        var_count = None

    return var_err, var_count

# %% binwidths

def get_variable_bin_width(series):

    data_range = (series.max()-series.min())

    if data_range > 20:
        bin_width = data_range // 20

    elif data_range > 10:
        bin_width = 1

    else:
        bin_width = 0.5

    return bin_width

# %% limits

def get_variable_range(ind_var, ind_src, dep_var=None, restrict=True, bounds=None, shift_centre=True):

    invert = False

    if ind_var.startswith(('B_z','V_x','S_x')):
        invert = True

    limits = get_var_limits(ind_var, ind_src, restrict, bounds, shift_centre)

    return limits, invert

def get_var_limits(var, region, restrict=True, bounds=None, shift_centre=True):

    shift_centre = shift_centre and var.startswith('B_clock')
    restrict     = restrict and not var.startswith('B_clock')

    if bounds is not None:
        return bounds

    limits = [0, None]

    if var=='B_clock':
        if shift_centre: # Ensures 180 degrees is in the centre
            limits = [0, 2*np.pi]
        else:
            limits = [-np.pi, np.pi]

    elif restrict:

        upper_limits = {'V_flow': 1000, 'V_A': 200, 'P_': 10, 'M_A': 50, 'beta': 30,
                        'B_avg' : 15, 'B_para': 80, 'B_y': 25,
                        'E_'    : defaultdict(lambda: 14, {'sw': 12}),
                        'B_z'   : defaultdict(lambda: -100, {'sw': -40}),
                        'S_x'   : defaultdict(lambda: -300, {'sw': -100}),
                        'N_'    : defaultdict(lambda: 150, {'sw': 25}),
                        'T_'    : defaultdict(lambda: 5, {'sw': 2}), # MK; Up to ~0.2keV in the sw (1eV = 11.604525 MK)
                        }

        upper_limit = next((lim for lab, lim in upper_limits.items() if var.startswith(lab)), None)
        if isinstance(upper_limit, dict):
            upper_limit = upper_limit[region]

        if upper_limit is None:
            warnings.warn(f'No defined limit for {var}.')

        else:
            limits[1] = upper_limit

    if var.startswith(('B_z', 'V_x', 'S_x', 'S_perp')):
        limits = limits[::-1]

    return limits

# %% parameters

def grp_param_splitting(df, grp_var, grp_param, grp_unit, **kwargs):

    grp_split = kwargs.get('grp_split',None)
    quantiles = kwargs.get('quantiles',2)

    used_median = False

    if grp_unit in ('rad','deg','°'):
        z_unit_str = '°'
    elif grp_unit is not None and grp_unit not in ('1','NUM',''):
        z_unit_str = f' {grp_unit}'
    else:
        z_unit_str = ''

    grp_string = data_string(grp_var)
    grp_label = create_label(grp_var, '°' if grp_unit in ('rad','deg','°') else grp_unit)

    median_params = ('Delta B_theta','M_A','P_flow','L1_rho','R_y_GSE_mag')

    if grp_split is not None:
        edges = [grp_split]

    else:
        found_split = False

        if grp_var not in median_params: # Use median for these

            # Want to use defined boundaries
            found_split = True
            if grp_var=='theta_Bn':
                edges = [np.pi/4]

            elif grp_var=='B_clock':
                edges = [0]

            elif grp_var=='Delta B_z':
                edges = [0]

            elif 'E_' in grp_var:
                edges = [0]

            elif 'V_' in grp_var:
                edges = [400]

            elif 'B_' in grp_var:
                edges = [0]

            elif 'beta' in grp_var:
                edges = [1]

            else:
                found_split = False

        if not found_split:
            used_median = True
            if grp_var not in median_params:
                warnings.warn(f'Grouping parameter "{grp_param}" not implemented.')

            median = np.percentile(df[grp_param].dropna().to_numpy(),50)

            if quantiles==4:
                quar_1 = np.percentile(df[grp_param].dropna().to_numpy(),25)
                quar_3 = np.percentile(df[grp_param].dropna().to_numpy(),75)
                edges = [quar_1,median,quar_3]
            elif quantiles==3:
                tert_1 = np.percentile(df[grp_param].dropna().to_numpy(),100/3)
                tert_2 = np.percentile(df[grp_param].dropna().to_numpy(),200/3)
                edges = [tert_1,tert_2]
            else:
                edges = [median]

    if grp_var in ('Delta B_theta','theta_Bn','B_clock'):
        bin_width = np.pi/36

    elif grp_var=='Delta B_z':
        bin_width = 10

    elif grp_var=='R_y_GSE_mag':
        bin_width = 10

    elif grp_var=='M_A':
        bin_width = 1

    elif 'E_' in grp_var:
        bin_width = 0.5

    elif 'V_' in grp_var:
        bin_width = 50

    elif 'B_' in grp_var:
        bin_width = 1

    elif 'beta' in grp_var:
        bin_width = 0.1

    elif 'P_' in grp_var:
        bin_width = 1

    else:
        edges = [median]
        minimum = df[grp_param].dropna().min()
        maximum = df[grp_param].dropna().max()
        bin_width = int(np.log10(maximum-minimum))/20

    if grp_unit in ('rad','deg','°'):

        if len(edges)==1:
            z_labels = [f'${grp_string}$<{np.degrees(edges[0]):.1f}{z_unit_str}',
                        f'${grp_string}$$\\geq${np.degrees(edges[0]):.1f}{z_unit_str}',
                        f'${grp_string}$ = {np.degrees(edges[0]):.1f}{z_unit_str}']
        else:
            z_labels = []
            for edge in edges:
                z_labels.append(f'${grp_string}$={np.degrees(edge):.1f}{z_unit_str}')
    else:

        if len(edges)==1:
            z_labels = [f'${grp_string}$<{edges[0]:.1f}{z_unit_str}',
                        f'${grp_string}$$\\geq${edges[0]:.1f}{z_unit_str}',
                        f'${grp_string}$ = {edges[0]:.1f}{z_unit_str}']
        else:
            z_labels = []
            for edge in edges:
                z_labels.append(f'${grp_string}$={edge:.1f}{z_unit_str}')

    return edges, bin_width, grp_label, z_labels, used_median
