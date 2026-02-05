# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 13:43:31 2025

@author: richarj2
"""
import warnings

import numpy as np

from ...plotting.formatting import create_label, data_string

minimum_counts = {'mins': 100, 'counts': 50}

imf_cols = ['B_avg', 'B_x_GSE', 'B_y_GSE', 'B_z_GSE', 'B_y_GSM', 'B_z_GSM', 'B_avg_rms', 'B_vec_rms', 'R_x_BSN', 'R_y_BSN', 'R_z_BSN', 'prop_time_s', 'E_y', 'M_A', 'beta']

plasma_cols = ['P_flow', 'n_p', 'T_p', 'na_np_ratio', 'V_flow', 'V_x_GSE', 'V_y_GSE', 'V_z_GSE', 'R_x_GSE', 'R_y_GSE', 'R_z_GSE', 'E_y', 'M_A', 'M_ms', 'beta', 'E_mag', 'E_x_GSM', 'E_y_GSM', 'E_z_GSM', 'S_mag', 'S_x_GSM', 'S_y_GSM', 'S_z_GSM', 'E_R']

def shift_angular_data(df, *cols):
    # Shift angular data to centre lies at +-180 rather than 0

    df = df.copy()

    for to_shift in cols:
        shift_unit = df.attrs.get('units',{}).get(to_shift,'')
        if shift_unit == 'rad':
            df[to_shift] = (df[to_shift] + 2*np.pi) % (2*np.pi)
        elif shift_unit in ('deg','°'):
            df[to_shift] = (df[to_shift] + 360) % 360

    return df

def mask_df(df, col, limits=None):

        mask = ~df[col].isna()

        if limits:
            if limits[0] is not None:
                mask &= df[col] >= limits[0]
            if limits[-1] is not None:
                mask &= df[col] <= limits[1]

        return df.loc[mask]

def def_param_names(df, variable, source=None):

    if source in ('sw','pc'):
        var_err = None # Need to include
    else:
        var_err   = '_'.join((variable,'unc'))

    if source is not None:
        var_err = '_'.join((var_err,source))

    if var_err not in df:
        var_err = None

    if source in ('sw','pc'):
        if variable in plasma_cols:
            var_count   = 'plasma_counts'
        elif variable in imf_cols:
            var_count   = 'imf_counts'

    elif '_GS' in variable:
        parts = variable.split('_')
        field = parts[0]
        coords = parts[2]

        var_count = '_'.join((field,coords,'count'))

    else:
        var_count = '_'.join((variable,'count'))

    if source is not None:
        var_count = '_'.join((var_count,source))

    if var_count not in df:
        var_count = None

    return var_err, var_count

def get_variable_range(ind_var, ind_src, dep_var=None, restrict=True, bounds=None, shift_centre=True):

    invert = False

    if ind_var.startswith('B_') and not ind_var.startswith(('B_avg', 'B_para','B_y')):
        invert = True

    elif ind_var.startswith('V_') and not ind_var.startswith('V_flow'):
        invert = True

    bin_width = get_var_bin_width(ind_var, restrict)

    limits = get_var_limits(ind_var, ind_src, dep_var, restrict, bounds, shift_centre)

    return bin_width, limits, invert

def get_var_bin_width(var, restrict):

    if var.startswith('AA'):
        bin_width = 20

    elif var.startswith(('PCN','PCC')):
        bin_width = 0.5

    elif var.startswith(('SME','AE','SMC')):
        bin_width = 50

    elif var.startswith('B_'):

        if var.startswith(('B_avg', 'B_para','B_y')):
            bin_width = 1 if restrict else 2
        elif var=='B_clock':
            bin_width = np.pi/18
        else:
            bin_width = 2 if restrict else 5

    elif var.startswith('V_'):

        if var.startswith('V_A'):
            bin_width = 10 if restrict else 20
        else:
            bin_width = 50

    elif var.startswith(('E_','Ey')):
        bin_width = 1 if restrict else 2

    elif var.startswith('N_'):
        bin_width = 5 if restrict else 10

    elif var.startswith('P_'):
        bin_width = 1 if restrict else 5

    elif var.startswith('T_'):
        bin_width = 0.1 if restrict else 0.5

    elif var.startswith('S_'):
        bin_width = 5 if restrict or var.startswith('S_perp') else 10

    elif var.startswith('M_A'):
        bin_width = 5

    elif var.startswith('beta'):
        bin_width = 1

    else:
        raise ValueError(f'"{var} not implemented.')

    return bin_width


def get_var_limits(ind_var, ind_src, dep_var=None, restrict=True, bounds=None, shift_centre=True):

    shift_centre = shift_centre and ind_var.startswith('B_clock')
    restrict     = restrict and not ind_var.startswith('B_clock')

    if bounds is not None:
        return bounds

    limits = [0, None]

    if ind_var.startswith(('B_', 'V_', 'S_')) and not (ind_var.startswith(('B_avg','B_clock','B_parallel','B_y','V_flow','V_A')) or ind_var.endswith('_mag')):

        limits = [None, 0] # Negative quantities

    elif ind_var=='B_clock':
        if shift_centre: # Ensures 180 degrees is in the centre
            limits = [0, 2*np.pi]
        else:
            limits = [-np.pi, np.pi]

    if restrict:

        if ind_var.startswith('AA'):
            limits[1] = 400

        elif ind_var.startswith(('PCN','PCC')):
            limits[1] = 24

        elif ind_var.startswith('SME'):
            limits[1] = 2200

        elif ind_var.startswith('AE'):
            limits[1] = 1800

        elif ind_var.startswith('SMC'):
            limits[1] = 1000

        elif ind_var.startswith('B_'):

            if ind_var.startswith('B_avg'):
                limits[1] = 15

            elif ind_var.startswith('B_para'):
                limits[1] = 80

            elif ind_var.startswith('B_y'):
                limits[1] = 25

            else:
                limits[0] = -40
                if ind_src=='msh':
                    limits[0] = -80
                elif ind_var==dep_var:
                    limits[0] = -20

        elif ind_var.startswith('V_'):

            if ind_var.startswith('V_flow'):
                limits[1] = 1000

            elif ind_var.startswith('V_A'):
                limits[1] = 200

        elif ind_var.startswith('E_'):
            limits[1] = 20
            if ind_src=='msh':
                limits[1] = 20
            elif ind_var.startswith('E_para'):
                limits[1] = 15
            elif (dep_var is not None) and ((ind_var==dep_var) or (dep_var.startswith('E_para'))):
                limits[1] = 10

        elif ind_var.startswith('N_'):
            limits[1] = 50
            if ind_src=='msh':
                limits[1] = 100
            elif ind_var==dep_var:
                limits[1] = 40

        elif ind_var.startswith('P_'):
            limits[1] = 10

        elif ind_var.startswith('T_'): # MK
            # Up to ~0.2keV in the sw (1eV = 11.604525 MK)
            limits[1] = 5
            if ind_src in ('sw','omni'):
                limits[1] = 2

        elif ind_var.startswith('S_'):
            limits[0] = -100
            if ind_var.startswith('S_perp'):
                limits[0] = -150
            if ind_src=='msh':
                limits[0] = -600

        elif ind_var.startswith('M_A'):
            limits[1] = 50

        elif ind_var.startswith('beta'):
            limits[1] = 30

    return limits



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

def get_lagged_columns(df, dep, skip_zero=False):

    dep_vars = {}
    for col in df.columns:
        if col==dep:
            if not skip_zero:
                dep_vars[0] = col

        elif col.startswith(dep) and '_' in col:
            name = '_'.join(col.split('_')[:-1])
            if name==dep:
                lag = col.split('_')[-1][:2]
                dep_vars[int(lag)] = col

    return dep_vars