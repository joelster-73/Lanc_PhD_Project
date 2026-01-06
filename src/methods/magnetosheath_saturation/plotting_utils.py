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
        field, _, coords = variable.split('_')
        var_count = '_'.join((field,coords,'count'))

    else:
        var_count = '_'.join((variable,'count'))

    if source is not None:
        var_count = '_'.join((var_count,source))

    if var_count not in df:
        var_count = None

    return var_err, var_count

def ind_variable_range(ind_var, ind_src, dep_var=None, restrict=True, bounds=None, shift_centre=True):

    limits = [None, None]
    invert = False

    if ind_var.startswith('AA'):
        bin_width, limits[0] = 20, 0
        if restrict:
            limits[1] = 400

    elif ind_var.startswith(('PCN','PCC')):
        bin_width, limits[0] = 0.5, 0
        if restrict:
            limits[1] = 20

    elif ind_var.startswith('SME'):
        bin_width, limits[0] = 50, 0
        if restrict:
            limits[1] = 2200

    elif ind_var.startswith('AE'):
        bin_width, limits[0] = 50, 0
        if restrict:
            limits[1] = 1800

    elif ind_var=='B_avg':
        bin_width, limits[0] = 2, 0
        if restrict:
            bin_width, limits[1] = 1, 15

    elif ind_var=='B_parallel':
        bin_width, limits[0] = 2, 0
        if restrict:
            bin_width, limits[1] = 1, 80

    elif ind_var=='B_clock':
        invert = True
        # Ensures 180 degrees is in the centre
        bin_width = np.pi/18
        if shift_centre:
            limits[0], limits[1] = 0, 2*np.pi
        else:
            limits[0], limits[1] = -np.pi, np.pi

    elif 'B_y' in ind_var:
        bin_width, limits[0] = 2, 0
        if restrict:
            bin_width, limits[1] = 1, 25

    elif 'B_' in ind_var:
        invert = True
        bin_width, limits[1] = 5, 0
        if restrict:
            bin_width = 4
            if ind_src=='msh':
                limits[0] = -80
            elif ind_var==dep_var:
                bin_width, limits[0] = 2, -20
            else:
                limits[0] = -40

    elif ind_var=='V_flow':
        bin_width, limits[0] = 50, 200
        if restrict:
            limits[1] = 1000
            if ind_src=='msh':
                limits[1] = 750

    elif ind_var=='V_A':
        bin_width, limits[0] = 20, 0
        if restrict:
            bin_width, limits[1] = 10, 200

    elif 'V_' in ind_var:
        invert = True
        bin_width, limits[1] = 50, 0

    elif ind_var=='E_parallel':
        bin_width, limits[0] = 2, 0
        if restrict:
            bin_width = 1
            limits[1] = 14

    elif 'E_' in ind_var:
        bin_width, limits[0] = 2, 0
        if restrict:
            bin_width = 1
            if ind_src=='msh':
                limits[1] = 20
            elif (dep_var is not None) and (
                    (ind_var==dep_var) or (dep_var.startswith('E_parallel'))):
                bin_width, limits[1] = 0.5, 10
            elif ind_var.startswith('E_parallel'):
                bin_width, limits[1] = 0.5, 15
            else:
                limits[1] = 20

    elif 'N_' in ind_var:
        bin_width, limits[0] = 10, 0
        if restrict:
            bin_width = 5
            if ind_src=='msh':
                limits[1] = 100
            elif ind_var==dep_var:
                limits[1] = 40
            else:
                limits[1] = 50

    elif 'P_' in ind_var:
        bin_width, limits[0] = 5, 0
        if restrict:
            bin_width = 1
            if ind_src=='msh':
                limits[1] = 10
            elif ind_var==dep_var:
                limits[1] = 10
            else:
                limits[1] = 10

    elif 'T_' in ind_var:
        # Up to ~1keV in the sw (11.604525 MK)
        bin_width, limits[0] = 1e6, 0
        if restrict:
            bin_width = 5e5
            if ind_src=='sw':
                limits[1] = 20e6
            else:
                limits[1] = 75e6

    elif ind_var=='S_perp':
        invert = True
        bin_width, limits[1] = 5, 0
        if restrict:
            limits[0] = -150

    elif 'S_' in ind_var:
        bin_width, limits[1] = 10, 0
        if restrict:
            bin_width, limits[0] = 5, -100
            if ind_src=='msh':
                bin_width, limits[0] = 20, -600

    elif 'M_A' in ind_var:
        bin_width, limits[0] = 5, 0
        if restrict:
            bin_width, limits[1] = 5, 50

    elif 'beta' in ind_var:
        bin_width, limits[0] = 1, 0
        if restrict:
            limits[1] = 30

    else:
        raise ValueError(f'"{ind_var} not implemented.')

    if bounds is not None:
        limits = bounds

    return bin_width, limits, invert

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