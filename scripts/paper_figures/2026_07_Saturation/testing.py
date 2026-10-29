vars # -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""
from src.methods.saturation.plotting.general import plot_driver_response

dep_vars    = ('SME', 'SMU', 'THL', 'RES')
ind_vars    = ('E_R', 'E_yz', 'B_z_GSM', 'S_x_GSE')

for dep_var in dep_vars:

    for ind_var in ind_vars:

        for region in ('sw','msh','msh20'):

            try:
                _ = plot_driver_response(ind_var, dep_var, spacecraft='combined', region=region, fit_type='saturation', iterations=2500, condition='<0', cond_col='B_z_GSM')
            except:
                continue




