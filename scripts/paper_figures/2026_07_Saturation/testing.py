# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""
from src.methods.saturation.plotting.general import plot_driver_response


# process AU/AL or if they're in files adapt code

for dep_var, lag in zip(('SME', 'AE', 'THL', 'RES'), (53, 53, 17, 17)):

    for ind_var in ('E_R', 'E_y_GSM', 'B_z_GSM', 'S_x_GSE'):

        for region in ('sw','msh'):
            try:
                _ = plot_driver_response(ind_var, dep_var, lag=lag, spacecraft='combined', region=region, fit_type='saturation')
            except:
                continue
            break
        break
    break


