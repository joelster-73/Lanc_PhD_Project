vars # -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""
from src.methods.saturation.plotting.general import plot_driver_response, plot_multi_driver_responses


dep_vars    = ('SME', 'SMU', 'THL', 'RES')
ind_vars    = ('E_R', 'E_y_GSM', 'B_z_GSM', 'S_x_GSM')

#### fix delay time problem, i.e., very large
#### check S_x_GSM figure when restricting Bz
#### Weird bug that S_x_GSM and S_x_GSE_unc are the column names
#### also check code to add in THL_y and RES_y
#### Look at adding floor to uncertainty
#### create restricted msh dataset

for dep_var in dep_vars:

    for ind_var in ind_vars:

        for region in ('sw','msh'):

            try:
                _ = plot_driver_response(ind_var, dep_var, spacecraft='combined', region=region, fit_type='saturation', iteration=1000)
            except:
                continue


dep_vars = ('SME', 'AE', 'SMU', 'AU')

plot_multi_driver_responses(ind_vars, dep_vars, spacecraft='combined', region='sw', fit_type='saturation', iteration=1000)
plot_multi_driver_responses(ind_vars, dep_vars, spacecraft='combined', region='msh', fit_type='saturation', iteration=1000)

