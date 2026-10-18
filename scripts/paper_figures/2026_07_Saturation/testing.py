vars # -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""
from src.methods.saturation.plotting.general import plot_driver_response, plot_multi_driver_responses


dep_vars    = ('SME', 'SMU', 'THL', 'RES')
ind_vars    = ('E_R', 'E_y_GSM', 'B_z_GSM', 'S_x_GSE')

#### Condition should be implemented, e.g. Bz<0 for the Sx, so plot for meeting with Jim
#### Find when odd r_avg columns added
#### Change plot functions so errors can be used in plotting scatter and used in fits separately

for dep_var in dep_vars:

    for ind_var in ind_vars:

        for region in ('sw','msh','msh20'):

            try:
                _ = plot_driver_response(ind_var, dep_var, spacecraft='combined', region=region, fit_type='saturation', iteration=500)
            except:
                continue


dep_vars = ('SME', 'AE', 'SMU', 'AU')
dep_vars = ('THL_y_GSM', 'RES_y_GSM'); ind_vars = ('E_R','E_y_GSM')

plot_multi_driver_responses(ind_vars, dep_vars, spacecraft='combined', region='sw', fit_type='saturation', iteration=500)
plot_multi_driver_responses(ind_vars, dep_vars, spacecraft='combined', region='msh', fit_type='saturation', iteration=500)

_ = plot_driver_response('S_x_GSE', 'SME', spacecraft='combined', region='msh', fit_type='saturation', iteration=500)

