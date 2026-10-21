vars # -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""
from src.methods.saturation.plotting.general import plot_driver_response, plot_multi_driver_responses



#### Change plot functions so errors can be used in plotting scatter and used in fits separately
#### Do sensitive test of changing parameters and looking at fitted parameters

### STILL GETTING core/fromnumeric.py:59: RuntimeWarning: overflow encountered in multiply WARNING

dep_vars    = ('SME', 'SMU', 'THL', 'RES')
ind_vars    = ('E_R', 'E_y_GSM', 'B_z_GSM', 'S_x_GSE')

for dep_var in dep_vars:

    for ind_var in ind_vars:

        for region in ('sw','msh','msh20'):

            try:
                _ = plot_driver_response(ind_var, dep_var, spacecraft='combined', region=region, fit_type='saturation', iteration=500)
            except:
                continue

for region in ('sw','msh'):

    ind_vars = ('E_R','E_y_GSM')
    dep_vars = ('SME', 'AE', 'SMU', 'AU')

    plot_multi_driver_responses(ind_vars, dep_vars, spacecraft='combined', region=region, fit_type='saturation', iteration=500)


for region in ('sw','msh'):

    ind_vars = ('E_y_GSM', 'S_x_GSE')
    dep_vars = ('SME', 'SMU', 'RES')

    plot_multi_driver_responses(ind_vars, dep_vars, spacecraft='combined', region=region, fit_type='saturation', iteration=500)
    plot_multi_driver_responses(ind_vars, dep_vars, spacecraft='combined', region=region, fit_type='saturation', iteration=500, condition='<0', cond_col='B_z_GSM')


_ = plot_driver_response('E_y_GSM', 'SMU', spacecraft='combined', region='sw', fit_type='saturation', iteration=500, second_x=True)
_ = plot_driver_response('S_x_GSE', 'SMU', spacecraft='combined', region='sw', fit_type='saturation', iteration=500, second_x=True)


# %%

from src.methods.saturation.plotting.quality import plot_avg_counts_grid, plot_avg_distributions
from src.methods.saturation.plotting.general import plot_delay_hists

plot_avg_counts_grid('cluster', 'themis', 'mms')



plot_avg_distributions('combined', 'V_x_GSE', 'B_z_GSM', 'E_y_GSM', 'S_x_GSE', restrict=False)
plot_avg_distributions('combined', 'V_x_GSE', 'B_z_GSM', 'E_y_GSM', 'S_x_GSE', restrict=True)
plot_delay_hists('combined', 'sw')
plot_delay_hists('combined', 'msh')

