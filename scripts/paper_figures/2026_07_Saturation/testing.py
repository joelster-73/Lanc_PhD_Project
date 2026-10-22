vars # -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""
from src.methods.saturation.plotting.general import plot_driver_response, plot_multi_driver_responses



#### Do sensitive test of changing parameters and looking at fitted parameters

### STILL GETTING core/fromnumeric.py:59: RuntimeWarning: overflow encountered in multiply WARNING


dep_vars    = ('SME', 'SMU', 'THL', 'RES')
ind_vars    = ('E_R', 'E_yz', 'B_z_GSM', 'S_x_GSE')

for dep_var in dep_vars:

    for ind_var in ind_vars:

        for region in ('sw','msh','msh20'):

            try:
                _ = plot_driver_response(ind_var, dep_var, spacecraft='combined', region=region, fit_type='saturation', iterations=2500, condition='<0', cond_col='B_z_GSM')
            except:
                continue


# %% test

### MOVE TO SECTION

from src.methods.saturation.plotting.quality import plot_avg_counts_grid, plot_avg_distributions
from src.methods.saturation.plotting.general import plot_delay_hists

plot_avg_counts_grid('cluster', 'themis', 'mms')



plot_avg_distributions('combined', 'V_x_GSE', 'B_z_GSM', 'E_y_GSM', 'S_x_GSE', restrict=False)
plot_avg_distributions('combined', 'V_x_GSE', 'B_z_GSM', 'E_y_GSM', 'S_x_GSE', restrict=True)
plot_delay_hists('combined', ('msh','sw'))

