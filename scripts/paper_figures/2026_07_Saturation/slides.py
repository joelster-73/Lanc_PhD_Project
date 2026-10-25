# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:14:59 2026

@author: richarj2
"""

# %% inventory

from src.methods.saturation.plotting.general import plot_delay_hists
from src.methods.saturation.plotting.space_time import plot_data_inventory

sw_keys  = ('c1','c3','mms1')
msh_keys = ('c1','c3','mms1','tha','thd','the')

plot_data_inventory(*sw_keys, region='sw', display='monthly', print_info=True)
plot_data_inventory(*msh_keys, region='msh', display='monthly', print_info=True)

plot_delay_hists('combined', ('msh','sw'), resolution='5min')



# %% quality

from src.methods.saturation.plotting.general import plot_driver_response
from src.methods.saturation.plotting.uncertainties import plot_independent_uncertainties, table_response_uncertainties

_ = plot_driver_response('E_R', 'AE', spacecraft='combined', region='sw', fit_type='saturation', iterations=0, display='scatter')
_ = plot_driver_response('E_R', 'AE', spacecraft='combined', region='sw', fit_type='saturation', iterations=0)

_ = plot_independent_uncertainties('B_z_GSM', 'E_R', 'E_yz', 'S_x_GSE', regions=('sw','msh'))
_ = table_response_uncertainties(('B_z_GSM', 'E_R', 'E_yz', 'S_x_GSE'), ('SME', 'SMU', 'THL', 'RES'), region='sw')
_ = table_response_uncertainties(('B_z_GSM', 'E_R', 'E_yz', 'S_x_GSE'), ('SME', 'SMU', 'THL', 'RES'), region='msh')


_ = plot_driver_response('E_R', 'AE', spacecraft='combined', region='sw', fit_type='saturation', iterations=2500)
_ = plot_driver_response('E_R', 'AE', spacecraft='combined', region='msh', fit_type='saturation', iterations=2500)



# %% compares

from src.methods.saturation.plotting.general import plot_multi_driver_responses, plot_compare_systems

# SuperMAG
plot_multi_driver_responses(('E_R',), ('AE','SME'), spacecraft='combined', region='sw', fit_type='saturation', iterations=0)
plot_multi_driver_responses(('E_R',), ('AU','SMU'), spacecraft='combined', region='sw', fit_type='saturation', iterations=0)


# Restricting Bz
plot_compare_systems(('E_yz','S_x_GSE'), ('SME',), spacecraft='combined', region='sw', fit_type='saturation', iterations=0, condition='<0', cond_col='B_z_GSM')


# %% results

from src.methods.saturation.plotting.general import plot_compare_regions

# retrieve confidence intervals

plot_compare_regions(('E_yz',), ('SME',), spacecraft='combined', regions=('sw','msh'), fit_type='saturation', iterations=2500, condition='<0', cond_col='B_z_GSM', second_x=True)

plot_compare_regions(('S_x_GSE',), ('SME',), spacecraft='combined', regions=('sw','msh'), fit_type='saturation', iterations=2500, condition='<0', cond_col='B_z_GSM', second_x=True)
