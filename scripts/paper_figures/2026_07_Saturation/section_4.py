# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:22:14 2026

@author: richarj2
"""

# %% uncertainties

from src.methods.saturation.plotting.uncertainties import table_response_uncertainties

# Include as one combined table, think about how to combine into text file such that it can be loaded and formatted in overleaf

_ = table_response_uncertainties(('B_z_GSM', 'E_R', 'E_yz', 'S_x_GSE'), ('SME', 'SMU', 'THL', 'RES'), region='sw')
_ = table_response_uncertainties(('B_z_GSM', 'E_R', 'E_yz', 'S_x_GSE'), ('SME', 'SMU', 'THL', 'RES'), region='msh')

