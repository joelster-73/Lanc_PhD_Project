# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


from src.methods.saturation.plotting.general import plot_driver_response


df = plot_driver_response('E_R', 'SME', lag=53, spacecraft='combined', fit_type='saturation')
