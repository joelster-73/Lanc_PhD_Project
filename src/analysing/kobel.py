# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 13:18:45 2025

@author: richarj2
"""
import numpy as np

def load_compression_ratios(compressions):

    data = np.load(compressions)
    B_imf = data['B_imf']
    B_msh = data['B_msh']
    B_rho = data['B_rho']

    return B_imf, B_msh, B_rho

def are_points_above_line(x_line, y_line, x_points, y_points):

    y_line_interp = np.interp(x_points, x_line, y_line)
    return y_points > y_line_interp



