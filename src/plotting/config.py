# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:21:52 2025

@author: richarj2
"""

import matplotlib.pyplot as plt


dark_mode = False
large_text = False
save_fig = True


white = 'w'
black = 'k'
blue = 'b'
green = 'g'
pink = 'deeppink'
if dark_mode:
    blue = 'c'
    white = 'k'
    black = 'w'
    green = 'lime'


plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['legend.facecolor'] = 'w'
plt.rcParams['legend.edgecolor'] = 'k'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 15

if large_text:
    plt.rcParams['font.size'] = 15
    plt.rcParams['figure.titlesize'] = 22
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18

