# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:21:52 2025

@author: richarj2
"""

import matplotlib.pyplot as plt


dark_mode = True
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
    plt.style.use('dark_background')

    #### DARK BACKGROUND MAKES CHANGING FONT COLOUR TO BLACK/WHITE OBSOLETE ####


plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 400
plt.rcParams['font.size'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fontsize'] = 12
# plt.rcParams['legend.facecolor'] = 'w'
# plt.rcParams['legend.edgecolor'] = 'k'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 15

if large_text:
    plt.rcParams['font.size'] = 15
    plt.rcParams['figure.titlesize'] = 22
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18

scatter_markers = [ 's', 'o', 'v', '1', '2', '3', '4', '^', '<', '>', '.', ',', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']

colour_dict = {
    'OMNI': 'orange',
    'C1':   'blue',
    'C2':   'cornflowerblue',
    'C3':   'lightskyblue',
    'C4':   'lightblue',
    'TH':   'green',
    'THA':  'forestgreen',
    'THB':  'seagreen',
    'THC':  'mediumseagreen',
    'THD':  'lightgreen',
    'THE':  'palegreen',
    'ACE':  'darkviolet',
    'DSC':  'deeppink',
    'GEO':  'cyan',
    'IMP8': 'crimson',
    'WIND': 'magenta',
    'M1': 'red',
    'M2': 'tomato',
    'M3': 'lightsalmon',
    'M4': 'mistyrose'
}

database_colour_dict = {'CFA': 'b', 'Donki': 'r', 'Helsink': 'g'}
