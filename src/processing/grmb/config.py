# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:30:33 2025

@author: richarj2
"""

crossing_variables = {'epoch': 'time_tags__C1_CT_AUX_GRMB',
                      'loc_num': 'location_code__C1_CT_AUX_GRMB',
                      'loc_name': 'location_label__C1_CT_AUX_GRMB',
                      'quality_num': 'quality_location_code__C1_CT_AUX_GRMB',
                      'complexity_num': 'crossing_complexity__C1_CT_AUX_GRMB'}

GRMB_qualities = {0: 'Bad',
                  1: 'Low',
                  2: 'Fair',
                  3: 'Good',
                  4: 'Top'}

GRMB_complexities = {0: 'simple',
                     1: 'complex'}



# Bad days to be removed from data analysis of solar wind

# Read the file and convert to a list of tuples
with open(bad_crossings, 'r') as file:
    time_windows_strings = eval(file.read())

bad_GRMB = [
    (datetime.strptime(start, "%Y-%m-%d %H:%M:%S"), datetime.strptime(end, "%Y-%m-%d %H:%M:%S"), label)
    for start, end, label in time_windows_strings
]
bad_GRMB.sort(key=lambda window: window[0])