# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""

from src.methods.saturation.plotting import plot_driver_multi_responses

param_names  = {'E_y_GSM': 'E_y',
                'V_flow' : 'V',
                'B_z_GSM': 'B_z',
                'N_tot'  : 'N'}

responses = ('THL','SME','SMR')
param     = 'E_R'

for bottom_axis in ('heat','scatter','hist'):
    plot_driver_multi_responses(param, *responses, lags=(17,53,60), restrict=True, bottom_axis=bottom_axis, data1_name=param_names.get(param,param), data_name_map=param_names)


# %% Columns
from src.processing.reading import import_processed_data, import_processed_spacecraft
from src.methods.saturation.sc_delay_time import calc_bs_sc_delay

df_sc = import_processed_data('sw', dtype='plasma', resolution='5min', file_name='sw_times_combined')


#calc_bs_sc_delay(df, omni_key='sw', sc_key='sc', region='sw')

test = import_processed_spacecraft('c1', populations=['fgm'],year=2001)
