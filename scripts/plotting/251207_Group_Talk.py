# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 10:52:25 2025

@author: richarj2
"""


# %% Import
import os

from src.config import SW_DIR, MSH_DIR, OMNI_DIR
from src.processing.reading import import_processed_data
from src.methods.magnetosheath_saturation.plotting import plot_driver_multi_responses, plot_driver_response, plot_saturation_overview, plot_different_lags
from src.methods.magnetosheath_saturation.plot_space_time import plot_sc_years

sample_interval = '1min'
data_pop = 'with_plasma'

if sample_interval=='1min':
    data_type = 'mins'
else:
    data_type = 'counts'

omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_omni  = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

sw_dir   = os.path.join(SW_DIR, data_pop, sample_interval)
df_sw    = import_processed_data(sw_dir, 'sw_times_combined.cdf')

msh_dir  = os.path.join(MSH_DIR, data_pop, sample_interval)


# %% Recreating_Saturation
param_names  = {'E_y_GSM': 'E_y',
                'V_flow' : 'V',
                'B_z_GSM': 'B_z',
                'N_tot'  : 'N'}

param = 'E_R'

plot_driver_multi_responses(df_omni, None, param, 'PCC_17m', 'PCN_17m', 'PCN', restrict=True, data_type=data_type, data1_name=param_names.get(param,param), data_name_map=param_names)

param = 'E_y_GSM'

plot_driver_multi_responses(df_omni, None, param, 'SME_53m','AE_53m','AE', restrict=True, data_type=data_type, data1_name=param_names.get(param,param), data_name_map=param_names)

# %% Data_Distribution

sw_keys  = ('c1','m1','thb')
msh_keys = ('c1','m1','the')

plot_sc_years(sw_dir, sc_keys=('c1',), combined=False, data_type=data_type, region='sw', year_range=(2000,2026))
plot_sc_years(sw_dir, sc_keys=sw_keys, combined=True, data_type=data_type, region='sw')

plot_sc_years(msh_dir, sc_keys=('c1',), combined=False, data_type=data_type, region='msh', year_range=(2000,2026))
plot_sc_years(msh_dir, sc_keys=msh_keys, combined=True, data_type=data_type, region='msh')

# %% Saturation_in_situ

param_names  = {'E_y_GSM': 'E_y',
                'V_flow' : 'V',
                'B_z_GSM': 'B_z',
                'N_tot'  : 'N'}

param = 'E_R'

plot_driver_multi_responses(df_omni, df_sw, param, 'PCC_17m', 'PCN_17m', 'PCN', restrict=True, data_type=data_type, data1_name=param_names.get(param,param), data_name_map=param_names)

param = 'E_y_GSM'

plot_driver_multi_responses(df_omni, df_sw, param, 'SME_53m','AE_53m','AE', restrict=True, data_type=data_type, data1_name=param_names.get(param,param), data_name_map=param_names)

# %% Lags

plot_different_lags(df_omni, param, dep_var='PCC', data_type=data_type)

# %% Saturation_Cause

sample_interval = '5min'
data_pop = 'with_plasma'

if sample_interval=='1min':
    data_type = 'mins'
else:
    data_type = 'counts'

omni_dir = os.path.join(OMNI_DIR, 'with_lag')
df_omni  = import_processed_data(omni_dir, f'omni_{sample_interval}.cdf')

msh_dir  = os.path.join(MSH_DIR, data_pop, sample_interval)
df_msh   = import_processed_data(msh_dir, 'msh_times_combined.cdf')

param_names  = {'E_y_GSM': 'E_y',
                'V_flow' : 'V',
                'B_z_GSM': 'B_z',
                'N_tot'  : 'N',
                'SME_53m': 'SME',
                'na_np_ratio': 'R_alpha_p'}

param    = 'E_y_GSM'
response = 'SME_53m'

plot_driver_response(df_omni, df_msh, param, response, restrict=True, data1_name=param_names.get(param,param), data2_name=param_names.get(response,response))

split = 'na_np_ratio'
plot_saturation_overview(df_omni, df_msh, param, response, split, grp_src='sw', plot_test=True, max_expansions=0, data1_name=param_names.get(param,param), data2_name=param_names.get(response,response), data3_name=param_names.get(split,None), bounds=[0,10])
