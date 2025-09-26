# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""

from src.config import LUNA_CLUS_DIR_SPIN, PROC_CLUS_DIR_SPIN, LUNA_CLUS_DIR_HIAM, LUNA_CLUS_DIR_HIAQ, PROC_CLUS_DIR_5VPS, LUNA_CLUS_DIR_5VPS, PROC_CLUS_DIR_FGM, PROC_CLUS_DIR_MSHS

from src.processing.writing import resample_cdf_files

from src.processing.cluster.config import CLUSTER_VARIABLES_SPIN, CLUSTER_VARIABLES_HIA, CLUSTER_VARIABLES_HIA_QUALITY, CLUSTER_VARIABLES_5VPS

from src.processing.cluster.handling import process_cluster_files, combine_spin_data, filter_spin_data


# %%  Process_5VPS

process_cluster_files(LUNA_CLUS_DIR_5VPS, PROC_CLUS_DIR_5VPS, CLUSTER_VARIABLES_5VPS)

#process_cluster_files(LUNA_CLUS_DIR_5VPS, PROC_CLUS_DIR_5VPS, CLUSTER_VARIABLES_5VPS, sub_folders=True, year=2001)

# %% Process_spin_fgm

process_cluster_files(LUNA_CLUS_DIR_SPIN, PROC_CLUS_DIR_SPIN, CLUSTER_VARIABLES_SPIN, sample_interval='None')

# %% Process_spin_plasma

process_cluster_files(LUNA_CLUS_DIR_HIAM, PROC_CLUS_DIR_SPIN, CLUSTER_VARIABLES_HIA, sub_folders=True, sample_interval='None', quality_directory=LUNA_CLUS_DIR_HIAQ, quality_variables=CLUSTER_VARIABLES_HIA_QUALITY)

# %% Combine_spin
combine_spin_data(PROC_CLUS_DIR_SPIN, PROC_CLUS_DIR_FGM)

# %% MSH_filter

filter_spin_data(PROC_CLUS_DIR_SPIN, region='msh')

# %% Average_1min

resample_cdf_files(PROC_CLUS_DIR_MSHS, sample_interval='1min')

# %% Average_5min

resample_cdf_files(PROC_CLUS_DIR_MSHS, sample_interval='5min')

# %% Average_5min_field

resample_cdf_files(PROC_CLUS_DIR_FGM, sample_interval='5min', yearly_files=True)


# %%
import os
import pandas as pd
import numpy as np

from spacepy import pycdf
from src.processing.handling import get_processed_files
from src.processing.reading import import_processed_data

from src.config import PROC_OMNI_DIR_5MIN

from scipy.constants import mu_0, m_p
from scipy.constants import physical_constants
m_a = physical_constants['alpha particle mass'][0]

omni_df = import_processed_data(PROC_OMNI_DIR_5MIN)

omni_times = omni_df.index.to_numpy()
omni_df.loc[omni_df['na_np_ratio']>1,'na_np_ratio'] = np.nan
omni_values = omni_df['na_np_ratio'].to_numpy()


directory = os.path.join(PROC_CLUS_DIR_SPIN,'combined','raw')
cdf_files = get_processed_files(directory)
for file in cdf_files:
    with pycdf.CDF(file) as cdf:

        #cdf.readonly(False)

        times = cdf['epoch'][...]
        density = cdf['N_ion'][...]
        velocity = cdf['V_mag'][...]
        fields = cdf['B_avg'][...]


        times = pd.to_datetime(times).to_numpy()
        idx = np.searchsorted(omni_times, times, side='right') - 1

        idx[idx < 0] = 0

        ratio_values = omni_values[idx]
        m_avg = (m_p + ratio_values * m_a) / (ratio_values + 1)

        pressure = 0.5 * m_avg * density * velocity**2 * 1e21

        betas = pressure / (fields**2) * (2*mu_0) * 1e9

        cdf['beta'][...] = betas

        if 'P_flow' not in cdf:
            cdf.new('P_flow', data=pressure)
        else:
            cdf['P_flow'][...] = pressure
        cdf['P_flow'].attrs['units'] = 'nPa'

        if 'na_np_ratio' not in cdf:
            cdf.new('na_np_ratio', data=ratio_values)
        else:
            cdf['na_np_ratio'][...] = ratio_values
        cdf['na_np_ratio'].attrs['units'] = '1'

    print('Done:',file)

