# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 09:34:24 2025

@author: richarj2
"""

from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data
import pandas as pd

# %% Computing

helsinki_with_normals = import_processed_data(PROC_SHOCKS_DIR, file_name='helsinki_shocks.cdf')
helsinki_with_normals = helsinki_with_normals.loc[:,['spacecraft', 'r_x_GSE', 'r_y_GSE', 'r_z_GSE', 'res_B', 'res_p', 'N_x_GSE', 'N_x_GSE_unc', 'N_y_GSE', 'N_y_GSE_unc', 'N_z_GSE', 'N_z_GSE_unc','v_sh', 'v_sh_unc']]
helsinki_with_normals.rename(columns={f'N_{comp}_GSE':f'N{comp}' for comp in ('x','y','z')},inplace=True)
helsinki_with_normals.rename(columns={f'N_{comp}_GSE_unc':f'N{comp}_unc' for comp in ('x','y','z')},inplace=True)
helsinki_with_normals['time_unc'] = helsinki_with_normals.apply(lambda row: 0.5*max(row['res_B'], row['res_p']), axis=1)
helsinki_with_normals.drop(columns=['res_B','res_p'],inplace=True)
helsinki_with_normals['source'] = 'H'

cfa_shocks = import_processed_data(PROC_SHOCKS_DIR, file_name='cfa_shocks.cdf')
cfa_shocks = cfa_shocks.loc[:,['time_s_unc', 'spacecraft', 'r_x_GSE', 'r_y_GSE', 'r_z_GSE', 'Nx', 'Nx_unc', 'Ny', 'Ny_unc', 'Nz', 'Nz_unc', 'v_sh', 'v_sh_unc']]
cfa_shocks.rename(columns={'time_s_unc':'time_unc'},inplace=True)
cfa_shocks['spacecraft'] = cfa_shocks['spacecraft'].str.upper()
cfa_shocks = cfa_shocks.loc[
    ~(cfa_shocks['spacecraft'] == cfa_shocks['spacecraft'].shift()) |
    (cfa_shocks.index.to_series().diff() > pd.Timedelta(minutes=5))
]
cfa_shocks['source'] = 'C'

shocks_with_normals = pd.concat([helsinki_with_normals,cfa_shocks]).sort_index()

# %% Writing
from src.processing.utils import add_unit

new_attrs = helsinki_with_normals.attrs.copy()
for col in list(helsinki_with_normals.attrs['units']):
    if col not in shocks_with_normals:
        del new_attrs['units'][col]
for col in shocks_with_normals:
    if col not in new_attrs:
        new_attrs['units'][col] = add_unit(col)

new_attrs['units']['eventNum'] = 'STRING'
new_attrs['units']['source'] = 'STRING'
new_attrs['units']['time_unc'] = 's'

shocks_with_normals.attrs = new_attrs

# %%
from src.processing.writing import write_to_cdf
from src.config import PROC_SHOCKS_DIR
import os

output_file = os.path.join(PROC_SHOCKS_DIR, 'shocks_with_normals.cdf')

write_to_cdf(shocks_with_normals, output_file, new_attrs, reset_index=True)