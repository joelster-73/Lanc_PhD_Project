vars # -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""
from src.methods.saturation.plotting.general import plot_driver_response

dep_vars    = ('SME', 'SMU', 'THL', 'RES')
ind_vars    = ('E_R', 'E_yz', 'B_z_GSM', 'S_x_GSE')

for dep_var in dep_vars:

    for ind_var in ind_vars:

        for region in ('sw','msh','msh20'):

            try:
                _ = plot_driver_response(ind_var, dep_var, spacecraft='combined', region=region, fit_type='saturation', iterations=2500, condition='<0', cond_col='B_z_GSM')
            except:
                continue




# %%

from src.processing.reading import import_processed_data

df = import_processed_data('msh', dtype='plasma', resolution='5min', file_name='msh_times_combined')

df.plot('r_F', 'V_x_GSE', kind='scatter')
df.plot('r_cone', 'V_x_GSE', kind='scatter')
df['V_x_GSE'].plot()


df_large = df.loc[df['V_x_GSE']<-600]
