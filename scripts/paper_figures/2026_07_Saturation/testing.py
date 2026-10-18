vars # -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""
from src.methods.saturation.plotting.general import plot_driver_response, plot_multi_driver_responses


dep_vars    = ('SME', 'SMU', 'THL', 'RES')
ind_vars    = ('E_R', 'E_y_GSM', 'B_z_GSM', 'S_x_GSM')

#### implement so can specify another condition, e.g. Bz<0 for the Sx
#### Weird bug that S_x_GSM and S_x_GSE_unc are the column names
#### also check code to add in THL_y and RES_y
#### Find when odd r_avg columns added
#### Change plot functions so errors can be used in plotting scatter and used in fits separately

for dep_var in dep_vars:

    for ind_var in ind_vars:

        for region in ('sw','msh','msh20'):

            try:
                _ = plot_driver_response(ind_var, dep_var, spacecraft='combined', region=region, fit_type='saturation', iteration=500)
            except:
                continue


dep_vars = ('SME', 'AE', 'SMU', 'AU')

plot_multi_driver_responses(ind_vars, dep_vars, spacecraft='combined', region='sw', fit_type='saturation', iteration=500)
plot_multi_driver_responses(ind_vars, dep_vars, spacecraft='combined', region='msh', fit_type='saturation', iteration=500)

_ = plot_driver_response('S_x_GSM', 'SME', spacecraft='combined', region='msh', fit_type='saturation', iteration=500)



# %%

from src.processing.reading import import_processed_spacecraft
from src.config import get_data_populations



sc = 'c1'
data_pop = 'plasma'
resolution = '5min'
region = 'sw'
populations = get_data_populations(sc, data_pop, region)

df_sc = import_processed_spacecraft(sc, populations, 'spin', year=2003)





# %% renaming

from src.processing.writing import rename_cdf_variables
from src.config import get_proc_directory

rename_map = {
    'S_x_GSM': 'S_x_GSE',
    'E_x_GSM': 'E_x_GSE',
}

for resolution in ('1min','5min','15min'):

    direc = get_proc_directory('msh20', 'plasma', resolution)

    rename_cdf_variables(direc, rename_map, dry_run=False)