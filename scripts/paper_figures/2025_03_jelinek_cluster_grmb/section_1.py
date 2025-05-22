# %% Imports
from handling_files import import_processed_data

from config import *

from datetime import datetime


# Bad days to be removed from data analysis of solar wind

# Read the file and convert to a list of tuples
with open(bad_crossings, 'r') as file:
    time_windows_strings = eval(file.read())

bad_GRMB = [
    (datetime.strptime(start, "%Y-%m-%d %H:%M:%S"), datetime.strptime(end, "%Y-%m-%d %H:%M:%S"), label)
    for start, end, label in time_windows_strings
]
bad_GRMB.sort(key=lambda window: window[0])


# Define the date range to analyse data
c1_range = (datetime(2001, 1, 1), datetime(2024, 1, 1))

buffer_R = 4

# %% Importing
from data_processing import filter_sw, merge_dataframes, insert_bs_diff
from handling_crossings import assign_region
from coordinates import insert_car_coords, insert_cyl_coords

# Import Cluster
cluster1 = import_processed_data(proc_clus_dir)
insert_car_coords(cluster1, field='B', coords='GSM', r_col='B_mag', th_col='B_pitch', ph_col='B_clock')
insert_cyl_coords(cluster1, field='r', coords='GSE')

# Imports ESA Crossings
crossings = import_processed_data(proc_cros_dir)
cross_labels = crossings.attrs['crossings']

# Import Jelinek filtering data
c1_bs_distances = import_processed_data(proc_scbs_dir, date_range=c1_range)
insert_bs_diff(cluster1, c1_bs_distances, 'C1')

assign_region(cluster1, crossings, bad_days=bad_GRMB)


# Import OMNI
omni = import_processed_data(proc_omni_dir,
                             date_range=c1_range)
insert_cyl_coords(omni, field='r', coords='GSE')

# Merging and filtering

# GRMB Database
c1_sw_grmb = filter_sw(cluster1, 'GRMB', regions=(12,))
c1_omni_grmb = merge_dataframes(c1_sw_grmb, omni, 'C1', 'OMNI') # C1 vs OMNI using GRMB

# Jelinek Model
c1_sw_jel = filter_sw(cluster1, method='model', sc_key='C1', buffer=buffer_R)
c1_omni_jel = merge_dataframes(c1_sw_jel, omni, 'C1', 'OMNI') # C1 vs OMNI using Jelinek

# Combined
c1_sw_com = filter_sw(cluster1, method='combined', regions=(12,), sc_key='C1', buffer=buffer_R)
c1_omni_com = merge_dataframes(c1_sw_com, omni, 'C1', 'OMNI')

c1_omni = merge_dataframes(cluster1, omni, 'C1', 'OMNI') # C1 vs OMNI

file_path = 'timestamps.txt'
c1_sw_com.index.strftime("%Y-%m-%d %H:%M:%S").to_series().to_csv(file_path, index=False, header=False)


# %% Orbit
from data_plotting import plot_orbit

plot_orbit(cluster1, plane='x-rho', coords='GSE', models='Typical',
           display='Heat', bin_width=0.1, brief_title='Cluster\'s orbit from 2001 to 2023')

# %% Best_Buffer
from data_plotting import plot_best_buffer

plot_best_buffer(c1_omni, 'r_bs_diff_C1', 'B_avg_C1', 'B_avg_OMNI', scale='lin',
                 compressions=compressions_dir, buff_max=6, data_name='Jelínek BS')

# %% High_activity
from data_plotting import plot_compare_datasets_with_activity

plot_compare_datasets_with_activity(c1_sw_jel, c1_sw_grmb)


# %%

from data_plotting import plot_method_sketch

plot_method_sketch()
