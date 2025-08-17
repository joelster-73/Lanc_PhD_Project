# Section_5_Comparing
from datetime import datetime

from src.config import PROC_CLUS_DIR_1MIN as PROC_CLUS_DIR, PROC_OMNI_DIR, PROC_SCBS_DIR, BAD_CROSSINGS, CROSSINGS_DIR, C1_RANGE
from src.coordinates.spatial import insert_car_coords, insert_cyl_coords
from src.methods.bowshock_filtering.jelinek import insert_bs_diff
from src.methods.bowshock_filtering.grison import assign_region

from src.processing.dataframes import merge_dataframes
from src.processing.filtering import filter_sw
from src.processing.reading import import_processed_data

# %% Data

# Cluster
cluster1 = import_processed_data(PROC_CLUS_DIR)
insert_car_coords(cluster1, field='B', coords='GSM', r_col='B_mag', th_col='B_pitch', ph_col='B_clock')
insert_cyl_coords(cluster1, field='r', coords='GSE')

c1_bs_distances = import_processed_data(PROC_SCBS_DIR)
insert_bs_diff(cluster1, c1_bs_distances, 'C1')

# OMNI
omni = import_processed_data(PROC_OMNI_DIR, date_range=C1_RANGE)
insert_cyl_coords(omni, field='r', coords='GSE')

# GRMB Crossings
crossings = import_processed_data(CROSSINGS_DIR)
cross_labels = crossings.attrs['crossings']

with open(BAD_CROSSINGS, 'r') as file:
    time_windows_strings = eval(file.read())

bad_GRMB = [
    (datetime.strptime(start, "%Y-%m-%d %H:%M:%S"), datetime.strptime(end, "%Y-%m-%d %H:%M:%S"), label)
    for start, end, label in time_windows_strings
]
bad_GRMB.sort(key=lambda window: window[0])
assign_region(cluster1, crossings, bad_days=bad_GRMB)

# Merging
buffer_R = 4

# GRMB Database
print('\nGRMB')
c1_sw_grmb = filter_sw(cluster1, 'GRMB', regions=(12,))
c1_omni_grmb = merge_dataframes(c1_sw_grmb, omni, 'C1', 'OMNI') # C1 vs OMNI using GRMB

# Jelinek Model
print('\nJelínek')
c1_sw_jel = filter_sw(cluster1, method='model', sc_key='C1', buffer=buffer_R)
c1_omni_jel = merge_dataframes(c1_sw_jel, omni, 'C1', 'OMNI') # C1 vs OMNI using Jelinek

# Combined
print('\nCombined')
c1_sw_com = filter_sw(cluster1, method='combined', regions=(12,), sc_key='C1', buffer=buffer_R)
c1_omni_com = merge_dataframes(c1_sw_com, omni, 'C1', 'OMNI')

file_path = 'timestamps.txt'
if False:
    c1_sw_com.index.strftime("%Y-%m-%d %H:%M:%S").to_series().to_csv(file_path, index=False, header=False)

# %%
print(c1_sw_com['GRMB_region'].value_counts().to_dict())
print(c1_sw_jel['GRMB_region'].value_counts().to_dict())
# %% Poster
from src.plotting.space_time import plot_dataset_years_months

plot_dataset_years_months(c1_sw_com, colour='deepskyblue')

# %% High_activity
from data_plotting import plot_compare_datasets_with_activity

plot_compare_datasets_with_activity(c1_sw_jel, c1_sw_grmb)