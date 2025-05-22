
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
insert_bs_diff(cluster1, c1_bs_distances, 'C1', rel=True)

assign_region(cluster1, crossings, bad_days=bad_GRMB)

# Import OMNI
omni = import_processed_data(proc_omni_dir,
                             date_range=c1_range)
"""omni = import_processed_data(proc_omni_dir,
                             date_range=c1_range,
                             bad_data=bad_omni)
filter_by_spacecraft(omni, 'sc_id', 60, include=False) # removing geotail data"""
insert_cyl_coords(omni, field='r', coords='GSE')

# Merging and filtering
c1_sw_grmb = filter_sw(cluster1, 'GRMB', regions=(12,))
c1_omni_grmb = merge_dataframes(c1_sw_grmb, omni, 'C1', 'OMNI') # C1 vs OMNI using GRMB

# Jelinek Model
c1_sw_jel = filter_sw(cluster1, method='model', sc_key='C1', buffer=buffer_R)
c1_omni_jel = merge_dataframes(c1_sw_jel, omni, 'C1', 'OMNI') # C1 vs OMNI using Jelinek

# Deleting variables
del c1_sw_grmb
del c1_sw_jel

# %% Test

# Original GRMB
cluster1_test = cluster1.copy()
cluster1_test.attrs = cluster1.attrs
assign_region(cluster1_test, crossings, print_changes=False)

# %% Orbits
from data_plotting import plot_orbit


plot_orbit(cluster1, plane='x-rho', coords='GSE', models='Simple_BS',
           display='Heat', bin_width=0.1, brief_title='Cluster\'s orbit from 2001 to 2023')


# %% Finding_buffer
from data_plotting import compare_columns, plot_grid_bowshock_buffer, investigate_difference, plot_best_buffer
from data_analysis import find_best_buffer

# Jelinek no buffer
c1_sw_jel_0 = filter_sw(cluster1, method='model', sc_key='C1', buffer=0)
c1_omni_jel_0 = merge_dataframes(c1_sw_jel_0, omni, 'C1', 'OMNI') # C1 vs OMNI using Jelinek no buffer

compare_columns(c1_omni_jel_0, 'B_avg_C1', 'B_avg_OMNI',
                display='Heat', bin_width=0.2, brief_title='Comparing |B| for Cluster and OMNI',
                compressions=compressions_dir, contam_info=True)

investigate_difference(c1_omni_jel_0, 'B_avg_C1', 'B_avg_OMNI', 'r_bs_diff_C1',
                       display='Heat', x_data_name='r_C1 - r_BS', bin_width=[0.1,1])

# Finding Buffer
c1_omni = merge_dataframes(cluster1, omni, 'C1', 'OMNI') # C1 vs OMNI

plot_grid_bowshock_buffer(c1_omni, 'r_bs_diff_C1', 'B_avg_C1', 'B_avg_OMNI', 'r_x_GSE_C1', 'r_rho_C1',
                          r_diff_name='r_C1 - r_BS', y_diff_name='B_mag_C1 - B_mag_OMNI', rx_name='r_x_C1',
                          compressions=compressions_dir)

plot_best_buffer(c1_omni, 'r_bs_diff_C1', 'B_avg_C1', 'B_avg_OMNI', scale='lin',
                 compressions=compressions_dir, buff_max=6, data_name='Region 13')

find_best_buffer(c1_omni, 'r_bs_diff_C1', 'B_avg_C1', 'B_avg_OMNI',scale='both',
                 compressions=compressions_dir, buff_max=10)

del c1_sw_jel_0
del c1_omni_jel_0
del c1_omni



# %% Region_12_Analysis

from data_plotting import plot_extreme_diffs_windows


# Find bad Region 12
c1_sw_grmb_contam_12 = filter_sw(cluster1_test, 'GRMB', regions=(12,)) # Original "sw" data
c1_omni_grmb_contam_12 = merge_dataframes(c1_sw_grmb_contam_12, omni, 'C1', 'OMNI')


#plot_extreme_diffs_windows(cluster1_test, omni, c1_omni_grmb_contam_12, 'B_avg', 'C1', 'OMNI', crossings,
#                                         compressions=compressions_dir)

# %% Cleaning_Region_12
print_window = ('2018-03-22 05:10:00','2018-03-22 05:30:00')
from data_analysis import extreme_diffs
extreme_diffs(cluster1_test, omni, c1_omni_grmb_contam_12, 'B_avg', 'C1', 'OMNI', crossings,
              on_day='2016-05-19', all_cluster=None, compressions=compressions_dir)


# %% Post_Region_12_Analysis

from data_plotting import compare_columns_and_datasets

# Original GRMB with my changed Region 12s
assign_region(cluster1_test, crossings, bad_days=bad_GRMB, print_changes=True)

# Clean Region 12 Data
c1_sw_grmb_12 = filter_sw(cluster1_test, 'GRMB', regions=(12,))
c1_omni_grmb_12 = merge_dataframes(c1_sw_grmb_12, omni, 'C1', 'OMNI')

compare_columns(c1_omni_grmb_contam_12, 'B_avg_C1', 'B_avg_OMNI',
                display='Heat', bin_width=0.2, brief_title='Comparing |B| for Cluster and OMNI',
                compressions=compressions_dir, contam_info=True)

compare_columns_and_datasets(c1_omni_grmb_contam_12, c1_omni_grmb_12, 'B_avg_C1', 'B_avg_OMNI',
                display='Heat', bin_width=0.2, title1='Before Analysis', title2='After Analysis',
                compressions=compressions_dir, contam_info=True)


del c1_sw_grmb_12
del c1_sw_grmb_contam_12

# %% Region_11_Analysis
c1_sw_contam_grmb_11 = filter_sw(cluster1_test, 'GRMB', regions=(11,))
c1_omni_grmb_contam_11 = merge_dataframes(c1_sw_contam_grmb_11, omni, 'C1', 'OMNI')


compare_columns(c1_omni_grmb_contam_11, 'B_avg_C1', 'B_avg_OMNI',
                display='Heat', bin_width=(0.2,0.4), brief_title='Comparing |B| for Cluster and OMNI',
                compressions=compressions_dir, contam_info=True)

plot_best_buffer(c1_omni_grmb_contam_11, 'r_bs_diff_C1', 'B_avg_C1', 'B_avg_OMNI', scale='lin',
                 compressions=compressions_dir, buff_max=6, data_name='Region 11')

del c1_sw_contam_grmb_11
del c1_omni_grmb_contam_11


# %% Region_13_Analysis

from data_plotting import plot_best_buffer, compare_columns

c1_sw_grmb_contam_13 = filter_sw(cluster1_test, 'GRMB', regions=(13,))
c1_omni_grmb_contam_13 = merge_dataframes(c1_sw_grmb_contam_13, omni, 'C1', 'OMNI')


compare_columns(c1_omni_grmb_contam_13, 'B_avg_C1', 'B_avg_OMNI',
                display='Heat', bin_width=(0.2,0.4), brief_title='Comparing |B| for Cluster and OMNI',
                compressions=compressions_dir, contam_info=True)

del c1_sw_grmb_contam_13
del c1_omni_grmb_contam_13

# %%