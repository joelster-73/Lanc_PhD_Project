# %% Imports
from handling_files import import_processed_data

from pandas import Timedelta
from datetime import datetime

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow encountered in square')
warnings.filterwarnings("ignore", category=FutureWarning)
# %% Globals
# LUNA Directories
luna_clus_dir_spin = 'Z:/spacecraft/cluster/c1/C1_CP_FGM_SPIN/'
luna_clus_dir_5vps = 'Z:/spacecraft/cluster/c1/C1_CP_FGM_5VPS/'
luna_cros_dir      = 'Z:/spacecraft/cluster/c1/C1_CT_AUX_GRMB/'
luna_omni_dir      = 'Z:/omni/omni_1min_yearly/'
luna_wind_dir      = 'Z:/spacecraft/wind/mfi/mfi_h0/'

# Saving processed data
proc_clus_dir_spin = '../Data/Processed_Data/Cluster1/SPIN/'
proc_clus_dir_5vps = '../Data/Processed_Data/Cluster1/5VPS/'
proc_clus_dir_raw  = '../Data/Processed_Data/Cluster1/5VPS/raw/'
proc_clus_dir      = '../Data/Processed_Data/Cluster1/5VPS/1min/'
proc_cros_dir      = '../Data/Processed_Data/Cluster1/Crossings/'
good_crossings     = '../Data/Processed_Data/Cluster1/Crossings/good_time_windows.txt'
bad_crossings      = '../Data/Processed_Data/Cluster1/Crossings/bad_time_windows.txt'
compressions_dir   = '../Data/Processed_Data/Cluster1/Crossings/compression_ratios.npz'

proc_omni_dir      = '../Data/Processed_Data/OMNI/'
proc_wind_dir      = '../Data/Processed_Data/WIND/1_min/'
proc_wind_dir_1    = '../Data/Processed_Data/WIND/1_hour/'
proc_wind_dir_3    = '../Data/Processed_Data/WIND/3_sec/'
proc_scbs_dir      = '../Data/Processed_Data/Bowshock/'

sample_size = '1min'
deltaT = Timedelta(minutes=1)

# Data to be extracted from cluster\n
cluster_variables_spin = {
    'epoch' : 'time_tags__C1_CP_FGM_SPIN',      # extracts the epoch time in milliseconds (scalar).
    'r'     : 'sc_pos_xyz_gse__C1_CP_FGM_SPIN', # extracts the position in km from centre of Earth (vector).
    'B_avg' : 'B_mag__C1_CP_FGM_SPIN',          # extracts the fgm data in nT (magnitude).
    'B'     : 'B_vec_xyz_gse__C1_CP_FGM_SPIN',  # extracts the fgm data in nT (vector).
}
cluster_variables_5vps = {
    'epoch' : 'time_tags__C1_CP_FGM_5VPS',      # extracts the epoch time in milliseconds (scalar).
    'r'     : 'sc_pos_xyz_gse__C1_CP_FGM_5VPS', # extracts the position in km from centre of Earth (vector).
    'B_avg' : 'B_mag__C1_CP_FGM_5VPS',          # extracts the fgm data in nT (magnitude).
    'B'     : 'B_vec_xyz_gse__C1_CP_FGM_5VPS',  # extracts the fgm data in nT (vector).
}
cluster_variables_5vps_pos = {
    'epoch' : 'epoch',
    'r_x'   : 'r_x_GSE',
    'r_y'   : 'r_y_GSE',
    'r_z'   : 'r_z_GSE',
}

omni_variables = ('time','satellite','B_field','pressure','velocity','propagation')
omni_spacecraft = {
    71: 'Ace',
    60: 'Geotail',
    50: 'IMP 8',
    51: 'Wind',
}

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

wind_variables = { # 1 minute intervals (at midpoints, i.e. 30 seconds)
    'epoch' : 'Epoch', # epoch time in milliseconds (scalar).
    'r'     : 'PGSE',  # position in Re=6738km from centre of Earth (vector).
    'B_avg' : 'BF1',   # fgm nT (magnitude).
    'B_GSE' : 'BGSE',  # fgm nT (vector).
    'B_GSM' : 'BGSM',  # fgm nT (vector).
}

wind_variables_3_sec = { # at 3 second intervals (at midpoints, i.e. 1.5 seconds)
    'epoch' : 'Epoch3', # No 3-sec res on position
    'B_avg' : 'B3F1',
    'B_GSE' : 'B3GSE',
    'B_GSM' : 'B3GSM',
}

wind_variables_1_hour = { # at 1 hour intervals (at midpoints, i.e. 30 minutes)
    'epoch' : 'Epoch1',
    'r'     : 'P1GSE',
    'B_avg' : 'B1F1',
    'B_GSE' : 'B1GSE',
    'B_GSM' : 'B1GSM',
}

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
c1_sw_jel_0 = filter_sw(cluster1, method='model', sc_key='C1', buffer=0)
print(len(c1_sw_jel_0))

c1_sw_jel = filter_sw(cluster1, method='model', sc_key='C1', buffer=buffer_R)
c1_omni_jel = merge_dataframes(c1_sw_jel, omni, 'C1', 'OMNI') # C1 vs OMNI using Jelinek

# Combined
c1_sw_rich = filter_sw(cluster1, method='combined', regions=(12,), sc_key='C1', buffer=buffer_R)
c1_omni_rich = merge_dataframes(c1_sw_rich, omni, 'C1', 'OMNI')


file_path = 'timestamps.txt'
c1_sw_rich.index.strftime("%Y-%m-%d %H:%M:%S").to_series().to_csv(file_path, index=False, header=False)


c1_omni = merge_dataframes(cluster1, omni, 'C1', 'OMNI') # C1 vs OMNI

del c1_sw_jel_0
# %% Comparison

c1_sw_jel_times = c1_sw_jel.index
c1_sw_grmb_times = c1_sw_grmb.index
c1_sw_jel_times_no_2023 = c1_sw_jel[c1_sw_jel.index.year<2023].index

common_times = c1_sw_grmb_times.intersection(c1_sw_jel_times)
print(f'Overlap:         {len(common_times):,}')
print(f'GRMB:            {len(c1_sw_grmb_times):,}, {len(common_times)/len(c1_sw_grmb_times)*100:.2f}%, '
      f'{len(c1_sw_grmb_times)-len(common_times):,}')
print(f'Jelinek:         {len(c1_sw_jel_times):,}, {len(common_times)/len(c1_sw_jel_times)*100:.2f}%, '
      f'{len(c1_sw_jel_times)-len(common_times):,}')
print(f'Jelinek no 2023: {len(c1_sw_jel_times_no_2023):,}, {len(common_times)/len(c1_sw_jel_times_no_2023)*100:.2f}%, '
      f'{len(c1_sw_jel_times_no_2023)-len(common_times):,}')

#from data_analysis import compare_datasets

#compare_datasets(c1_sw_jel, c1_sw_grmb, omni, 'B_avg', 'B_cos(th)', 'B_sin(th)_sin(ph)', 'B_sin(th)_cos(ph)', 'B_clock', column_names={'B_clock':'Clock Angle', 'B_cos(th)':'Bx', 'B_sin(th)_sin(ph)':'By', 'B_sin(th)_cos(ph)':'Bz'})



# %% Pressures
from data_analysis import calc_mean_error
from numpy import std

print('OMNI:',calc_mean_error(omni['p_flow']),f"{std(omni['p_flow']):.4f}")
print('Jel: ',calc_mean_error(c1_omni_jel['p_flow_OMNI']),f"{std(c1_omni_jel['p_flow_OMNI']):.4f}")
print('GRMB:',calc_mean_error(c1_omni_grmb['p_flow_OMNI']),f"{std(c1_omni_grmb['p_flow_OMNI']):.4f}")

# %% Comparison_Plots

from data_plotting import compare_columns_and_datasets, plot_compare_datasets_with_activity

compare_columns_and_datasets(c1_omni_jel, c1_omni_grmb, 'B_avg_C1', 'B_avg_OMNI',
                display='Heat', bin_width=0.2, title1='Jelínek Bow shock', title2='GRMB Database',
                compressions=compressions_dir, contam_info=True)

from data_plotting import plot_compare_datasets_space, plot_compare_datasets_years

plot_compare_datasets_space(c1_sw_jel, c1_sw_grmb, 'r_x_GSE', 'r_rho', omni)
plot_compare_datasets_years(c1_sw_jel, c1_sw_grmb, show_apogee=True, full_cluster=cluster1, print_data=True)

plot_compare_datasets_with_activity(c1_sw_jel, c1_sw_grmb)
plot_compare_datasets_with_activity(c1_sw_rich, df_names=('Combined',), df_colours=('deepskyblue',), df_markers=('s',))


# %% Comparison
from data_plotting import plot_compare_dataset_parameters, plot_dataset_years_months

column_names = {'B_clock':'Clock Angle',
                'B_cos(th)':'Bx GSM',
                'B_sin(th)_sin(ph)':'By GSM',
                'B_sin(th)_cos(ph)':'Bz GSM'}
column_display_ranges = {'B_avg':(0,20),
                         'B_cos(th)':(-15,15),
                         'B_sin(th)_sin(ph)':(-15,15),
                         'B_sin(th)_cos(ph)':(-15,15)}
fit_types = {'B_avg':'Log_Normal',
             'B_clock':'Bimodal_offset',
             'B_cos(th)':'Bimodal',
             'B_sin(th)_sin(ph)':'Bimodal',
             'B_sin(th)_cos(ph)':'Gaussian'}
omni_columns={'B_cos(th)':'B_x_GSE',
              'B_sin(th)_sin(ph)':'B_y_GSM',
              'B_sin(th)_cos(ph)':'B_z_GSM'}
column_bin_widths = {'B_avg':0.2,
                     'B_cos(th)':0.2,
                     'B_sin(th)_sin(ph)':0.2,
                     'B_sin(th)_cos(ph)':0.2,
                     'B_clock':5}


plot_compare_dataset_parameters(c1_sw_jel, c1_sw_grmb, 'B_avg',
                 q_q_plot=True, column_bin_widths=column_bin_widths,
                 column_names=column_names, column_display_ranges=column_display_ranges, fit_types=fit_types)

plot_compare_dataset_parameters(c1_sw_rich, omni, 'B_avg', 'B_cos(th)', 'B_sin(th)_sin(ph)', 'B_sin(th)_cos(ph)', 'B_clock',
                 df1_colour='deepskyblue', df1_name='Cluster', df2_colour='orange', df2_name='OMNI',
                 q_q_plot=True, column_bin_widths=column_bin_widths,
                 omni_columns=omni_columns, column_names=column_names, column_display_ranges=column_display_ranges, fit_types=fit_types)

plot_compare_dataset_parameters(c1_sw_rich, omni, 'B_avg', 'B_cos(th)', 'B_sin(th)_sin(ph)', 'B_sin(th)_cos(ph)', 'B_clock',
                 df1_colour='deepskyblue', df1_name='Cluster', df2_colour='orange', df2_name='OMNI',
                 hist_diff=True, column_bin_widths=column_bin_widths, contemp_omni=True, diff_type='Absolute',
                 omni_columns=omni_columns, column_names=column_names, column_display_ranges=column_display_ranges, fit_types=fit_types)

plot_dataset_years_months(c1_sw_rich, df_colour='deepskyblue')

print(c1_sw_rich['GRMB_region'].value_counts().to_dict())
print(c1_sw_jel['GRMB_region'].value_counts().to_dict())
# %% Poster


plot_compare_dataset_parameters(c1_sw_jel, omni, 'B_sin(th)_cos(ph)',
                 df1_colour='deepskyblue', df1_name='Cluster', df2_colour='orange', df2_name='OMNI',
                 hist_diff=True, column_bin_widths={'B_avg':0.125}, contemp_omni=True, diff_type='Absolute',
                 omni_columns=omni_columns, column_names=column_names, column_display_ranges=column_display_ranges, fit_types=fit_types)

plot_dataset_years_months(c1_sw_jel, df_colour='deepskyblue',show_apogee=True,full_cluster=cluster1)
