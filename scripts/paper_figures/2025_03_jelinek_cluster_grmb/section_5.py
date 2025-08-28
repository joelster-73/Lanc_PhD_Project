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
# %% Comparison

c1_sw_jel_times = c1_sw_jel.index
c1_sw_grmb_times = c1_sw_grmb.index
c1_sw_jel_times_no_2023 = c1_sw_jel[c1_sw_jel.index.year<2023].index

common_times = c1_sw_grmb_times.intersection(c1_sw_jel_times)
print('\nData')
print(f'Total Cluster:   {len(cluster1):,}')
print(f'Overlap:         {len(common_times):,}')
print(f'GRMB:            {len(c1_sw_grmb_times):,}, {len(common_times)/len(c1_sw_grmb_times)*100:.2f}%, '
      f'{len(c1_sw_grmb_times)-len(common_times):,}')
print(f'Jelinek:         {len(c1_sw_jel_times):,},   {len(common_times)/len(c1_sw_jel_times)*100:.2f}%, '
      f'{len(c1_sw_jel_times)-len(common_times):,}')
print(f'Jelinek no 2023: {len(c1_sw_jel_times_no_2023):,},   {len(common_times)/len(c1_sw_jel_times_no_2023)*100:.2f}%, '
      f'{len(c1_sw_jel_times_no_2023)-len(common_times):,}')

# %% Pressures
from src.analysing.calculations import calc_mean_error, calc_sample_std

print('\nPressures')
print(f'OMNI: mu={calc_mean_error(omni.loc[:,"p_flow"]):u} nPa, s={calc_sample_std(omni.loc[:,"p_flow"]):.4f} nPa')
print(f'Jel:  mu={calc_mean_error(c1_omni_jel.loc[:,"p_flow_OMNI"]):u} nPa, s={calc_sample_std(c1_omni_jel.loc[:,"p_flow_OMNI"]):.4f} nPa')
print(f'GRMB: mu={calc_mean_error(c1_omni_grmb.loc[:,"p_flow_OMNI"]):u} nPa, s={calc_sample_std(c1_omni_grmb.loc[:,"p_flow_OMNI"]):.4f} nPa')

# %% Comparison_Plots

from src.plotting.comparing.space_time import plot_compare_datasets_space
from src.methods.bowshock_filtering.comparing import plot_compare_years_with_apogee

plot_compare_datasets_space(c1_sw_jel, c1_sw_grmb, plane='x-rho', coords='GSE', df1_name='Jelínek Bow shock', df2_name='Grison Database', display='heat', bin_width=0.1, df_omni=omni)

plot_compare_years_with_apogee(c1_sw_jel, c1_sw_grmb, cluster1, df1_name='Jelínek Bow shock', df2_name='Grison Database', df1_colour='g', df2_colour='b', show_apogee=True, print_data=True)


# %% Comparison
from src.plotting.comparing.datasets import plot_compare_dataset_parameters

column_names = {'B_clock':'Clock Angle',
                'B_cos(th)':'Bx GSM',
                'B_sin(th)_sin(ph)':'By GSM',
                'B_sin(th)_cos(ph)':'Bz GSM'}
display_ranges = {'B_avg':(0,99),
                         'B_cos(th)':(0.2,99.8),
                         'B_sin(th)_sin(ph)':(0.5,99.5),
                         'B_sin(th)_cos(ph)':(0.5,99.5)}
fit_types = {'B_avg':'lognormal',
             'B_clock':'bimodal_offset',
             'B_cos(th)':'bimodal',
             'B_sin(th)_sin(ph)':'bimodal',
             'B_sin(th)_cos(ph)':'gaussian'}

omni_columns={'B_cos(th)':'B_x_GSE',
              'B_sin(th)_sin(ph)':'B_y_GSM',
              'B_sin(th)_cos(ph)':'B_z_GSM'}
bin_widths = {'B_avg':0.1,
                     'B_cos(th)':0.05,
                     'B_sin(th)_sin(ph)':0.05,
                     'B_sin(th)_cos(ph)':0.05,
                     'B_clock':5}

# %% Comparing_B
plot_compare_dataset_parameters(c1_sw_jel, c1_sw_grmb, 'B_avg', df1_name='Jelínek Dataset', df2_name='GRMB Database', bin_widths=bin_widths, column_names=column_names, display_ranges=display_ranges, fit_types=fit_types, compare_type='q_q_plot')

# %% Comparing_all_q_q

plot_compare_dataset_parameters(c1_sw_com, omni, 'B_avg', 'B_cos(th)', 'B_sin(th)_sin(ph)', 'B_sin(th)_cos(ph)', 'B_clock',df1_colour='deepskyblue', df1_name='Cluster', df2_colour='orange', df2_name='OMNI', bin_widths=bin_widths, column_names=column_names, display_ranges=display_ranges, fit_types=fit_types, compare_type='q_q_plot', df2_columns=omni_columns)

# %% Comparing_all_hist
plot_compare_dataset_parameters(c1_sw_com, omni, 'B_avg', 'B_cos(th)', 'B_sin(th)_sin(ph)', 'B_sin(th)_cos(ph)', 'B_clock',df1_colour='deepskyblue', df1_name='Cluster', df2_colour='orange', df2_name='OMNI', bin_widths=bin_widths, column_names=column_names, display_ranges=display_ranges, fit_types=fit_types, compare_type='hist_diff', df2_columns=omni_columns, compare_label='C1 - OMNI')

