# Section_4_Grison
from datetime import datetime

from src.config import PROC_CLUS_DIR_1MIN as PROC_CLUS_DIR, PROC_SCBS_DIR, C1_RANGE, PROC_OMNI_DIR, BAD_CROSSINGS, CROSSINGS_DIR
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

with open(BAD_CROSSINGS, 'r') as file: # Bad days to be removed from data analysis of solar wind
    time_windows_strings = eval(file.read())

bad_GRMB = [
    (datetime.strptime(start, "%Y-%m-%d %H:%M:%S"), datetime.strptime(end, "%Y-%m-%d %H:%M:%S"), label)
    for start, end, label in time_windows_strings
]
bad_GRMB.sort(key=lambda window: window[0])
assign_region(cluster1, crossings, bad_days=bad_GRMB, print_changes=True)

# Merging
c1_sw_grmb_11 = filter_sw(cluster1, 'GRMB', regions=(11,))
c1_omni_grmb_11 = merge_dataframes(c1_sw_grmb_11, omni, 'C1', 'OMNI') # C1 vs OMNI using GRMB


# %% Region_11
from src.methods.bowshock_filtering.comparing import plot_compare_columns_with_compression
from src.config import COMPRESSIONS_DIR

plot_compare_columns_with_compression(c1_omni_grmb_11, 'B_avg_OMNI', 'B_avg_C1', display='heat', bin_width=(0.2,0.4), brief_title='Region 11 Comparison', compressions=COMPRESSIONS_DIR, contam_info=True)

# %% Buffer
from src.methods.bowshock_filtering.jelinek import find_best_buffer

find_best_buffer(c1_omni_grmb_11, 'r_bs_diff_C1', 'B_avg_C1', 'B_avg_OMNI', plot=False, scale='lin', compressions=COMPRESSIONS_DIR, buff_max=6)

