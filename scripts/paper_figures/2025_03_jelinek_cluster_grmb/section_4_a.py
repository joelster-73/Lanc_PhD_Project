# Section_4_Grison

from src.config import PROC_CLUS_DIR_1MIN as PROC_CLUS_DIR, PROC_SCBS_DIR, C1_RANGE, PROC_OMNI_DIR, CROSSINGS_DIR
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

# Merging
crossings = import_processed_data(CROSSINGS_DIR)
cross_labels = crossings.attrs['crossings']
assign_region(cluster1, crossings, print_changes=False)

c1_sw_grmb_v0 = filter_sw(cluster1, 'GRMB', regions=(12,))
c1_omni_grmb_v0 = merge_dataframes(c1_sw_grmb_v0, omni, 'C1', 'OMNI') # C1 vs OMNI using GRMB


# %% Region_12
from src.plotting.comparing.parameter import compare_columns
from src.config import COMPRESSIONS_DIR

compare_columns(c1_omni_grmb_v0, 'B_avg_C1', 'B_avg_OMNI',
                display='Heat', bin_width=(0.2,0.4), brief_title='Region 12 Comparison',
                compressions=COMPRESSIONS_DIR, contam_info=True)

# %% Differences
from src.methods.bowshock_filtering.extreme import plot_extreme_diffs_windows

plot_extreme_diffs_windows(cluster1, omni, c1_omni_grmb_v0, 'B_avg', 'C1', 'OMNI', crossings, compressions=COMPRESSIONS_DIR)