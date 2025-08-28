# Section_3_Jelinek

import numpy as np

from src.config import PROC_CLUS_DIR_1MIN as PROC_CLUS_DIR, PROC_SCBS_DIR, C1_RANGE, PROC_OMNI_DIR
from src.coordinates.spatial import insert_car_coords, insert_cyl_coords
from src.methods.bowshock_filtering.jelinek import insert_bs_diff

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
c1_omni = merge_dataframes(cluster1, omni, 'C1', 'OMNI') # C1 vs OMNI
c1_sw_jel_0 = filter_sw(cluster1, method='model', sc_key='C1', buffer=0)
c1_omni_jel_0 = merge_dataframes(c1_sw_jel_0, omni, 'C1', 'OMNI') # C1 vs OMNI using Jelinek no buffer


# %% Initial_comparison
from src.methods.bowshock_filtering.comparing import plot_compare_columns_with_compression
from src.config import COMPRESSIONS_DIR

plot_compare_columns_with_compression(c1_omni_jel_0, 'B_avg_OMNI', 'B_avg_C1', display='heat', bin_width=(0.2,0.4), brief_title='Comparing |B| for Cluster and OMNI', compressions=COMPRESSIONS_DIR, contam_info=True)


# %% Kobel
from src.methods.bowshock_filtering.kobel_figures import plot_over_B

Pd_0=1
B_dir = np.array([0, 0, -1])
B_unit_0 =  B_dir/np.linalg.norm(B_dir)

plot_over_B(Pd_0,B_unit_0,quantity='min',show_both=True,max_B=30)

# %% Against_r
from src.plotting.comparing.parameter import investigate_difference

investigate_difference(c1_omni_jel_0, 'B_avg_OMNI', 'B_avg_C1', ind_col='r_bs_diff_C1', display='heat', bin_width=(0.05,0.5), add_count=False, want_legend=False, ind_name_lat='r_C1 - r_BS', brief_title='Comparing |B| for Cluster and OMNI')

# %% Best_Buffer
from src.config import COMPRESSIONS_DIR
from src.methods.bowshock_filtering.jelinek import find_best_buffer

find_best_buffer(c1_omni, 'r_bs_diff_C1', 'B_avg_C1', 'B_avg_OMNI', plot=True, scale='lin', compressions=COMPRESSIONS_DIR, buff_max=6)


# %% Grid
from src.methods.bowshock_filtering.grid_gif_plots import plot_grid_bowshock_buffer

plot_grid_bowshock_buffer(c1_omni, 'r_bs_diff_C1', 'B_avg_OMNI', 'B_avg_C1', 'r_x_GSE_C1', 'r_rho_C1', buff_max=4, display='heat', compressions=COMPRESSIONS_DIR, sc_key='C1', r_diff_lat='r_C1 - r_BS', df_omni=omni)



