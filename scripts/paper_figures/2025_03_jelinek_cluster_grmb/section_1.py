# Section_1_Introduction

from src.config import PROC_CLUS_DIR_1MIN as PROC_CLUS_DIR, PROC_SCBS_DIR, C1_RANGE, PROC_OMNI_DIR
from src.processing.reading import import_processed_data
from src.coordinates.spatial import insert_car_coords, insert_cyl_coords
from src.methods.bowshock_filtering.jelinek import insert_bs_diff

# %% Cluster

# 1-minute Cluster data
cluster1 = import_processed_data(PROC_CLUS_DIR)
# Undoing circular averaging for B
insert_car_coords(cluster1, field='B', coords='GSM', r_col='B_mag', th_col='B_pitch', ph_col='B_clock')
insert_cyl_coords(cluster1, field='r', coords='GSE') # Using 1-min data fine for position

# Import Jelinek filtering data
c1_bs_distances = import_processed_data(PROC_SCBS_DIR)
insert_bs_diff(cluster1, c1_bs_distances, 'C1')

# %% OMNI

omni = import_processed_data(PROC_OMNI_DIR, date_range=C1_RANGE)
insert_cyl_coords(omni, field='r', coords='GSE')

# %% Orbit
from src.plotting.space_time import plot_orbit

plot_orbit(cluster1, plane='x-rho', coords='GSE', models='Median BS', display='heat', bin_width=0.1, brief_title='Cluster\'s orbit from 2001 to 2023', equal_axes=True, df_omni=omni)

