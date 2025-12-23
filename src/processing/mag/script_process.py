# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 13:42:09 2025

@author: richarj2
"""


# %% Lagged_indices
from src.processing.mag.indices import build_lagged_indices

for sample_interval in ('1min','5min'): # for OMNI's indices
    build_lagged_indices(sample_interval)


# %% NetCDF_Files
from src.processing.mag.supermag import process_supermag_data


station = 'THL'
process_supermag_data(station)



# %% Convert_to_GSE
from src.processing.mag.supermag import convert_supermag_gse

station = 'THL'
convert_supermag_gse(station)

# %% aGSE



# instead of OMNI use a fixed average DP2 angle, around 30 degrees
# then include this with the aberration - whichever is larger (so during quiet times, aberration larger, during active times, DP2 more angled)


###
    # convert_supermag_agse(station)
    # add convert_GSE_to_aGSE() method to the build_lagged_indices

###