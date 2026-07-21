# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""

from src.processing.themis.config import THEMIS_SPACECRAFT, THEMIS_PLASMA_SPACECRAFT
from src.processing.themis.handling import process_themis_files, resample_themis_files
from src.processing.updating import update_plasma_data

# %% Position

for spacecraft in THEMIS_SPACECRAFT:

    process_themis_files(spacecraft, 'STATE', sample_intervals=('1min','5min','15min'))

# %% Field

for spacecraft in THEMIS_SPACECRAFT:

    process_themis_files(spacecraft, 'FGM', sample_intervals=('raw','1min','5min','15min'))

# %% Plasma

for spacecraft in THEMIS_PLASMA_SPACECRAFT:

    process_themis_files(spacecraft, 'MOM', sample_intervals=('raw',)) # thb for msh; the for sw

# %% Filter

regions = {'thb': 'sw', 'the': 'msh'}

for spacecraft in THEMIS_PLASMA_SPACECRAFT:

    region = regions.get(spacecraft)

    update_plasma_data(spacecraft, 'FGM', 'MOM', 'omni', (region,), field_res='raw')

    resample_themis_files(spacecraft, region, 'spin', sample_intervals=('1min','5min','15min'))


# %% TEMP

for spacecraft in THEMIS_SPACECRAFT:

    resample_themis_files(spacecraft, 'STATE', '1min', sample_intervals=('5min','15min'))
    resample_themis_files(spacecraft, 'FGM', 'raw', sample_intervals=('1min','5min','15min'))

# re-run the position and field cells
# then resample plasma below
# state and field can then be done in a similar fashion to below

resample_themis_files('thb', 'sw', 'spin', sample_intervals=('1min','5min','15min'))
resample_themis_files('the', 'msh', 'spin', sample_intervals=('1min','5min','15min'))