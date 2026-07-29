# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""

from src.processing.themis.config import THEMIS_SPACECRAFT
from src.processing.themis.handling import process_themis_files, resample_themis_files
from src.processing.updating import update_plasma_data

# %% Position

for spacecraft in THEMIS_SPACECRAFT:

    process_themis_files(spacecraft, 'STATE', sample_intervals=('1min','5min','15min'), start_year=2020)

# %% Field

for spacecraft in THEMIS_SPACECRAFT:

    process_themis_files(spacecraft, 'FGM', sample_intervals=('raw','1min','5min','15min'), start_year=2020)

# %% Plasma

for spacecraft in THEMIS_SPACECRAFT:

    process_themis_files(spacecraft, 'MOM', sample_intervals=('raw',), start_year=2020) # thb for msh; the for sw

# %% Filter

import itertools as it

for spacecraft, region in it.product(THEMIS_SPACECRAFT,('sw','msh')):

    print(spacecraft,region)

    update_plasma_data(spacecraft, 'FGM', 'MOM', 'omni', (region,), field_res='raw', start_year=2020)

    resample_themis_files(spacecraft, region, 'spin', sample_intervals=('1min','5min','15min'), start_year=2020)


# %% Resample-only

#### REMOVE START YEARS

for spacecraft in THEMIS_SPACECRAFT:

    resample_themis_files(spacecraft, 'STATE', '1min', sample_intervals=('5min','15min'))
    resample_themis_files(spacecraft, 'FGM', 'raw', sample_intervals=('1min','5min','15min'))

    resample_themis_files(spacecraft, 'sw', 'spin', sample_intervals=('1min','5min','15min'))
    resample_themis_files(spacecraft, 'msh', 'spin', sample_intervals=('1min','5min','15min'))

