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

    if spacecraft in THEMIS_PLASMA_SPACECRAFT:
        sample_intervals = ('raw','1min','5min','15min')
    else:
        sample_intervals = ('1min','5min','15min')

    process_themis_files(spacecraft, 'FGM', sample_intervals=sample_intervals)

# %% Plasma

for spacecraft in THEMIS_PLASMA_SPACECRAFT:
    # thb for msh; the for sw

    process_themis_files(spacecraft, 'MOM', sample_intervals=('raw',))

# %% Filter

regions = {'thb': 'sw', 'the': 'msh'}

for spacecraft in THEMIS_PLASMA_SPACECRAFT:

    region = regions.get(spacecraft)

    update_plasma_data(spacecraft, 'FGM', 'MOM', 'omni', (region,), field_res='raw')

    resample_themis_files(spacecraft, region, 'spin', sample_intervals=('1min','5min','15min'))

