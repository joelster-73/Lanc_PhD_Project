# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:58:08 2025

@author: richarj2
"""

from src.processing.themis.config import THEMIS_SPACECRAFT
from src.processing.themis.handling import process_themis_files, resample_themis_files
from src.processing.updating import update_plasma_data

import warnings

warnings.filterwarnings('error', category=RuntimeWarning)


# %% Position

for spacecraft in THEMIS_SPACECRAFT:

    process_themis_files(spacecraft, 'STATE', sample_intervals=('1min','5min','15min'))

# %% Field

for spacecraft in THEMIS_SPACECRAFT:

    process_themis_files(spacecraft, 'FGM', sample_intervals=('raw','1min','5min','15min'))


# %% Plasma

for spacecraft in THEMIS_SPACECRAFT:

    process_themis_files(spacecraft, 'MOM', sample_intervals=('raw',))

# %% Filter

for spacecraft in THEMIS_SPACECRAFT:

    print(f'{spacecraft}:')

    update_plasma_data(spacecraft, 'FGM', 'MOM', 'omni', ('sw','msh'), field_res='raw')

    for region in ('sw','msh'):

        print(f'{region}:')

        resample_themis_files(spacecraft, region, 'spin', sample_intervals=('1min','5min','15min'))


# %% Resample-only

def resample():

    for spacecraft in THEMIS_SPACECRAFT:

        resample_themis_files(spacecraft, 'STATE', '1min', sample_intervals=('5min','15min'))
        resample_themis_files(spacecraft, 'FGM', 'raw', sample_intervals=('1min','5min','15min'))

        resample_themis_files(spacecraft, 'sw', 'spin', sample_intervals=('1min','5min','15min'))
        resample_themis_files(spacecraft, 'msh', 'spin', sample_intervals=('1min','5min','15min'))

if False:

    resample()

