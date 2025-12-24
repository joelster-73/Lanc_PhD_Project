# -*- coding: utf-8 -*-
"""
Created on Thu May  8 16:17:14 2025

@author: richarj2
"""
# src/config.py

import os
import warnings
from .processing.utils import create_directory

R_E = 6370 # Cluster takes 1 earth radius to be 6370 km


def short_warn_format(message, category, filename, lineno, line=None):
    # Get just the parent folder and filename, e.g. "magnetosheath_saturation/plotting.py"
    parent = os.path.basename(os.path.dirname(filename))
    base = os.path.basename(filename)
    short_path = f'{parent}/{base}'
    return f'{short_path}:{lineno}: {category.__name__}: {message}\n'

warnings.formatwarning = short_warn_format



CLUSTER_SPACECRAFT = ('c1', 'c2', 'c3', 'c4')
MMS_SPACECRAFT     = ('mms1', 'mms2', 'mms3', 'mms4')
THEMIS_SPACECRAFT  = ('tha', 'thb', 'thc', 'thd', 'the')


#####----------LUNA DIRECTORIES----------#####


LUNA_CLUS_DIR   = 'Z:/spacecraft/cluster'
LUNA_MMS_DIR    = 'Z:/spacecraft/mms'
LUNA_THEMIS_DIR = 'Z:/spacecraft/themis/'
LUNA_WIND_DIR   = 'Z:/spacecraft/wind/'
LUNA_ACE_DIR    = 'Z:/superdarn/ace/'
LUNA_OMNI_DIR   = 'Z:/omni/'

LUNA_PC_INDEX_DIR  = 'Z:/PC_index/'

# temporary
LUNA_MAG_DIR    = 'Y:/Processed_Data/GROUND/SuperMAG/NetCDF/'



def get_luna_directory(source, instrument=None, info=None):

    # cluster
    if source in CLUSTER_SPACECRAFT:
        path = f'{LUNA_CLUS_DIR}/{source}/'

        folder = f'{source.upper()}_CP'

        if instrument.upper() in ('FGM','STATE'):
            # STATE data is in the FGM files
            folder += '_FGM'

            if info is None:
                warnings.warn('No sample resolution provided; using SPIN.')
                info = 'SPIN'
            elif info not in ('SPIN','5VPS'):
                warnings.warn('Sample resolution "{sample}" not valid; using SPIN.')
                info = 'SPIN'

            folder += f'_{info.upper()}/'

        elif instrument.upper() in ('CIS','HIA','CIS-HIA'):
            folder += '_CIS-HIA'

            if info in ('mom', 'moments', 'hiam'):
                folder += '_ONBOARD_MOMENTS/'

            elif info in ('quality','hiaq'):
                folder += '_QUALITY/'

        elif instrument.upper() == 'GRMB':
            folder = f'{source.upper()}_CT_AUX_GRMB/'

        path += folder

    # mms
    elif source in MMS_SPACECRAFT:
        path = f'{LUNA_MMS_DIR}/{source}/'

        folder = f'{source.upper()}'

        if instrument.upper() == 'STATE':
            instrument = 'fgm' # STATE data in fgm files

        if instrument.upper() == 'FGM':
            folder += f'_{instrument.upper()}_SRVY_L2'

        elif instrument.upper() =='HPCA':
            folder += f'_{instrument.upper()}_SRVY_L2_MOMENTS'

        elif instrument.upper() == 'FPI':
            folder += f'_{instrument.upper()}_FAST_L2_DIS-MOMS'

        path += folder

    # themis
    elif source in THEMIS_SPACECRAFT:
        path = f'{THEMIS_DIR}/{source}/'

        if instrument.upper() in ('FGM', 'GMOM', 'MOM', 'STATE'):
            folder = f'{instrument.upper()}/'

        elif instrument.lower() in ('esa',):
            folder = f'{instrument.lower()}/'

        path += folder

    # omni
    elif source=='omni':
        path = LUNA_OMNI_DIR

        if info == '1min' or info is None:
            path += 'omni_min_def/'
        elif info == '5min':
            path += 'omni_5min_def/'

    # wind
    elif source=='wind':
        path = LUNA_WIND_DIR

        if instrument.lower()=='mfi':
            path += 'mfi/mfi_h0/'

    # supermag
    elif source=='supermag':
        path = LUNA_MAG_DIR

        if instrument is not None:
            path += instrument.upper()

    # supermag
    elif source=='pc_index':
        path = LUNA_PC_INDEX_DIR


    else:
        raise ValueError(f'Spacecraft "{source} does not have processed directory.')

    if not os.path.isdir(path):
        raise ValueError(f'Directory does not exist on LUNA: {path}.')


    return path



#####----------PROCESSED DIRECTORIES----------#####

# Define DATA_DIR relative to the module's location
MODULE_DIR             = os.path.dirname(os.path.abspath(__file__))  # Path of the current module
DATA_DIR               = os.path.join(MODULE_DIR, '..', 'data')
TEMP_DATA_DIR          = os.path.join(MODULE_DIR, '..', 'tempData')
FIGURES_DIR            = os.path.join(MODULE_DIR, '..', 'figures')

# Base Directories
PROCESSED_DATA_DIR     = 'Y:/Processed_Data'

CLUSTER_DIR            = f'{PROCESSED_DATA_DIR}/CLUSTER'
THEMIS_DIR             = f'{PROCESSED_DATA_DIR}/THEMIS'
MMS_DIR                = f'{PROCESSED_DATA_DIR}/MMS'

SHOCKS_DIR             = f'{PROCESSED_DATA_DIR}/SHOCKS'
REGION_DIR             = f'{PROCESSED_DATA_DIR}/REGION'
INDEX_DIR              = f'{PROCESSED_DATA_DIR}/INDICES'

MSH_DIR                = f'{REGION_DIR}/MSH'
SW_DIR                 = f'{REGION_DIR}/SW'
C1_BS_CROSSINGS        = f'{REGION_DIR}/Bowshock'

CROSSINGS_DIR          = f'{CLUSTER_DIR}/c1/Crossings'

GROUND_DIR             = f'{PROCESSED_DATA_DIR}/GROUND'

def get_proc_directory(source, dtype=' ', resolution=' ', create=False):

    # cluster
    if source in CLUSTER_SPACECRAFT:

        if dtype == 'base':
            path = CLUSTER_DIR

        else:
            path = f'{CLUSTER_DIR}/{source}/'

            if dtype=='plasma':
                raise ValueError('"plasma" has been replaced by "hpca" and "fpi"')
            elif dtype=='field':
                raise ValueError('"field" has been replaced by "fgm"')

            if dtype in ('crossings','fgm','hia','hiqa','state','combined','sw','msh'):
                path += f'{dtype}/'

            if resolution in ('raw','5vps','spin','1min','5min'):
                path += f'{resolution}/'

    # mms
    elif source in MMS_SPACECRAFT:

        if dtype == 'base':
            path = MMS_DIR

        else:
            path = f'{MMS_DIR}/{source}/'

            if dtype=='plasma':
                raise ValueError('"plasma" has been replaced by "hpca" and "fpi"')
            elif dtype=='field':
                raise ValueError('"field" has been replaced by "fgm"')

            if dtype in ('fgm','hpca','fpi','state'):
                path += f'{dtype}/'

            if resolution in ('fast','spin','raw','1min','5min'):
                path += f'{resolution}/'

    # themis
    elif source in THEMIS_SPACECRAFT:

        if dtype == 'base':
            path = THEMIS_DIR

        else:
            path = f'{THEMIS_DIR}/{source}/'

            if dtype=='plasma':
                raise ValueError('"plasma" has been replaced by "hpca" and "fpi"')
            elif dtype=='field':
                raise ValueError('"field" has been replaced by "fgm"')

            if dtype in ('sw', 'msh', 'combined', 'fgm', 'mom', 'state'):
                path += f'{dtype}/'

            if resolution in ('raw','1min','5min'):
                path += f'{resolution}/'

    # omni
    elif source=='omni':
        path = f'{PROCESSED_DATA_DIR}/OMNI/'

        if resolution == '':
            resolution = '1min'
        if resolution in ('1min','5min'):
            path += f'{resolution}/'


    # wind
    elif source=='wind':
        path = f'{PROCESSED_DATA_DIR}/WIND'
        if resolution in ('3sec','1min','1hour'):
            path += f'{resolution}/'

    # combined
    elif source in ('msh', 'sw'):
        if source=='msh':
            path = f'{MSH_DIR}/'
        elif source=='sw':
            path = f'{SW_DIR}/'

        if 'field' in dtype:
            path += 'field/'
        elif 'plasma' in dtype:
            path += 'plasma/'

        if resolution in ('1min','5min'):
            path += f'{resolution}/'

    # index
    elif source in ('pcn','pcc','aa','sme','indices'):

        if source=='indices':
            path = INDEX_DIR
        elif source=='pcc':
            path = f'{INDEX_DIR}/PCN_PCS'
        else:
            path = f'{INDEX_DIR}/{source.upper()}/'

    # supermag
    elif source=='supermag':

        path = f'{GROUND_DIR}/SuperMAG/CDF/'

        if dtype is not None:
            path += f'{dtype.upper()}/'

        if resolution in ('raw','gse','agse'):
            path += f'{resolution}/'

    # index
    elif source=='crossings':
        path = CROSSINGS_DIR

    else:
        raise ValueError(f'Spacecraft "{source} does not have processed directory.')

    if not os.path.isdir(path):
        if create:
            warnings.warn(f'Directory does not exist on processed drive: {path}. Creating directory...')
            create_directory(path)
        else:
            raise ValueError(f'Directory does not exist on processed drive: {path}.')


    return path
