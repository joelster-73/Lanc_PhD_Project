# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 12:50:42 2025

@author: richarj2
"""
import os
import glob
import numpy as np
import pandas as pd

from netCDF4 import Dataset

from ..reading import import_processed_data
from ..writing import write_to_cdf
from ...config import get_luna_directory, get_proc_directory
from ...coordinates.magnetic import convert_GEO_to_GSE, convert_GSE_to_aGSE


# All nT
bfield_cols = {'dbn_geo': 'B_n_GEO', # Magnetic field north component, geographic coordinates
               'dbe_geo': 'B_e_GEO', # Magnetic field east component, geographic coordinates
               'dbz_geo': 'B_z_GEO', # Magnetic field vertical component, geographic coordinates
               'dbn_nez': 'B_n_NEZ', # Magnetic field north component, NEZ coordinates
               'dbe_nez': 'B_e_NEZ', # Magnetic field east component, NEZ coordinates
               'dbz_nez': 'B_z_NEZ', # Magnetic field vertical component, NEZ coordinates
}

# all degrees
position_cols = {'decl'  : 'mdecl',    # Magnetic Declination, (D_m)
                 'mcolat': 'mcolat',   # Magnetic Colatitude,  (theta_m)
                 'glat'  : 'glat',     # Geographic Latitude,  (phi_g)
                 'glon'  : 'glon',     # Geographic Longtiude, (lambda_g)
                 'sza'   : 'chi_s',    # Solar Zenith Angle,
}

other_cols = {'extent': 'duration', # Extent of Record [seconds]
              'id'    : 'id',       # Station Identifier
              'mlt'   : 'MLT',      # Magnetic Local Time [h]
              'npnt'  : 'count',    # Number of Points in Record
              }

all_mappings = bfield_cols | position_cols | other_cols

def process_supermag_data(*stations):

    for station in stations:
        try:
            mag_dir = get_luna_directory('supermag', station)
        except:
            continue

        # data in netcdf format
        pattern = os.path.join(mag_dir, '*.netcdf') # netcdf
        files = sorted(glob.glob(pattern))

        if not files:
            raise ValueError(f"No files found in the directory: {mag_dir}")

        df_years = []
        for file in files:
            df_year = pd.DataFrame()
            ds = Dataset(file)
            for key, val in ds.variables.items():
                if key not in df_year:
                    df_year[key] = val[...].flatten()
            df_years.append(df_year)
            print(os.path.basename(file),'done')


        ###----------PROCESSING----------###
        df_station = pd.concat(df_years)
        df_station['epoch'] = pd.to_datetime({ #Time in UTC, I believe
            'year':   df_station['time_yr'],
            'month':  df_station['time_mo'],
            'day':    df_station['time_dy'],
            'hour':   df_station['time_hr'],
            'minute': df_station['time_mt'],
            'second': df_station['time_sc'],
        })

        df_station.drop(columns=['time_yr','time_mo','time_dy','time_dy','time_hr','time_mt','time_sc'], inplace=True)
        df_station.set_index('epoch', inplace=True)

        df_station.rename(columns=all_mappings, inplace=True)
        df_station.attrs.update({'units': {}})

        for _, col in bfield_cols.items():
            df_station.attrs['units'][col] = 'nT'

        for _, col in position_cols.items():
            df_station[col] = np.radians(df_station[col])
            df_station.attrs['units'][col] = 'rad'

        df_station.attrs['units']['duration'] = 's'
        df_station.attrs['units']['MLT'] = 'h'

        # All time independent
        df_station.attrs['glat']     = np.degrees(df_station['glat'].iloc[0])
        df_station.attrs['glon']     = np.degrees(df_station['glon'].iloc[0])
        df_station.attrs['id']       = df_station['id'].iloc[0]
        df_station.attrs['duration'] = df_station['duration'].iloc[0]
        df_station.attrs['count']    = df_station['count'].iloc[0]

        df_station = df_station[['MLT', 'B_n_GEO', 'B_e_GEO', 'B_z_GEO', 'B_n_NEZ', 'B_e_NEZ', 'B_z_NEZ', 'mdecl', 'mcolat', 'chi_s']]

        df_station.insert(1, 'B_mag', (df_station[[f'B_{c}_NEZ' for c in ('n','e','z')]].pow(2).sum(axis=1)) ** 0.5)
        df_station.insert(2, 'H_mag', (df_station[[f'B_{c}_NEZ' for c in ('n','e')]].pow(2).sum(axis=1)) ** 0.5) # Horizontal intensity, not H-field
        df_station.attrs['units']['H_mag'] = 'nT'

        print(f'Writing {station}....')
        direc = get_proc_directory('supermag', dtype=station, resolution='raw', create=True)
        attributes = {'sample_interval': '1min', 'time_col': 'epoch'}
        write_to_cdf(df_station, directory=direc, file_name=f'{station}_raw', attributes=attributes, time_col='epoch', reset_index=True)


def convert_supermag_gse(*stations):

    ###----------ROTATING FROM GEO TO GSE----------###

    for station in stations:

        try:
            df_station = import_processed_data('supermag', dtype=station, resolution='raw')
        except:
            continue

        print(f'Converting {station}.')
        direc = get_proc_directory('supermag', dtype=station, resolution='gse', create=True)

        for (year, month), mag_month in (df_station.groupby([df_station.index.year, df_station.index.month], sort=True)):
            print(f'{year} - {month:02d}')

            convert_GEO_to_GSE(mag_month)

            write_to_cdf(mag_month, directory=direc, file_name=f'{station}_gse_{year}-{month:02d}', attributes={'time_col': 'epoch'}, time_col='epoch', reset_index=True)


def convert_supermag_agse(*stations, lag='17min', resolution='1min'):
    """
    This method rotates the coordinates based on the aberration of the solar wind (mainly due to Earth's orbital motion')
    DP2 current system can be much more angled than this (up to 30 deg)
    """

    omni = import_processed_data('omni', resolution=resolution)
    omni = omni.shift(freq=lag) # use 17-minute delay for BS to PC lag

    ###----------ROTATING FROM GSE TO AGSE----------###

    for station in stations:

        try:
            df_station = import_processed_data('supermag', dtype=station, file_name=f'{station}_gse')
        except:
            continue

        ###

            # if resolution == 5 min, need to resample the data:

        ###

        convert_GSE_to_aGSE(df_station, omni)

        print(f'Writing {station}....')
        direc = get_proc_directory('supermag', dtype=station, create=True)
        attributes = {'sample_interval': resolution, 'time_col': 'epoch'}
        write_to_cdf(df_station, directory=direc, file_name=f'{station}_agse_{resolution}', attributes=attributes, time_col='epoch', reset_index=True)


def project_supermag_optimum(*stations):
    """
    DP2 current system can be angled up to 30 degrees, based on upstream conditions and time of year.
    Optimum direction is perpendicular to this.
    """


    ## change to be the optimum direction based on DP2 maps
    ## need to find a way to do this ideally not using OMNI data


    ###----------ROTATING FROM GSE TO AGSE----------###

    for station in stations:

        try:
            df_station = import_processed_data('supermag', dtype=station, file_name=f'{station}_gse')
        except:
            continue


        # e.g. construct vector (cos30,sin30) and dot with each row

        print(f'Writing {station}....')
        direc = get_proc_directory('supermag', dtype=station, create=True)
        attributes = {'sample_interval': resolution, 'time_col': 'epoch'}
        write_to_cdf(df_station, directory=direc, file_name=f'{station}_agse_{resolution}', attributes=attributes, time_col='epoch', reset_index=True)