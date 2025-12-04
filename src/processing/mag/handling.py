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

from ...config import get_luna_directory


# All nT
bfield_cols = {'dbn_geo': 'B_n_GEO', # Magnetic field north component, geographic coordinates
               'dbe_geo': 'B_e_GEO', # Magnetic field east component, geographic coordinates
               'dbz_geo': 'B_z_GEO', # Magnetic field vertical component, geographic coordinates
               'dbn_nez': 'B_n_NEZ', # Magnetic field north component, NEZ coordinates
               'dbe_nez': 'B_e_NEZ', # Magnetic field east component, NEZ coordinates
               'dbz_nez': 'B_z_NEZ', # Magnetic field vertical component, NEZ coordinates
}

# all degrees
position_cols = {'decl'  : 'D_m',      # Magnetic Declination
                 'mcolat': 'theta_m',  # Magnetic Colatitude
                 'glat'  : 'phi_m',    # Geographic Latitude
                 'glon'  : 'lambda_m', # Geographic Longtiude
                 'sza'   : 'chi_s',    # Solar Zenith Angle
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


        df_station = pd.concat(df_years)

        df_station['epoch'] = pd.to_datetime({
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


        df_station = df_station[['MLT', 'B_n_GEO', 'B_e_GEO', 'B_z_GEO', 'B_n_NEZ', 'B_e_NEZ', 'B_z_NEZ', 'D_m', 'theta_m', 'phi_m', 'lambda_m', 'chi_s', 'id', 'duration', 'count']]

        return df_station


