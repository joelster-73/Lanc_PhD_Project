# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 12:50:42 2025

@author: richarj2
"""
import os
import glob
import numpy as np
import pandas as pd
import warnings

from netCDF4 import Dataset

from ..reading import import_processed_data
from ..writing import write_to_cdf
from ...config import get_luna_directory, get_proc_directory
from ...coordinates.magnetic import convert_GEO_to_GSE, convert_GSE_to_GSM_with_angles


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

        print('Uncertainties.')
        if False: # excluding for now
            for coords in ('GEO','NEZ'):
                sigma_nez = calc_supermag_uncertainty(df_station, coords=coords)
                for comp in ('n','e','z'):
                    print(coords,comp)
                    idx = df_station.columns.get_loc(f'B_{comp}_{coords}')
                    df_station.insert(idx+1, f'B_{comp}_{coords}_unc', sigma_nez)
                    df_station.attrs['units'][f'B_{comp}_{coords}_unc'] = 'nT'

            for quantity in ('B','H'):
                print(f'Propagating {quantity}')
                sigma = prop_supermag_uncertainty(df_station, quantity)
                idx = df_station.columns.get_loc(f'{quantity}_mag')
                df_station.insert(idx+1, f'{quantity}_mag_unc', sigma)
                df_station.attrs['units'][f'{quantity}_mag_unc'] = 'nT'

        print(f'Writing {station}....')
        direc = get_proc_directory('supermag', dtype=station, resolution='raw', create=True)
        attributes = {'sample_interval': '1min', 'time_col': 'epoch'}
        write_to_cdf(df_station, directory=direc, file_name=f'{station}_raw', attributes=attributes, reset_index=True)


def calc_supermag_uncertainty(df, coords='NEZ', comp='n', window_minutes=1440):
    """
    Taken from Gjerloev (2012) 10.1029/2012JA017683
    """

    # Select components
    Bx = df[f'B_n_{coords}']
    By = df[f'B_e_{coords}']
    Bz = df[f'B_z_{coords}']

    # Rolling 24h mean
    B = pd.concat([Bx, By, Bz], axis=1)
    Bbar = B.rolling(window_minutes, center=True, min_periods=1).mean()

    # Instantaneous variance v(t)
    k = 1 / (3 * window_minutes)
    v = k * ((B - Bbar) ** 2).sum(axis=1)

    mcolat = df['mcolat']
    mlat = np.pi/2 - mcolat
    mlat_deg = np.degrees(mlat)

    if np.sum(mlat_deg<=60)>0:
        warnings.warn('Memory term d(t) is not implemented (correctly).')

        # Memory term d(t)

        tau_max = 8 * window_minutes  # 8 days in minutes
        k = 1 / (tau_max)
        w = np.cos(np.pi * np.arange(tau_max) / (2 * tau_max))
        v_pad = np.pad(v.values, (tau_max, 0), mode='edge')
        d = np.zeros_like(v)

        f = 0
        if comp=='n':
            f = np.abs(np.cos(mlat))
        elif comp=='z':
            f = np.abs(np.sin(mlat))

        for i in range(len(v)):
            # Only for |magnetic latitude| <= 60 deg
            if abs(mlat_deg.iloc[i]) <= 60:
                d[i] = np.sum(w * v_pad[i:i + tau_max]) # I don't think this is correct
            # else d[i] remains 0

        d *= f

        # Total uncertainty
        U = v + d

        return np.sqrt(U)

    return np.sqrt(v)


def prop_supermag_uncertainty(df, quantity='H', coords='NEZ'):

    Bx = df[f'B_n_{coords}']
    By = df[f'B_e_{coords}']
    Bz = df[f'B_z_{coords}']

    sx = df[f'B_n_{coords}_unc']
    sy = df[f'B_e_{coords}_unc']
    sz = df[f'B_z_{coords}_unc']

    mag = df[f'{quantity}_mag']

    sigma_sum = (Bx*sx)**2 + (By*sy)**2

    if quantity=='B':
        sigma_sum += (Bz*sz)**2

    return sigma_sum**(0.5)/mag


def convert_supermag_gse(*stations):

    ###----------ROTATING FROM GEO TO GSE----------###

    for station in stations:

        try:
            df_station = import_processed_data('supermag', dtype=station, resolution='raw')
        except:
            continue

        glat, glon = df_station.attrs.get('glat',77.46999), df_station.attrs.get('glon',290.76996) # default THL

        print(f'Converting {station}.')
        direc = get_proc_directory('supermag', dtype=station, resolution='gse', create=True)

        for (year, month), mag_month in (df_station.groupby([df_station.index.year, df_station.index.month], sort=True)):
            print(f'{year} - {month:02d}')

            convert_GEO_to_GSE(mag_month, glat, glon, param='H') # only care about horizontal component

            write_to_cdf(mag_month, directory=direc, file_name=f'{station}_gse_{year}-{month:02d}', attributes={'time_col': 'epoch'}, reset_index=True)


def convert_supermag_gsm(*stations):

    ###----------ROTATING FROM GSE TO GSM USING OMNI----------###

    for station in stations:

        try:
            df_station = import_processed_data('supermag', dtype=station, resolution='gse')
        except:
            continue

        print(f'Converting {station}.')
        direc = get_proc_directory('supermag', dtype=station, resolution='gsm', create=True)

        for year, mag_year in (df_station.groupby(df_station.index.year, sort=True)):
            print(f'{year}')

            df_omni = import_processed_data('omni', resolution='1min', year=year)

            df_year = convert_GSE_to_GSM_with_angles(mag_year, vectors=[[f'H_{c}_GSE' for c in ('x','y','z')]], df_coords=df_omni, include_unc=('B_x_GSE_unc' in mag_year))

            new_cols = ['H_y_GSM','H_z_GSM']
            mag_year[new_cols] = df_year[new_cols]

            for col in new_cols:
                mag_year.attrs['units'][col] = 'nT'

            write_to_cdf(mag_year, directory=direc, file_name=f'{station}_gsm_{year}', attributes={'time_col': 'epoch'}, reset_index=True)
