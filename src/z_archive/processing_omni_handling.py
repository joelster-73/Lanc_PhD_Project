# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 15:34:51 2026

@author: richarj2
"""

import numpy as np
import pandas as pd

from datetime import datetime, timedelta

from ...coordinates.magnetic import calc_B_GSM_angles

def extract_omni_data_old(asc_file, variables):
    """
    For the old version of OMNI that is in ascii files
    The newer version is in .list files
    And uses definitive WIND data
    The columns are also in a different order
    """

    # Load data from the ASCII file
    data_set = np.array(np.loadtxt(asc_file))  # each element is a row; one row per minute

    # Initialize DataFrame to store extracted data
    df = pd.DataFrame()

    # Add spacecraft IDs for filtering
    df['imf_sc']    = data_set[:, 4]
    df['plasma_sc'] = data_set[:, 5]

    # Iterate over requested variables and extract data
    for var in variables:
        if var == 'time':
            years = data_set[:, 0].astype(int)
            days = data_set[:, 1]
            hours = data_set[:, 2]
            minutes = data_set[:, 3]

            dates = [
                datetime(year, 1, 1) + timedelta(days=day - 1, hours=hour, minutes=minute)
                for year, day, hour, minute in zip(years, days, hours, minutes)
            ]

            # Convert datetime to CDF epoch format
            df['epoch'] = dates

        elif var == 'B_field':
            # Magnetic field data (nT)
            df['B_avg']   = data_set[:, 13]  # nT
            df['B_x_GSE'] = data_set[:, 14]
            df['B_y_GSE'] = data_set[:, 15]
            df['B_z_GSE'] = data_set[:, 16]
            df['B_y_GSM'] = data_set[:, 17]
            df['B_z_GSM'] = data_set[:, 18]

            # Set to NaN for bad data (sc_id 99)
            df.loc[df['imf_sc'] == 99, ['B_avg', 'B_x_GSE', 'B_y_GSE', 'B_z_GSE', 'B_y_GSM', 'B_z_GSM']] = np.nan

            gsm = calc_B_GSM_angles(df, time_col='epoch')
            df = pd.concat([df, gsm], axis=1)

        elif var == 'pressure':
            # Flow pressure calculation: (2*10**-6)*Np*Vp**2 nPa
            df['p_flow'] = data_set[:, 27]  # nPa

            # Set to NaN for bad data (sc_id 99)
            df.loc[df['plasma_sc'] == 99, 'p_flow'] = np.nan

        elif var == 'density':
            df['n_p'] = data_set[:, 25]  # n/cc

            # Set to NaN for bad data (sc_id 99)
            df.loc[df['plasma_sc'] == 99, 'n_p'] = np.nan

        elif var == 'velocity':
            df['v_flow']  = data_set[:, 21]  # km/s - something odd
            df['v_x_GSE'] = data_set[:, 22]  # km/s
            df['v_y_GSE'] = data_set[:, 23]  # km/s
            df['v_z_GSE'] = data_set[:, 24]  # km/s

            # Set to NaN for bad data (sc_id 99)
            df.loc[df['plasma_sc'] == 99, ['v_flow', 'v_x_GSE', 'v_y_GSE', 'v_z_GSE']] = np.nan

        elif var == 'satellite':
            # Satellite ID and position data
            df['sc_id']   = data_set[:, 5]
            df['r_x_GSE'] = data_set[:, 31]
            df['r_y_GSE'] = data_set[:, 32]
            df['r_z_GSE'] = data_set[:, 33]

            # Set to NaN for bad data (sc_id 99)
            df.loc[df['plasma_sc'] == 99, ['sc_id', 'r_x_GSE', 'r_y_GSE', 'r_z_GSE']] = np.nan

        elif var == 'bow_shock_nose':
            df['r_x_BSN'] = data_set[:, 34]
            df['r_y_BSN'] = data_set[:, 35]
            df['r_z_BSN'] = data_set[:, 36]

            # Set to NaN for bad data (sc_id 99)
            df.loc[df['imf_sc'] == 99, ['r_x_BSN', 'r_y_BSN', 'r_z_BSN']] = np.nan

        elif var == 'propagation':
            # Satellite ID and position data
            df['prop_time_s'] = data_set[:, 9]

            # Set to NaN for bad data (sc_id 99)
            df.loc[df['imf_sc'] == 99, ['prop_time_s']] = np.nan

        else:
            raise ValueError(f'Unknown variable {var}')

    # Drop spacecraft IDs as they are no longer needed
    df = df.drop(columns=['imf_sc', 'plasma_sc'])
    return df