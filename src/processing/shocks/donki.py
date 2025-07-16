# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 14:19:00 2025

@author: richarj2
"""

import requests
import pandas as pd

def get_donki_shocks(start_date='1995-01-01',end_date='2024-12-31',print_stuff=False):

    # Define the API endpoint and parameters
    url = 'https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/IPS'
    params = {
        'startDate': start_date,
        'endDate': end_date
    }
    # Make the GET request
    response = requests.get(url, params=params)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        df_donki = pd.DataFrame(data)
    else:
        print(f'Failed to fetch data. HTTP Status Code: {response.status_code}')
        return pd.DataFrame()

    if print_stuff:
        print('DataFrame created:')
        print(df_donki.head())  # Display the first few rows of the DataFrame

    df_donki = df_donki[['location','eventTime','instruments']]

    def extract_spacecraft(instruments):
        return list(set(entry['displayName'].split(':')[0] for entry in instruments))

    df_donki['spacecraft_list'] = df_donki['instruments'].apply(extract_spacecraft)
    df_donki.drop(columns=['instruments'],inplace=True)
    df_donki['eventTime'] = pd.to_datetime(df_donki['eventTime'])
    df_donki.rename(columns={'eventTime': 'epoch'},inplace=True)
    df_donki.set_index('epoch',inplace=True)
    # Need to remove the "time-awareness" - I'm not sure why
    df_donki.index = df_donki.index.tz_localize(None)

    df_donki = df_donki[df_donki['location']=='Earth'][['spacecraft_list']] # all ace and/or dscover
    #df_donki_else = df_donki[df_donki['location']!='Earth'] # all but one aren't ace or dscover

    # Shock times are given to nearest minute
    df_donki['time_s_unc'] = 30

    # Option 1
    # # One row only, taking the first spacecraft
    df_donki['spacecraft'] = df_donki['spacecraft_list'].apply(lambda x: x[0].lower() if isinstance(x, list) and x else None)
    df_donki.drop(columns=['spacecraft_list'],inplace=True)

    # Option 2
    # "Explodes" the dataframe, a row created for each element in the list
    # df_donki = df_donki.explode('spacecraft_list')
    # df_donki.rename(columns={'spacecraft_list':'spacecraft'},inplace=True)
    # df_donki['spacecraft'] = df_donki['spacecraft'].str.lower()

    # For both options
    df_donki['spacecraft'] = df_donki['spacecraft'].str.replace('dscovr', 'dsc', regex=False)
    return df_donki


def combine_cfa_donki(cfa_shocks, donki_shocks=None):
    attrs_dict = cfa_shocks.attrs

    cfa_shocks = cfa_shocks.copy()
    cfa_shocks = cfa_shocks[['time_s_unc','spacecraft']]
    cfa_shocks['source'] = 'C'

    if donki_shocks is None:
        donki_shocks = get_donki_shocks()
    else:
        donki_shocks = donki_shocks.copy()

    donki_shocks['source'] = 'D'

    shocks = pd.concat([cfa_shocks,donki_shocks]).sort_index()

    shocks['rounded_index'] = shocks.index.round('1min')
    shocks.reset_index(inplace=True)

    def custom_deduplication(group):
        c_rows = group[group['source'] == 'C'] # prefer C columns - from CFA database
        return c_rows.iloc[0] if not c_rows.empty else group.iloc[0]

    shocks = shocks.groupby(['rounded_index', 'spacecraft'], group_keys=False).apply(custom_deduplication)
    shocks.set_index('epoch',inplace=True)
    shocks.drop(columns=['rounded_index'],inplace=True)

    # Removes shocks with exact same timestamp - likely duplicate entries from DONKI, preferring the first spacecraft retrieved
    shocks = shocks[~shocks.index.duplicated(keep='first')]

    shocks.attrs = attrs_dict
    for col in list(attrs_dict['units']):
        if col not in shocks:
            del shocks.attrs['units'][col]
    shocks.attrs['units']['source'] = 'STRING'

    return shocks