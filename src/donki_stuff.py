# -*- coding: utf-8 -*-
"""
Created on Thu May 15 08:44:01 2025

@author: richarj2
"""

# %%
import requests
import pandas as pd


# Define the API endpoint and parameters
url = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/IPS"
params = {
    "startDate": "1995-01-01",
    "endDate": "2024-12-31"
}

# Make the GET request
response = requests.get(url, params=params)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Create a DataFrame from the data
    df_donki_raw = pd.DataFrame(data)
    print("DataFrame created:")
    print(df_donki_raw.head())  # Display the first few rows of the DataFrame
else:
    print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")

df_donki = df_donki_raw[['location','eventTime','instruments']]
print(df_donki)

def extract_spacecraft(instruments):
    return list(set(entry['displayName'].split(":")[0] for entry in instruments))

df_donki['spacecraft'] = df_donki['instruments'].apply(extract_spacecraft)
df_donki.drop(columns=['instruments'],inplace=True)
df_donki['eventTime'] = pd.to_datetime(df_donki['eventTime']).dt.strftime('%Y-%m-%d %H:%M:%S')
df_donki.set_index('eventTime',inplace=True)

df_donki_earth = df_donki[df_donki['location']=='Earth'] # all ace and/or dscover
df_donki_else = df_donki[df_donki['location']!='Earth'] # all but one aren't ace or dscover

# %%
def find_close_matches(df1, df2, tolerance='10min'):
    df1.index = pd.to_datetime(df1.index)
    df2.index = pd.to_datetime(df2.index)
    df1_matched = []
    df2_matched = []

    for time1, row1 in df1.iterrows():
        close_times = df2.index[(df2.index >= time1 - pd.Timedelta(tolerance)) &
                                (df2.index <= time1 + pd.Timedelta(tolerance))]
        for time2 in close_times:
            row2 = df2.loc[time2]
            # Check if spacecraft from df2 matches any in df1
            if row2['spacecraft'].upper() in [sc.upper() for sc in row1['spacecraft']]:
                df1_matched.append(time1)
                df2_matched.append(time2)


    return df1.loc[df1_matched], df2.loc[df2_matched]

# Find common rows
common_df1, common_df2 = find_close_matches(df_donki_earth, shocks)

print('Common:',len(common_df1))

# Rows unique to df1
unique_df1 = df_donki_earth.drop(common_df1.index)
print('Unique donki:',len(unique_df1))

# Rows unique to df2
unique_df2 = shocks.drop(common_df2.index)
print('Unique CFA:',len(unique_df2))

counts = {}
for element in shocks['spacecraft']:
    counts[element] = counts.get(element, 0) + 1
print(counts)

counts_donki = {}
for element in ('ACE','DSCOVR'):
    counts_donki[element] = df_donki_earth['spacecraft'].apply(lambda x: element in x).sum()
counts_donki['both'] = df_donki_earth['spacecraft'].apply(lambda x: 'ACE' in x and 'DSCOVR' in x).sum()

print(counts_donki)
