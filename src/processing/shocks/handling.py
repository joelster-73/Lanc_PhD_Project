import os
import re
import requests

import pandas as pd

from bs4 import BeautifulSoup

from .utils import calculate_shock_datetime, split_uncertainty
from ..utils import create_directory
from ..writing import write_to_cdf
from ..dataframes import add_df_units


def get_all_shocks(tag_strings, tag_labels, tags_uncertainty, tags_up_dw, shock_directory, spacecraft=('wind','ace','dsc'), year=None, time_col='epoch', overwrite=True):

    all_shocks = []
    for sc in spacecraft:
        if not year:
            all_years = get_shock_years(sc)
        else:
            all_years = [year]

        yearly_list = []
        for y in all_years:
            yearly_list.append(get_shocks_for_year(y, tag_strings, tag_labels, tags_uncertainty, tags_up_dw, sc))

        df_sc = pd.concat(yearly_list, axis=0) # axis=0 stacks rows together
        try:
            df_sc.insert(2,'spacecraft',sc)
            all_shocks.append(df_sc)
        except:
            print(f'No shocks for {sc}.')

    df_shocks = pd.concat(all_shocks, axis=0)

    df_shocks.sort_values('time',inplace=True)
    df_shocks.rename(columns={'time': time_col}, inplace=True)
    add_df_units(df_shocks)

    create_directory(shock_directory)
    output_file = os.path.join(shock_directory, 'cfa_shocks.cdf')
    attributes = {'time_col': time_col}
    write_to_cdf(df_shocks, output_file, attributes, overwrite)

def get_shock_years(spacecraft='wind'):
    """

    """
    # Get the overview page for the year
    cfa_url = 'https://lweb.cfa.harvard.edu/shocks'
    if spacecraft == 'wind':
        spacecraft_url = f'{cfa_url}/wi_data'
    elif spacecraft == 'ace':
        spacecraft_url = f'{cfa_url}/ac_master_data'
    elif spacecraft == 'dsc':
        spacecraft_url = f'{cfa_url}/dsc_data'
    else:
        raise ValueError(f'{spacecraft} not valid.')

    response = requests.get(spacecraft_url)

    # Check if the request was successful
    if response.status_code != 200:
        print(f'Failed to retrieve the overview page for {spacecraft}.')
        return pd.DataFrame()

    # Parse the overview page with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all shock links on this page (each link corresponds to a shock event)
    shock_years = []
    for link in soup.find_all('a', class_='leftnav', href=True):
        href = link['href']
        match = re.search(r'19[8-9][0-9]|20[0-2][0-9]|2040', href)
        if match:
            shock_years.append(int(match.group(0)))

    return shock_years

def get_shocks_for_year(year, tag_strings, tag_labels, tags_uncertainty, tags_up_dw, spacecraft='wind'):
    """
    Scrapes shock data from CFA Shock Database for a given year and returns it as a DataFrame.

    Parameters:
        year (int): The year to scrape shocks for (e.g., 2001).

    Returns:
        pd.DataFrame: A DataFrame containing the shock data with columns: Year, Month, Day, UT, X, Y, Z, Type.
    """


    # Get the overview page for the year
    cfa_url = 'https://lweb.cfa.harvard.edu/shocks'
    if spacecraft == 'wind':
        spacecraft_url = f'{cfa_url}/wi_data'
        overview_url = f'{spacecraft_url}/wi_{year}.html'
    elif spacecraft == 'ace':
        spacecraft_url = f'{cfa_url}/ac_master_data'
        overview_url = f'{spacecraft_url}/ac_master_{year}.html'
    elif spacecraft == 'dsc':
        spacecraft_url = f'{cfa_url}/dsc_data'
        overview_url = f'{spacecraft_url}/dsc_data_{year}.html'

    response = requests.get(overview_url)

    # Check if the request was successful
    if response.status_code != 200:
        print(f'Failed to retrieve the overview page for {spacecraft} {year}')
        return pd.DataFrame()

    # Parse the overview page with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all shock links on this page (each link corresponds to a shock event)
    shock_links = []
    for link in soup.find_all('a', href=True):
        # Example link: './00193/wi_00193.html'
        href = link['href']
        if href.startswith('./'):
            shock_links.append(href)

    if len(shock_links) == 0:
        print ('No shocks found for {spacecraft} {year}.')
        return pd.DataFrame()

    # Scrape details for each shock by visiting its event page
    shock_data = []

    for shock_link in shock_links:
        # Get the full URL for the shock event
        shock_url = f'{spacecraft_url}{shock_link[1:]}'

        shock_response = requests.get(shock_url)
        if shock_response.status_code != 200:
            print(f'Failed to retrieve data for shock {shock_url}')
            continue

        # Parse the event page with BeautifulSoup
        shock_soup = BeautifulSoup(shock_response.text, 'html.parser')

        # Extract the details for the shock

        values=[]

        cells = shock_soup.find_all('td')
        #print(cells)
        for pattern in tag_strings:

            # Match exact cell text (case-insensitive)
            cell = shock_soup.find('td',string=pattern)

            if cell is None:
                for a_cell in cells:
                    if pattern in a_cell:
                        cell = a_cell
            if cell:
                # Extract the next cell's text, normalising whitespace and line breaks
                values.append(cell.find_next('td').get_text(separator=' ').strip())
                if pattern in tags_up_dw:
                    values.append(cell.find_next('td').find_next('td').get_text(separator=' ').strip())
            else:
                print(f'Pattern not found: {pattern}')
                values.append(None)

            if pattern == 'Method selected':
                method = values[-1]
                if method:
                    method_cells = shock_soup.find_all('td', string=method)
                    # First row is the shock normal Nx, Ny, Nz
                    # Second row is the key shock parameters ThetaBn, Shock Speed, Compression
                    normal_row = method_cells[0]
                    Nx = normal_row.find_next('td')
                    Ny = Nx.find_next('td')
                    Nz = Ny.find_next('td')

                    params_row = method_cells[1]
                    speed = params_row.find_next('td').find_next('td')

                    for quantity in (Nx,Ny,Nz,speed):
                        values.append(quantity.get_text(separator=' ').strip())

        shock_data.append(values)

    # Processes shock data
    df = pd.DataFrame(shock_data, columns=tag_labels)

    results = df.apply(
        lambda row: calculate_shock_datetime(row['year'], row['day'], row['time_of_day']),
        axis=1
    )
    arrival_times, arrival_uncs = zip(*results)
    df.insert(0,'time',arrival_times)
    df.drop(columns=['year','day','time_of_day'],inplace=True)
    df.insert(1,'time_s_unc',arrival_uncs)
    df['process_time'] = pd.to_datetime(df['process_time'], format='%a %b %d %H:%M:%S %Y')

    # Removes duplicates in list and keeps most recent processed
    df = df.sort_values(by=['time', 'process_time'], ascending=[True, False])
    df = df.drop_duplicates(subset='time', keep='first')
    df.drop(columns=['process_time'],inplace=True)

    for col in tags_uncertainty:
        split_results = df.apply(
            lambda row: split_uncertainty(row[col]),
            axis=1
        )
        values, uncs = zip(*split_results)
        df[col] = values
        df[f'{col}_unc'] = uncs
        if col == 'delay_s':
            df[col] *= 60 # convert mins to secs
            df[f'{col}_unc'] *= 60

    # All other columns should be floats
    not_float = ('time','spacecraft','type','method')
    df = df.apply(lambda col: col.astype(float) if col.name not in not_float else col)

    return df

