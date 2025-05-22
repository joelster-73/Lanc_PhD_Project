import os

import pandas as pd

from spacepy import pycdf

from .utils import calculate_region_duration
from ..writing import write_to_cdf
from ..handling import get_cdf_file
from ..dataframes import add_df_units


def extract_crossing_data(cdf_file, variables):

    # Initialise a dictionary to store the data
    data_dict = {}

    # Load the CDF file (auto closes)
    with pycdf.CDF(cdf_file) as cdf:

        # Loop through the dictionary of variables and extract data
        for var_name, var_code in variables.items():

            data = cdf[var_code][...]  # Extract the data using the CDF variable code
            if data.ndim == 2:
                data = data[:, 0]
            data_dict[var_name] = data  # pycdf extracts as datetime, no conversion from epoch needed

    return data_dict

def process_crossing_file(directory, data_directory, variables, time_col='epoch', overwrite=True):

    cdf_file = get_cdf_file(directory)

    directory_name = os.path.basename(os.path.normpath(directory))

    try:  # Bad data check
        df = pd.DataFrame(extract_crossing_data(cdf_file, variables))
    except (AttributeError, ValueError):
        print('Crossings not processed.')
    else:
        try:
            mapping_dict = dict(zip(df['loc_num'].drop_duplicates(), df['loc_name'].drop_duplicates()))
        except:
            print('Cannot make crossing mappings.')
        calculate_region_duration(df)
        add_df_units(df)

        output_file = os.path.join(data_directory, f'{directory_name}.cdf')
        attributes = {'time_col': time_col, 'crossings': mapping_dict}
        write_to_cdf(df, output_file, attributes, overwrite)

        print('Crossings processed.')


