import numpy as np

def difference_columns(df, c1, c2, diff_type='absolute'):

    series1 = df.loc[:,c1]
    series2 = df.loc[:,c2]
    return difference_series(series1,series2,diff_type)

def difference_dataframes(df1, df2, c1, c2, diff_type='absolute'):

    series1 = df1.loc[:,c1]
    series2 = df2.loc[:,c2]
    return difference_series(series1,series2,diff_type)


def difference_series(series1, series2, diff_type='absolute'):
    """
    Computes the difference between two columns in a DataFrame, optionally
    adjusting for angular differences (radians).

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the columns `c1` and `c2` to be subtracted.

    c1 : str
        The name of the first column in the DataFrame.

    c2 : str
        The name of the second column in the DataFrame.

    Returns
    -------
    data : pandas.Series
        A Series containing the difference between the values of `c1` and `c2`.
        If the unit of `c1` is 'rad' of 'deg', the result will be adjusted to handle
        angular differences within the range [-π, π) or [-180, 180).
    """
    data = series1 - series2
    data.name = f'{series1.name} - {series2.name}'
    if diff_type=='relative':
        data /= series2
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.name = f'({series1.name} - {series2.name}) / {series2.name}'

    if series1.attrs['units'][series1.name] == 'rad':
        data = (data + np.pi) % (2 * np.pi) - np.pi
    elif series1.attrs['units'][series1.name] in ('deg','°'):
        data = (data + 180) % 360 - 180

    data.attrs = {}
    data.attrs['units'] = {data.name: series1.attrs['units'][series1.name]}

    return data




