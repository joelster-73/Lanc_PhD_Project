import numpy as np

from ..processing.writing import write_to_cdf
from ..processing.dataframes import merge_dataframes


def analyse_merged(df, func, output_file=None, **kwargs):
    """
    Applies a specified function to a DataFrame and optionally writes the result to a CDF file.

    This function takes a DataFrame `df` and applies the provided function `func` to it. Additional
    keyword arguments can be passed to `func`. If an `output_file` is provided, the resulting DataFrame
    will be saved to that file in CDF format, overwriting any existing file.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to be processed by the function `func`.

    func : function
        A function to be applied to the DataFrame `df`. It must accept `df` as its first argument
        and can take additional keyword arguments as specified by `kwargs`.

    output_file : str, optional
        The path to the CDF file where the resulting DataFrame will be saved. If not provided, the
        processed DataFrame will be returned instead of being saved.

    **kwargs : additional arguments, optional
        Additional keyword arguments passed to `func`.

    Returns
    -------
    pandas.DataFrame or None
        If `output_file` is not provided, the processed DataFrame is returned. Otherwise, the result
        is written to the specified CDF file, and the function returns `None`.
    """
    new_df = func(df, **kwargs)
    if not output_file:
        return new_df
    write_to_cdf(new_df, output_file, overwrite=True)


def difference_columns(df, c1, c2):
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
    data = df[c1] - df[c2]
    if df.attrs['units'][c1] == 'rad':
        data = (data + np.pi) % (2 * np.pi) - np.pi
    elif df.attrs['units'][c1] in ('deg','°'):
        data = (data + 180) % 360 - 180
    return data

def difference_series(data1, data2, data_col):
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
    differences = merge_dataframes(data1, data2, '1', '2')
    differences['diff'] = difference_columns(differences, data_col+'_1', data_col+'_2')
    return differences

def difference_columns_dfs(df1, df2, data_col1, data_col2=None):
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
    data_col2 = data_col1 if data_col2 is None else data_col2
    differences = merge_dataframes(df1, df2, '1', '2')
    return difference_columns(differences, data_col2+'_1', data_col2+'_2')




